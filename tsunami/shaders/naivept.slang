// ─── OpenPBR texture-LUT mode ─────────────────────────────────────────────────
#define OPENPBR_LANGUAGE_TARGET_SLANG 1
#define OPENPBR_USE_TEXTURE_LUTS      1

static const uint NUM_LUTS         = 8;
static const uint MAX_MAT_TEXTURES = 256;

[[vk::binding(3, 0)]] SamplerState g_lut_sampler;
[[vk::binding(9,  0)]] Texture2D<float4> openpbr_luts_2d[NUM_LUTS];
[[vk::binding(10, 0)]] Texture3D<float4> openpbr_luts_3d[NUM_LUTS];
[[vk::binding(12, 0)]] SamplerState g_material_sampler;

float4 openpbr_sample_2d_impl(int lut_id, float2 uv) {
    return openpbr_luts_2d[lut_id].SampleLevel(g_lut_sampler, uv, 0.f);
}
float4 openpbr_sample_3d_impl(int lut_id, float3 uvw) {
    return openpbr_luts_3d[lut_id].SampleLevel(g_lut_sampler, uvw, 0.f);
}
#define OPENPBR_SAMPLE_2D_TEXTURE(lut_id, uv)  openpbr_sample_2d_impl((lut_id), (uv))
#define OPENPBR_SAMPLE_3D_TEXTURE(lut_id, uvw) openpbr_sample_3d_impl((lut_id), (uvw))

#include "openpbr-bsdf/openpbr.h"

// ─── Push constants ───────────────────────────────────────────────────────────
struct PushConstants {
    uint   frame;
    uint   material_count;
    int    selected_mesh_index;
    uint   outline_width;
    int    debug_view_mode;
    uint   stage;   // 0 = visibility pass, 1 = path trace selected object
    uint   spp;
    uint   max_bounces;
    float4 outline_color;
    uint   enable_tonemapping;
    float  exposure_bias;
    uint   hipr_object_count;
    uint   hipr_top_k;
    int    hipr_render_rank;
    uint   hipr_incremental_sort;
    uint   hipr_clear_order;
    uint   hipr_vis_enable_tint;
    uint   hipr_vis_rainbow_tint;
    uint   hipr_reserved0;
    uint   hipr_frames_per_object;
    float  hipr_score_blend;
    float  hipr_vis_tint_strength;
    uint   skybox_enabled;
    uint   directional_light_enabled;
    float  sun_dir_x;
    float  sun_dir_y;
    float  sun_dir_z;
    float  sun_intensity;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> push;

// ─── Bindings ─────────────────────────────────────────────────────────────────
[[vk::binding(0,  0)]] RWTexture2D<float4>             output_image;
[[vk::binding(1,  0)]] RWTexture2D<float4>             accum_image;
[[vk::binding(2,  0)]] StructuredBuffer<GPUCamera>     camera_buf;
[[vk::binding(4,  0)]] StructuredBuffer<GPUMaterial>   materials_buf;
[[vk::binding(5,  0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(6,  0)]] StructuredBuffer<GPUMesh>       meshes_buf;
[[vk::binding(7,  0)]] StructuredBuffer<GPUVertex>     vertices_buf;
[[vk::binding(8,  0)]] StructuredBuffer<uint>          indices_buf;
[[vk::binding(11, 0)]] Texture2D<float4>               material_textures[MAX_MAT_TEXTURES];
[[vk::binding(13, 0)]] RWTexture2D<int>               object_id_image;

// ─── GPU structs — must match C++ layout exactly ──────────────────────────────
struct GPUCamera {
    float4 position;
    float4 target;
    float4 up;
    float4 fov_near_far;
};

struct GPUMaterial {
    // Base layer
    float4 base_color;
    float     base_metalness;
    float     base_diffuse_roughness;
    float     texture_weight;
    float     _pad0;

    // Specular layer
    float4 specular_color;
    float     specular_roughness;
    float     specular_ior;
    float     specular_anisotropy;
    float     _pad1;

    // Transmission
    float     transmission_weight;
    float     transmission_depth;
    float     transmission_scatter_anisotropy;
    float     geometry_opacity;
    float4 transmission_color;
    float4 transmission_scatter;

    // Emission
    float4 emission_color;
    float     emission_luminance;
    float     geometry_thin_walled; // store as float/int for alignment simplicity
    float     transmission_dispersion_scale;
    float     transmission_dispersion_abbe_number;

    // Coat
    float     coat_weight;
    float     coat_roughness;
    float     coat_ior;
    float     _pad4;
    float4 coat_color;

    // Fuzz/sheen
    float     fuzz_weight;
    float     fuzz_roughness;
    float     _pad5[2];
    float4 fuzz_color;

    // Thin film
    float     thin_film_weight;
    float     thin_film_ior;
    float     thin_film_thickness;
    float     _pad6;

    // Texture indices
    uint32_t albedo_tex_index;
    uint32_t normal_tex_index;
    uint32_t roughness_tex_index;
    uint32_t emissive_tex_index;
};

struct GPUVertex {
    float3 position;
    float  _pad0;
    float3 normal;
    float  _pad1;
    float4 tangent;
    float2 uv;
    float2 _pad2;
};

struct GPUMesh {
    float4x4 transform;
    float4x4 inverseTransform;
    uint     blasHandle_lo;
    uint     blasHandle_hi;
    int      matIndex;
    int      vertexOffset;
    int      indexOffset;
    int      indexCount;
    float    _pad[3];
};

// ─── Constants ────────────────────────────────────────────────────────────────
static const float PI          = 3.14159265358979f;
static const float T_MIN       = 1e-3f;
static const float T_MAX       = 1e30f;
static const uint  NO_TEXTURE  = 0xFFFFFFFFu;

// ─── Light: directional sun ───────────────────────────────────────────────────
// Direction and intensity come from push constants (UI-controlled).
// g_sun_dir is the direction FROM the surface TO the light (positive Y = above).
static const float3 g_sun_color = float3(1.0f, 0.85f, 0.5f);

// ─── Types ────────────────────────────────────────────────────────────────────
struct Ray { float3 origin; float3 dir; };
struct Hit {
    float  t;
    float3 normal;
    float3 geom_normal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
    bool   front_face;
    int    matIndex;
    int    instanceID;
    int    primID;
};

// ─── Halton low-discrepancy jitter ───────────────────────────────────────────
// Radical inverse base 2 via bit reversal — O(1), exact.
float halton2(uint n) { return float(reversebits(n)) * (1.f / 4294967296.f); }

// Radical inverse base 3 — converges in ≤12 iterations for n < 3^12 ≈ 530k.
float halton3(uint n) {
    float f = 1.f, r = 0.f;
    while (n > 0u) { f /= 3.f; r += f * float(n % 3u); n /= 3u; }
    return r;
}

// Returns a jittered pixel offset in [0,1)^2.
// sample_index should be frame * SPP + spp_i so consecutive frames extend the
// sequence rather than restarting it, giving full low-discrepancy coverage over time.
// The per-pixel Cranley-Patterson offset decorrelates neighbouring pixels so they
// don't all sample the same directions simultaneously.
float2 haltonJitter(uint sample_index, uint2 pixel) {
    float ox = frac(float(pixel.x * 0x9E3779B9u) * (1.f / 4294967296.f));
    float oy = frac(float(pixel.y * 0x6C62272Eu) * (1.f / 4294967296.f));
    return float2(frac(halton2(sample_index) + ox),
                  frac(halton3(sample_index) + oy));
}

// ─── RNG ──────────────────────────────────────────────────────────────────────
struct RNG {
    uint state;
    static RNG make(uint2 pixel, uint frame) {
        RNG r;
        r.state = pixel.x ^ (pixel.y << 16) ^ (frame * 719393u);
        r.next(); // warm-up: advance past the raw seed to break inter-pixel/frame correlation
        return r;
    }
    [mutating] uint next() {
        state     = state * 747796405u + 2891336453u;
        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }
    [mutating] float  nextFloat() { return float(next()) * (1.f / 4294967296.f); }
    [mutating] float3 next3()     { return float3(nextFloat(), nextFloat(), nextFloat()); }
};

// ─── Geometry helpers ─────────────────────────────────────────────────────────
struct VertexAttribs {
    float3 normal;
    float2 uv;
    float4 tangent;
    float3 p0;
    float3 p1;
    float3 p2;
};

float3 safeNormalize(float3 v, float3 fallback) {
    float len2 = dot(v, v);
    if (len2 > 1.0e-12f) {
        return v * rsqrt(len2);
    }
    return fallback;
}

float3 buildOrthoTangent(float3 n) {
    // Pick an axis least aligned with n, then project to get a stable tangent.
    float3 axis = (abs(n.y) < 0.999f) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    return safeNormalize(cross(axis, n), float3(1.0f, 0.0f, 0.0f));
}

VertexAttribs fetchAttribs(uint instanceID, uint primID, float2 bary) {
    uint voff = (uint)meshes_buf[NonUniformResourceIndex(instanceID)].vertexOffset;
    uint ioff = (uint)meshes_buf[NonUniformResourceIndex(instanceID)].indexOffset;
    uint i0 = indices_buf[ioff + primID * 3 + 0];
    uint i1 = indices_buf[ioff + primID * 3 + 1];
    uint i2 = indices_buf[ioff + primID * 3 + 2];
    float w0 = 1.f - bary.x - bary.y;
    VertexAttribs a;
    a.normal = normalize(vertices_buf[voff+i0].normal*w0 +
                         vertices_buf[voff+i1].normal*bary.x +
                         vertices_buf[voff+i2].normal*bary.y);
    a.tangent = vertices_buf[voff+i0].tangent*w0 +
                vertices_buf[voff+i1].tangent*bary.x +
                vertices_buf[voff+i2].tangent*bary.y;
    a.uv = vertices_buf[voff+i0].uv*w0 +
           vertices_buf[voff+i1].uv*bary.x +
           vertices_buf[voff+i2].uv*bary.y;
    a.p0 = vertices_buf[voff+i0].position;
    a.p1 = vertices_buf[voff+i1].position;
    a.p2 = vertices_buf[voff+i2].position;
    return a;
}

// ─── Shadow helper ────────────────────────────────────────────────────────────
// Returns true if the ray from (origin + normal*eps) in direction dir is
// blocked before tmax.
bool is_occluded(float3 origin, float3 normal, float3 dir, float tmax) {
    RayDesc sd;
    sd.Origin    = origin + normal * 1e-3f;
    sd.Direction = dir;
    sd.TMin      = 0.f;
    sd.TMax      = tmax;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, sd);
    while (q.Proceed()) {}
    return q.CommittedStatus() != COMMITTED_NOTHING;
}

int traceOccluderInstance(float3 origin, float3 normal, float3 dir, float tmax) {
    RayDesc sd;
    sd.Origin    = origin + normal * 1e-3f;
    sd.Direction = dir;
    sd.TMin      = 0.f;
    sd.TMax      = tmax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, sd);
    while (q.Proceed()) {}

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        return int(q.CommittedInstanceID());
    }
    return -1;
}

int traceVisibleInstance(Ray ray) {
    RayDesc rd;
    rd.Origin    = ray.origin;
    rd.Direction = ray.dir;
    rd.TMin      = T_MIN;
    rd.TMax      = T_MAX;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, rd);

    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }

    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return -1;
    return int(q.CommittedInstanceID());
}

// ─── Scene intersection ───────────────────────────────────────────────────────
bool intersectScene(Ray ray, out Hit hit, out GPUMaterial mat) {
    hit = (Hit)0;
    mat = (GPUMaterial)0;

    RayDesc rd;
    rd.Origin    = ray.origin;
    rd.Direction = ray.dir;
    rd.TMin      = T_MIN;
    rd.TMax      = T_MAX;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, rd);

    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }

    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return false;

    hit.t          = q.CommittedRayT();
    uint instID    = q.CommittedInstanceID();
    uint primID    = q.CommittedPrimitiveIndex();
    float2 bary    = q.CommittedTriangleBarycentrics();
    bool queryFrontFace = q.CommittedTriangleFrontFace();

    VertexAttribs attr = fetchAttribs(instID, primID, bary);

    hit.uv         = attr.uv;
    hit.instanceID = int(instID);
    hit.primID     = int(primID);
    hit.matIndex   = meshes_buf[NonUniformResourceIndex(instID)].matIndex;
    mat = materials_buf[NonUniformResourceIndex(hit.matIndex)];

    // Vertex normals/tangents are in local (BLAS) space.
    // The TLAS instance transform moves positions to world space, so we must
    // bring the TBN basis to world space too:
    //   normals  → multiply by inverse-transpose of model matrix
    //             = mul(n_local, inverseTransform)  [row-vec × mat in Slang]
    //   tangents → multiply by the forward model matrix (no inverse needed)
    float3x3 M    = (float3x3) meshes_buf[NonUniformResourceIndex(instID)].transform;
    float3x3 Minv = (float3x3) meshes_buf[NonUniformResourceIndex(instID)].inverseTransform;

    float3 N = safeNormalize(mul(attr.normal, Minv), float3(0.0f, 1.0f, 0.0f)); // inverse-transpose
    float3 T = mul(attr.tangent.xyz, M); // forward transform
    float  handedness = (attr.tangent.w < 0.0f) ? -1.0f : 1.0f;
    if (determinant(M) < 0.0f) {
        handedness = -handedness;
    }

    float3 Ng_local = cross(attr.p1 - attr.p0, attr.p2 - attr.p0);
    float3 Ng_a = safeNormalize(mul(Ng_local, Minv), N);
    float3 Ng_b = safeNormalize(mul(Ng_local, M), N);
    bool   a_front = dot(ray.dir, Ng_a) < 0.0f;
    bool   b_front = dot(ray.dir, Ng_b) < 0.0f;
    float3 Ng_world = Ng_a;
    if (a_front != queryFrontFace && b_front == queryFrontFace) {
        Ng_world = Ng_b;
    } else if (!(a_front == queryFrontFace && b_front != queryFrontFace)) {
        // Tie-breaker: prefer the geometric normal most aligned with the
        // transformed shading normal.
        Ng_world = (abs(dot(Ng_b, N)) > abs(dot(Ng_a, N))) ? Ng_b : Ng_a;
    }

    // Re-orthogonalize T against the (now world-space) N.
    T = T - N * dot(N, T);
    T = safeNormalize(T, buildOrthoTangent(N));
    float3 B = safeNormalize(handedness * cross(N, T), buildOrthoTangent(N));

    float3 N_world = N;
    if (mat.normal_tex_index != NO_TEXTURE && mat.texture_weight > 0.0f) {
        float3 normal_tex = sample_tex(
            mat.normal_tex_index,
            attr.uv,
            float4(0.5, 0.5, 1.0, 1.0)
        ).xyz;

        normal_tex = lerp(float3(0.5, 0.5, 1.0), normal_tex, saturate(mat.texture_weight));
        normal_tex = normal_tex * 2.0 - 1.0;

        // Perturb N by the normal map in the world-space TBN frame.
        N_world = safeNormalize(
            T * normal_tex.x +
            B * normal_tex.y +
            N * normal_tex.z,
            N);
    }

    bool frontFace = dot(ray.dir, Ng_world) < 0.0f;
    if (!frontFace) {
        Ng_world = -Ng_world;
    }
    hit.front_face = frontFace;

    // Keep shading and geometric normals in the same hemisphere to avoid
    // mirrored-looking response on rotated or negatively-scaled instances.
    if (dot(N_world, Ng_world) < 0.0f) {
        N_world = -N_world;
        T = -T;
        B = -B;
    }

    hit.normal      = N_world;
    hit.geom_normal = Ng_world;
    hit.tangent   = T;
    hit.bitangent = B;
    return true;
}

// ─── Texture sampling ─────────────────────────────────────────────────────────
float4 sample_tex(uint index, float2 uv, float4 fallback) {
    if (index == NO_TEXTURE) return fallback;
    return material_textures[NonUniformResourceIndex(index)].SampleLevel(
        g_material_sampler, uv, 0.f);
}

// ─── Fill OpenPBR inputs ──────────────────────────────────────────────────────
OpenPBR_ResolvedInputs material_to_openpbr(
    GPUMaterial m, float2 uv, float3 n, float3 t, float3 b)
{
    OpenPBR_ResolvedInputs inp = openpbr_make_default_resolved_inputs();

    float texture_weight = saturate(m.texture_weight);
    float4 alb_sample = sample_tex(m.albedo_tex_index, uv, float4(1.f, 1.f, 1.f, 1.f));
    float4 alb = lerp(float4(1.f, 1.f, 1.f, 1.f), alb_sample, texture_weight);
    // glTF ORM texture: R=occlusion, G=roughness, B=metalness.
    // The CPU values act as live multipliers so GUI edits still affect
    // textured materials.
    float4 orm_sample = sample_tex(m.roughness_tex_index, uv, float4(1.f, 1.f, 1.f, 1.f));
    float4 orm = lerp(float4(1.f, 1.f, 1.f, 1.f), orm_sample, texture_weight);

    float geometry_opacity = saturate(m.geometry_opacity);

    inp.base_color              = alb.rgb * m.base_color.rgb;
    inp.base_weight             = alb.a * m.base_color.a * geometry_opacity;
    inp.base_metalness          = saturate(orm.b * m.base_metalness);   // B channel = metalness
    inp.base_diffuse_roughness  = m.base_diffuse_roughness;

    inp.specular_color                 = m.specular_color.rgb;
    inp.specular_weight                = m.specular_color.a;
    inp.specular_roughness             = clamp(orm.g * m.specular_roughness, 1e-3f, 1.f);
    inp.specular_ior                   = m.specular_ior;
    inp.specular_roughness_anisotropy  = m.specular_anisotropy;

    inp.transmission_weight                 = saturate(max(m.transmission_weight, 1.0f - geometry_opacity));
    inp.transmission_color                  = m.transmission_color.rgb;
    //inp.transmission_color                  = float3(1,0,0); // debug red transmission to visualize where it's used
    inp.transmission_depth                  = m.transmission_depth;
    inp.transmission_scatter                = m.transmission_scatter.rgb;
    inp.transmission_scatter_anisotropy     = m.transmission_scatter_anisotropy;
    inp.transmission_dispersion_scale       = m.transmission_dispersion_scale;
    inp.transmission_dispersion_abbe_number = m.transmission_dispersion_abbe_number;

    inp.geometry_opacity     = geometry_opacity;
    inp.geometry_thin_walled = (m.geometry_thin_walled > 0.5);

    inp.coat_weight    = m.coat_weight;
    inp.coat_color     = m.coat_color.rgb;
    inp.coat_roughness = m.coat_roughness;
    inp.coat_ior       = m.coat_ior;

    inp.fuzz_weight    = m.fuzz_weight;
    inp.fuzz_color     = m.fuzz_color.rgb;
    inp.fuzz_roughness = m.fuzz_roughness;

    inp.thin_film_weight    = m.thin_film_weight;
    inp.thin_film_ior       = m.thin_film_ior;
    inp.thin_film_thickness = m.thin_film_thickness;

    inp.shading_basis = {t, b, n};
    inp.coat_basis    = {t, b, n};
    return inp;
}

// ─── Environment ──────────────────────────────────────────────────────────────
float3 skyColor(float3 dir) {
    float t = 0.5f * (normalize(dir).y + 1.f);
    return lerp(float3(1, 1, 1), float3(0.5f, 0.7f, 1.f), t);
}

// ─── Path tracing ─────────────────────────────────────────────────────────────
float3 tracePath(Ray ray, inout RNG rng) {
    float3 throughput = float3(1, 1, 1);
    float3 radiance   = float3(0, 0, 0);

    const uint bounce_count = max(push.max_bounces, 1u);
    [loop]
    for (uint bounce = 0u; bounce < bounce_count; ++bounce) {
        Hit         hit;
        GPUMaterial mat;

        if (!intersectScene(ray, hit, mat)) {
            // Ray escaped to sky — treat sky as an environment light
            if (push.skybox_enabled != 0u)
                radiance += throughput * skyColor(ray.dir);
            break;
        }

        // ── Emission ──
        float3 emissive_texture =
            sample_tex(mat.emissive_tex_index, hit.uv, float4(1.f, 1.f, 1.f, 1.f)).rgb;
        float3 emissive_sample = lerp(
            float3(1.f, 1.f, 1.f), emissive_texture, saturate(mat.texture_weight));
        radiance += throughput * emissive_sample * mat.emission_color.rgb * mat.emission_luminance;

        float3 hitPos   = ray.origin + hit.t * ray.dir;
        float3 view_dir = -ray.dir;

        // For opaque surfaces, keep the face-forwarded shading normal to avoid
        // winding-dependent mirrored seams across triangles.
        // For transmissive/partially-opaque materials, preserve side information
        // so OpenPBR can swap IORs correctly on back-face hits.
        float3 openpbr_normal = hit.normal;
        if (mat.transmission_weight > 0.0f || mat.geometry_opacity < 0.999f) {
            openpbr_normal = hit.front_face ? hit.normal : -hit.normal;
        }
        OpenPBR_ResolvedInputs inputs = material_to_openpbr(
            mat, hit.uv, openpbr_normal, hit.tangent, hit.bitangent);

        OpenPBR_PreparedBsdf prepared = openpbr_prepare_bsdf_and_volume(
            inputs, throughput,
            OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor,
            view_dir);

        // ── Direct: sun NEE ──────────────────────────────────────────────────
        // openpbr_eval() takes world-space directions and returns BSDF × |cos θ_i|.
        // Do NOT convert to local space; do NOT multiply by n_dot_l again.
        // For transmissive materials, also sample the back hemisphere (backlit glass).
        if (push.directional_light_enabled != 0u) {
            float3 g_sun_dir = normalize(float3(push.sun_dir_x, push.sun_dir_y, push.sun_dir_z));
            float  g_sun_intensity = push.sun_intensity;

            bool sun_front = dot(hit.normal,  g_sun_dir) > 0.f;
            bool sun_back  = dot(hit.normal, -g_sun_dir) > 0.f &&
                             inputs.transmission_weight > 0.f;

            int occluder_front = traceOccluderInstance(hitPos, hit.normal, g_sun_dir, T_MAX);
            if (sun_front && occluder_front < 0) {
                float3 f = openpbr_get_sum_of_diffuse_specular(openpbr_eval(prepared, g_sun_dir));
                radiance += throughput * f * g_sun_color * g_sun_intensity;
            }
            // For back-hemisphere shadow rays, offset the origin toward the sun side
            // so the ray doesn't self-intersect with the surface it just hit.
            int occluder_back = traceOccluderInstance(hitPos, -hit.normal, -g_sun_dir, T_MAX);
            if (sun_back && occluder_back < 0) {
                float3 f = openpbr_get_sum_of_diffuse_specular(openpbr_eval(prepared, -g_sun_dir));
                radiance += throughput * f * g_sun_color * g_sun_intensity;
            }
        }

        // ── Indirect: importance-sample BSDF ────────────────────────────────
        float3               light_dir;
        OpenPBR_DiffuseSpecular weight;
        float                pdf;
        OpenPBR_BsdfLobeType sampled_type;
        openpbr_sample(prepared, rng.next3(), light_dir, weight, pdf, sampled_type);

        if (pdf <= 0.f) break;

        // weight = BSDF × cosine / pdf — standard MC estimator
        throughput *= openpbr_get_sum_of_diffuse_specular(weight);

        // Russian roulette after bounce 2
        if (bounce >= 2u) {
            float p = max(throughput.r, max(throughput.g, throughput.b));
            if (rng.nextFloat() > p) break;
            throughput /= p;
        }

        // Adaptive ray epsilon reduces self-intersection on large scenes while
        // keeping close-up precision.
        float  ray_eps  = max(T_MIN, hit.t * 1.0e-4f);
        float3 offset_n =
            dot(light_dir, hit.geom_normal) >= 0.0f ? hit.geom_normal : -hit.geom_normal;
        ray.origin = hitPos + offset_n * ray_eps;
        ray.dir    = light_dir;
    }

    return radiance;
}

// ─── Camera ───────────────────────────────────────────────────────────────────
Ray makeRay(uint2 pixel, uint2 dims, float2 jitter) {
    GPUCamera cam = camera_buf[NonUniformResourceIndex(0)];
    float3    origin = cam.position.xyz;
    float3    target = cam.target.xyz;
    float3    up     = cam.up.xyz;
    float     fov    = cam.fov_near_far.x;
    float     aspect = float(dims.x) / float(dims.y);

    float  half_h = tan(fov * (PI / 180.f) * 0.5f);
    float  half_w = aspect * half_h;

    float3 fwd   = normalize(target - origin);
    float3 right = normalize(cross(fwd, up));
    float3 up_   = cross(right, fwd);

    float2 uv = (float2(pixel) + jitter) / float2(dims);
    float  u  =  (2.f * uv.x - 1.f) * half_w;
    float  v  =  (1.f - 2.f * uv.y) * half_h;

    Ray ray;
    ray.origin = origin;
    ray.dir    = normalize(fwd + u * right + v * up_);
    return ray;
}

bool isSelectionOutline(uint2 pixel, uint2 dims, int center_instance) {
    if (push.selected_mesh_index < 0) return false;
    if (center_instance != push.selected_mesh_index) return false;

    const int outline_width = max(int(push.outline_width), 1);
    const int2 dims_i = int2(dims.x, dims.y);
    const int2 pixel_i = int2(pixel.x, pixel.y);
    const int2 neighbor_offsets[4] = {
        int2(-1, 0),
        int2(1, 0),
        int2(0, -1),
        int2(0, 1)
    };

    [unroll]
    for (int i = 0; i < 4; ++i) {
        int2 neighbor = pixel_i + neighbor_offsets[i] * outline_width;
        if (neighbor.x < 0 || neighbor.y < 0 || neighbor.x >= dims_i.x || neighbor.y >= dims_i.y) {
            return true;
        }

        int neighbor_instance =
            traceVisibleInstance(makeRay(uint2(neighbor.x, neighbor.y), dims, float2(0.5f, 0.5f)));
        if (neighbor_instance != push.selected_mesh_index) {
            return true;
        }
    }

    return false;
}

float3 srgbToLinear(float3 color) {
    float3 cutoff = step(float3(0.04045f, 0.04045f, 0.04045f), color);
    float3 low    = color / 12.92f;
    float3 high   = pow((color + 0.055f) / 1.055f, float3(2.4f, 2.4f, 2.4f));
    return lerp(low, high, cutoff);
}

float3 unchartedTonemapPartial(float3 color) {
    float3 x = max(float3(0.0f, 0.0f, 0.0f), color - 0.004f);
    return (x * (6.2f * x + 0.5f)) / (x * (6.2f * x + 1.7f) + 0.06f);
}

float3 unchartedTonemapFilmic(float3 color, float exposureBias) {
    float3 exposed = color * exposureBias;
    exposed        = unchartedTonemapPartial(exposed);
    float3 W       = unchartedTonemapPartial(float3(11.2f, 11.2f, 11.2f));
    float3 whiteScale = 1.0f / max(W, float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
    return exposed * whiteScale;
}

// ─── Entry point ──────────────────────────────────────────────────────────────

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 pixel = dispatch_id.xy;
    uint2 dims;
    output_image.GetDimensions(dims.x, dims.y);

    if (any(pixel >= dims)) return;

    // Stage 0: visibility pre-pass used for selection outlines.
    if (push.stage == 0) {
        int visible_id = traceVisibleInstance(makeRay(pixel, dims, float2(0.5f, 0.5f)));
        object_id_image[pixel] = visible_id;
        return;
    }

    // Naive mode only writes in stage 2.
    if (push.stage != 2) return;

    int visible_object_id = object_id_image[pixel];

    RNG rng = RNG::make(pixel, push.frame);
    float3 colorAccum = float3(0, 0, 0);
    const uint spp = max(push.spp, 1u);
    [loop]
    for (uint i = 0u; i < spp; ++i) {
        uint sample_index = push.frame * spp + i;
        float2 jitter = haltonJitter(sample_index, pixel);
        colorAccum += tracePath(makeRay(pixel, dims, jitter), rng);
    }
    float3 color = colorAccum / float(spp);

    float3 prev  = accum_image[pixel].rgb;
    float  blend = 1.f / float(push.frame + 1u);
    float3 accum = lerp(prev, color, blend);
    accum_image[pixel] = float4(accum, 1.f);

    float3 display = accum;
    if (push.enable_tonemapping != 0u) {
        display = unchartedTonemapFilmic(display, push.exposure_bias);
    }
    if (isSelectionOutline(pixel, dims, visible_object_id)) {
        display = lerp(display, push.outline_color.rgb, saturate(push.outline_color.a));
    }
    output_image[pixel] = float4(display, 1.f);
}
