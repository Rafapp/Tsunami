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
    uint   accumulation_frame;
    uint   rng_seed;
    uint   material_count;
    int    selected_object_id;
    uint   outline_width;
    int    debug_view_mode;
    int    water_object_id;
    uint   water_enabled;
    int    water_material_index;
    float  water_height_to_world_scale;
    float  _pad0;
    float  _pad1;
    float4 outline_color;
    float4 water_center_trace_half_height;
    float4 water_axis_u_half_extent;
    float4 water_axis_v_half_extent;
    float4 water_normal;
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
[[vk::binding(14, 0)]] RWTexture2D<float>             water_height_image;

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
    float     _pad0[2];

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
static const int   MAX_BOUNCES = 8;
static const float T_MIN       = 1e-3f;
static const float T_MAX       = 1e30f;
static const uint  NO_TEXTURE  = 0xFFFFFFFFu;

// ─── Light: directional sun ───────────────────────────────────────────────────
// A directional light has no distance falloff and shoots shadow rays to T_MAX,
// so it passes cleanly through Sponza's open atrium top regardless of the
// camera position.  Angle it slightly so surfaces at grazing incidence still
// catch some light.
static float3 g_sun_dir       = normalize(float3(0.f, -1.0f, 0.f));
static const float3 g_sun_color     = float3(1.0f, 0.95f, 0.85f);
static const float  g_sun_intensity = 0.5f;   // tweak for desired exposure

// ─── Types ────────────────────────────────────────────────────────────────────
struct Ray { float3 origin; float3 dir; };
struct Hit {
    float  t;
    float3 position;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
    bool   front_face;
    int    matIndex;
    int    instanceID;
};

// ─── RNG ──────────────────────────────────────────────────────────────────────
struct RNG {
    uint state;
    static RNG make(uint2 pixel, uint frame) {
        RNG r;
        r.state = pixel.x ^ (pixel.y << 16) ^ (frame * 719393u);
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
struct VertexAttribs { float3 normal; float2 uv; float4 tangent;};

bool intersectAABB(float3 ray_origin, float3 ray_dir, float3 box_min, float3 box_max,
                   out float t_min, out float t_max) {
    float3 inv_dir = 1.0f / ray_dir;
    float3 t0 = (box_min - ray_origin) * inv_dir;
    float3 t1 = (box_max - ray_origin) * inv_dir;

    float3 t_small = min(t0, t1);
    float3 t_large = max(t0, t1);

    t_min = max(max(t_small.x, t_small.y), t_small.z);
    t_max = min(min(t_large.x, t_large.y), t_large.z);
    return t_max >= max(t_min, 0.0f);
}

bool waterEnabled() {
    return push.water_enabled != 0 &&
           push.water_axis_u_half_extent.w > 1e-4f &&
           push.water_axis_v_half_extent.w > 1e-4f &&
           push.water_height_to_world_scale > 1e-5f;
}

float3 worldToWater(float3 world_pos) {
    float3 delta = world_pos - push.water_center_trace_half_height.xyz;
    return float3(
        dot(delta, push.water_axis_u_half_extent.xyz) / max(push.water_axis_u_half_extent.w, 1e-4f),
        dot(delta, push.water_normal.xyz),
        dot(delta, push.water_axis_v_half_extent.xyz) / max(push.water_axis_v_half_extent.w, 1e-4f));
}

float3 waterToWorld(float3 water_pos) {
    return push.water_center_trace_half_height.xyz +
           push.water_axis_u_half_extent.xyz * (water_pos.x * push.water_axis_u_half_extent.w) +
           push.water_normal.xyz * water_pos.y +
           push.water_axis_v_half_extent.xyz * (water_pos.z * push.water_axis_v_half_extent.w);
}

float readWaterHeightRaw(int2 coord, int2 dims) {
    return water_height_image[clamp(coord, int2(0, 0), dims - 1)];
}

float sampleWaterHeightRaw(float2 uv, int2 dims) {
    uv = clamp(uv, float2(0.0f, 0.0f), float2(1.0f, 1.0f));

    float2 texel = uv * float2(dims - 1);
    int2 p0 = int2(floor(texel));
    int2 p1 = min(p0 + int2(1, 1), dims - 1);
    float2 f = texel - float2(p0);

    float h00 = readWaterHeightRaw(p0, dims);
    float h10 = readWaterHeightRaw(int2(p1.x, p0.y), dims);
    float h01 = readWaterHeightRaw(int2(p0.x, p1.y), dims);
    float h11 = readWaterHeightRaw(p1, dims);

    return lerp(lerp(h00, h10, f.x), lerp(h01, h11, f.x), f.y);
}

float sampleWaterHeight(float2 water_xz, int2 dims) {
    float2 uv = water_xz * 0.5f + 0.5f;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        return -1.0e6f;
    }

    return sampleWaterHeightRaw(uv, dims) * push.water_height_to_world_scale;
}

float3 waterSurfaceNormal(float2 water_xz, int2 dims) {
    float dx = 2.0f / max(float(dims.x), 1.0f);
    float dz = 2.0f / max(float(dims.y), 1.0f);

    float hL = sampleWaterHeight(water_xz - float2(dx, 0.0f), dims);
    float hR = sampleWaterHeight(water_xz + float2(dx, 0.0f), dims);
    float hD = sampleWaterHeight(water_xz - float2(0.0f, dz), dims);
    float hU = sampleWaterHeight(water_xz + float2(0.0f, dz), dims);

    float3 pL = waterToWorld(float3(water_xz.x - dx, hL, water_xz.y));
    float3 pR = waterToWorld(float3(water_xz.x + dx, hR, water_xz.y));
    float3 pD = waterToWorld(float3(water_xz.x, hD, water_xz.y - dz));
    float3 pU = waterToWorld(float3(water_xz.x, hU, water_xz.y + dz));

    float3 normal = normalize(cross(pR - pL, pU - pD));
    if (dot(normal, push.water_normal.xyz) < 0.0f) {
        normal = -normal;
    }
    return normal;
}

GPUMaterial makeWaterMaterial() {
    GPUMaterial water = (GPUMaterial)0;
    water.base_color = float4(1.0f, 1.0f, 1.0f, 1.0f);
    water.base_metalness = 0.0f;
    water.base_diffuse_roughness = 0.0f;
    water.specular_color = float4(1.0f, 1.0f, 1.0f, 1.0f);
    water.specular_roughness = 0.0125f;
    water.specular_ior = 1.333f;
    water.specular_anisotropy = 0.0f;
    water.transmission_weight = 1.0f;
    water.transmission_depth = 2.2f;
    water.transmission_scatter_anisotropy = 0.0f;
    water.geometry_opacity = 1.0f;
    water.transmission_color = float4(0.92f, 0.97f, 1.0f, 1.0f);
    water.transmission_scatter = float4(0.0f, 0.0f, 0.0f, 0.0f);
    water.emission_color = float4(0.0f, 0.0f, 0.0f, 0.0f);
    water.emission_luminance = 0.0f;
    water.geometry_thin_walled = 0.0f;
    water.transmission_dispersion_scale = 0.0f;
    water.transmission_dispersion_abbe_number = 0.0f;
    water.coat_weight = 0.0f;
    water.coat_roughness = 0.0f;
    water.coat_ior = 1.0f;
    water.coat_color = float4(0.0f, 0.0f, 0.0f, 0.0f);
    water.fuzz_weight = 0.0f;
    water.fuzz_roughness = 0.0f;
    water.fuzz_color = float4(0.0f, 0.0f, 0.0f, 0.0f);
    water.thin_film_weight = 0.0f;
    water.thin_film_ior = 1.0f;
    water.thin_film_thickness = 0.0f;
    water.albedo_tex_index = NO_TEXTURE;
    water.normal_tex_index = NO_TEXTURE;
    water.roughness_tex_index = NO_TEXTURE;
    water.emissive_tex_index = NO_TEXTURE;
    return water;
}

GPUMaterial loadWaterMaterial() {
    GPUMaterial water = makeWaterMaterial();
    if (push.water_material_index >= 0 &&
        uint(push.water_material_index) < push.material_count) {
        water = materials_buf[NonUniformResourceIndex(push.water_material_index)];
    }

    // The UI exposes "Base tint" but not a separate transmission tint, so use
    // the base color as the visible water tint for both opaque and transmissive looks.
    water.base_color.a = max(water.base_color.a, 1.0f);
    water.transmission_color.rgb *= water.base_color.rgb;
    return water;
}

bool intersectWaterSurface(Ray ray, out Hit hit, out GPUMaterial mat) {
    hit = (Hit)0;
    mat = (GPUMaterial)0;

    if (!waterEnabled()) return false;

    uint width = 0;
    uint height = 0;
    water_height_image.GetDimensions(width, height);
    int2 dims = int2(int(width), int(height));
    if (dims.x <= 1 || dims.y <= 1) return false;

    float3 local_origin = worldToWater(ray.origin);
    float3 local_dir = float3(
        dot(ray.dir, push.water_axis_u_half_extent.xyz) / max(push.water_axis_u_half_extent.w, 1e-4f),
        dot(ray.dir, push.water_normal.xyz),
        dot(ray.dir, push.water_axis_v_half_extent.xyz) / max(push.water_axis_v_half_extent.w, 1e-4f));

    float t_min;
    float t_max;
    if (!intersectAABB(local_origin, local_dir,
                       float3(-1.04f, -push.water_center_trace_half_height.w, -1.04f),
                       float3(1.04f, push.water_center_trace_half_height.w, 1.04f),
                       t_min, t_max)) {
        return false;
    }

    t_min = max(t_min, T_MIN);

    float prev_t = t_min;
    float3 prev_pos = ray.origin + ray.dir * prev_t;
    float3 prev_local = worldToWater(prev_pos);
    float prev_delta = prev_local.y - sampleWaterHeight(prev_local.xz, dims);

    [loop]
    for (int step = 1; step <= 80; ++step) {
        float t = lerp(t_min, t_max, float(step) / 80.0f);
        float3 pos = ray.origin + ray.dir * t;
        float3 local = worldToWater(pos);
        float delta = local.y - sampleWaterHeight(local.xz, dims);

        if (prev_delta == 0.0f || delta == 0.0f || prev_delta * delta <= 0.0f) {
            float lo = prev_t;
            float hi = t;
            float lo_delta = prev_delta;

            [unroll]
            for (int refine = 0; refine < 6; ++refine) {
                float mid = 0.5f * (lo + hi);
                float3 mid_local = worldToWater(ray.origin + ray.dir * mid);
                float mid_delta = mid_local.y - sampleWaterHeight(mid_local.xz, dims);
                if (lo_delta == 0.0f || lo_delta * mid_delta <= 0.0f) {
                    hi = mid;
                } else {
                    lo = mid;
                    lo_delta = mid_delta;
                }
            }

            hit.t = 0.5f * (lo + hi);
            float3 hit_local = worldToWater(ray.origin + ray.dir * hit.t);
            hit_local.y = sampleWaterHeight(hit_local.xz, dims);
            hit.position = waterToWorld(hit_local);

            float3 geometric_normal = waterSurfaceNormal(hit_local.xz, dims);
            float3 tangent = normalize(
                push.water_axis_u_half_extent.xyz -
                geometric_normal * dot(geometric_normal, push.water_axis_u_half_extent.xyz));
            float3 bitangent = normalize(cross(geometric_normal, tangent));
            if (dot(bitangent, push.water_axis_v_half_extent.xyz) < 0.0f) {
                bitangent = -bitangent;
            }

            bool front_face = dot(ray.dir, geometric_normal) < 0.0f;
            hit.normal = front_face ? geometric_normal : -geometric_normal;
            hit.tangent = tangent;
            hit.bitangent = bitangent;
            hit.uv = hit_local.xz * 0.5f + 0.5f;
            hit.front_face = front_face;
            hit.matIndex = push.water_material_index;
            hit.instanceID = push.water_object_id;
            mat = loadWaterMaterial();
            return true;
        }

        prev_t = t;
        prev_delta = delta;
    }

    return false;
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

bool traceVisibleMeshHit(Ray ray, out float hit_t, out int object_id) {
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

    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        hit_t = 0.0f;
        object_id = -1;
        return false;
    }

    hit_t = q.CommittedRayT();
    object_id = int(q.CommittedInstanceID());
    return true;
}

int traceVisibleObjectId(Ray ray) {
    float mesh_hit_t = 0.0f;
    int mesh_object_id = -1;
    bool hit_mesh = traceVisibleMeshHit(ray, mesh_hit_t, mesh_object_id);

    Hit water_hit;
    GPUMaterial water_mat;
    bool hit_water = intersectWaterSurface(ray, water_hit, water_mat);

    if (hit_water && (!hit_mesh || water_hit.t < mesh_hit_t)) {
        return water_hit.instanceID;
    }

    return hit_mesh ? mesh_object_id : -1;
}

// ─── Scene intersection ───────────────────────────────────────────────────────
bool intersectMeshScene(Ray ray, out Hit hit, out GPUMaterial mat) {
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
    bool frontFace = q.CommittedTriangleFrontFace();

    VertexAttribs attr = fetchAttribs(instID, primID, bary);

    hit.position   = ray.origin + hit.t * ray.dir;
    hit.uv         = attr.uv;
    hit.front_face = frontFace;
    hit.instanceID = int(instID);
    hit.matIndex = meshes_buf[NonUniformResourceIndex(instID)].matIndex;
    mat = materials_buf[NonUniformResourceIndex(hit.matIndex)];

    float3 N = normalize(attr.normal);
    float3 T = normalize(attr.tangent.xyz);
    float  handedness = attr.tangent.w;

    T = normalize(T - N * dot(N, T));
    float3 B = handedness * cross(N, T);

    float3 normal_tex = sample_tex(
        mat.normal_tex_index,
        attr.uv,
        float4(0.5, 0.5, 1.0, 1.0)
    ).xyz;

    normal_tex = normal_tex * 2.0 - 1.0;

    float3 N_world = normalize(
        T * normal_tex.x +
        B * normal_tex.y +
        N * normal_tex.z
    );

    hit.normal    = frontFace ? N_world : -N_world;
    hit.tangent   = T;
    hit.bitangent = B;
    return true;
}

bool intersectScene(Ray ray, out Hit hit, out GPUMaterial mat) {
    Hit mesh_hit;
    Hit water_hit;
    GPUMaterial mesh_mat;
    GPUMaterial water_mat;

    bool hit_mesh = intersectMeshScene(ray, mesh_hit, mesh_mat);
    bool hit_water = intersectWaterSurface(ray, water_hit, water_mat);

    if (hit_water && (!hit_mesh || water_hit.t < mesh_hit.t)) {
        hit = water_hit;
        mat = water_mat;
        return true;
    }

    if (hit_mesh) {
        hit = mesh_hit;
        mat = mesh_mat;
        return true;
    }

    hit = (Hit)0;
    mat = (GPUMaterial)0;
    return false;
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

    float4 alb = sample_tex(m.albedo_tex_index, uv, float4(1.f, 1.f, 1.f, 1.f));
    // glTF ORM texture: R=occlusion, G=roughness, B=metalness.
    // The CPU values act as live multipliers so GUI edits still affect
    // textured materials.
    float4 orm = sample_tex(m.roughness_tex_index, uv, float4(1.f, 1.f, 1.f, 1.f));

    inp.base_color              = alb.rgb * m.base_color.rgb;
    inp.base_weight             = alb.a * m.base_color.a;
    inp.base_metalness          = saturate(orm.b * m.base_metalness);   // B channel = metalness
    inp.base_diffuse_roughness  = m.base_diffuse_roughness;

    inp.specular_color                 = m.specular_color.rgb;
    inp.specular_weight                = m.specular_color.a;
    inp.specular_roughness             = clamp(orm.g * m.specular_roughness, 1e-3f, 1.f);
    inp.specular_ior                   = m.specular_ior;
    inp.specular_roughness_anisotropy  = m.specular_anisotropy;

    inp.transmission_weight                 = m.transmission_weight;
    inp.transmission_color                  = m.transmission_color.rgb;
    //inp.transmission_color                  = float3(1,0,0); // debug red transmission to visualize where it's used
    inp.transmission_depth                  = m.transmission_depth;
    inp.transmission_scatter                = m.transmission_scatter.rgb;
    inp.transmission_scatter_anisotropy     = m.transmission_scatter_anisotropy;
    inp.transmission_dispersion_scale       = m.transmission_dispersion_scale;
    inp.transmission_dispersion_abbe_number = m.transmission_dispersion_abbe_number;

    inp.geometry_opacity     = m.geometry_opacity;
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
float3 tracePath(Ray ray, inout RNG rng, out bool primary_visible_water) {
    float3 throughput = float3(1, 1, 1);
    float3 radiance   = float3(0, 0, 0);
    primary_visible_water = false;

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
        Hit         hit;
        GPUMaterial mat;

        if (!intersectScene(ray, hit, mat)) {
            // Ray escaped to sky — treat sky as an environment light
            radiance += throughput * skyColor(ray.dir);
            break;
        }

        if (bounce == 0 && hit.instanceID == push.water_object_id) {
            primary_visible_water = true;
        }

        // ── Emission ──
        float3 emissive_sample =
            sample_tex(mat.emissive_tex_index, hit.uv, float4(1.f, 1.f, 1.f, 1.f)).rgb;
        radiance += throughput * emissive_sample * mat.emission_color.rgb * mat.emission_luminance;

        float3 hitPos   = hit.position;
        float3 view_dir = -ray.dir;

        // Pass the unflipped geometric normal to OpenPBR so it can detect back-face
        // hits via dot(view_dir, shading_normal) < 0 and swap IORs for refraction.
        // hit.normal is face-forwarded (always toward the ray origin), so we undo that.
        float3 openpbr_normal = hit.front_face ? hit.normal : -hit.normal;
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
        {
            bool sun_front = dot(hit.normal,  g_sun_dir) > 0.f;
            bool sun_back  = dot(hit.normal, -g_sun_dir) > 0.f && inputs.transmission_weight > 0.f;

            if (sun_front && !is_occluded(hitPos, hit.normal, g_sun_dir, T_MAX)) {
                float3 f = openpbr_get_sum_of_diffuse_specular(openpbr_eval(prepared, g_sun_dir));
                radiance += throughput * f * g_sun_color * g_sun_intensity;
            }
            // For back-hemisphere shadow rays, offset the origin toward the sun side
            // so the ray doesn't self-intersect with the surface it just hit.
            if (sun_back && !is_occluded(hitPos, -hit.normal, -g_sun_dir, T_MAX)) {
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

        // Russian roulette after bounce 3
        if (bounce >= 6) {
            float p = max(throughput.r, max(throughput.g, throughput.b));
            if (rng.nextFloat() > p) break;
            throughput /= p;
        }

        // check for refraction and flip the ray bias direction if needed to avoid self-intersection
        float biasDir = 1.0f;
        if ((sampled_type & OpenPBR_BsdfLobeTypeTransmission) == 1 && hit.front_face) {
            biasDir = -1.0f;
        }

        float3 offset_n = dot(light_dir, hit.normal) >= 0.f ? hit.normal : -hit.normal;
        ray.origin = hitPos + offset_n * (T_MIN * biasDir);
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
    if (push.selected_object_id < 0) return false;
    if (center_instance != push.selected_object_id) return false;

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
            traceVisibleObjectId(makeRay(uint2(neighbor.x, neighbor.y), dims, float2(0.5f, 0.5f)));
        if (neighbor_instance != push.selected_object_id) {
            return true;
        }
    }

    return false;
}

uint hash_u32(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float3 debugColorForObjectId(int object_id) {
    if (object_id < 0) {
        return float3(0.02f, 0.02f, 0.02f);
    }

    uint hashed = hash_u32(uint(object_id + 1));
    float3 rgb = float3(
        float(hashed & 0xFFu),
        float((hashed >> 8) & 0xFFu),
        float((hashed >> 16) & 0xFFu)) / 255.0f;
    return 0.15f + 0.85f * rgb;
}

// ─── Entry point ──────────────────────────────────────────────────────────────

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 pixel = dispatch_id.xy;
    uint2 dims;
    output_image.GetDimensions(dims.x, dims.y);
    if (any(pixel >= dims)) return;

    RNG rng = RNG::make(pixel, push.rng_seed);
    int visible_object_id = traceVisibleObjectId(makeRay(pixel, dims, float2(0.5f, 0.5f)));
    object_id_image[pixel] = visible_object_id;

    static const int SPP = 1;
    float3 colorAccum = float3(0, 0, 0);
    bool primary_visible_water = false;
    for (int i = 0; i < SPP; ++i) {
        float2 jitter = float2(rng.nextFloat(), rng.nextFloat());
        bool sample_hits_water = false;
        colorAccum += tracePath(makeRay(pixel, dims, jitter), rng, sample_hits_water);
        primary_visible_water = primary_visible_water || sample_hits_water;
    }
    float3 color = colorAccum / float(SPP);

    // Temporal accumulation — blend current frame into running average
    float3 prev  = accum_image[pixel].rgb;
    float  blend = primary_visible_water ? 1.f : 1.f / float(push.accumulation_frame + 1u);
    float3 accum = lerp(prev, color, blend);

    accum_image[pixel] = float4(accum, 1.f);

    float3 display = (push.debug_view_mode == 1) ? debugColorForObjectId(visible_object_id) : accum;
    if (isSelectionOutline(pixel, dims, visible_object_id)) {
        display = lerp(display, push.outline_color.rgb, saturate(push.outline_color.a));
    }
    output_image[pixel] = float4(display, 1.f);
}
