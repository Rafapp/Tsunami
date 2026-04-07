// ─── OpenPBR texture-LUT mode ─────────────────────────────────────────────────
#define OPENPBR_LANGUAGE_TARGET_SLANG 1
#define OPENPBR_USE_TEXTURE_LUTS      1

static const uint NUM_LUTS         = 8;
static const uint MAX_MAT_TEXTURES = 256;

[[vk::binding(3,  0)]] SamplerState g_lut_sampler;
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
    uint frame;
    uint material_count;
    int selected_mesh_index;
    uint outline_width;
    int debug_view_mode;
    uint stage; // 0 = probe pass, 1 = path trace pass
    uint _pad0;
    uint _pad1;
    float4 outline_color;
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
[[vk::binding(13, 0)]] RWTexture2D<int> object_id_image;
[[vk::binding(14, 0)]] RWStructuredBuffer<TileData> tile_buffer;
[[vk::binding(15, 0)]] RWStructuredBuffer<uint> tile_render_flags; // 1 = render this tile

//Groupshare buffers for counting hits on other objects to drive importance sampling in next pass
groupshared uint gs_prim_hist[256];
groupshared uint gs_sec_hist[256];

// ─── GPU structs ──────────────────────────────────────────────────────────────
struct GPUCamera {
    float4 position;
    float4 target;
    float4 up;
    float4 fov_near_far;
};

// Tile Struct
struct TileData {
    int primaryObject;
    int nextObjects[3]; // top-3 secondary-hit objects
};

struct GPUMaterial {
    float4 base_color;
    float  base_metalness;
    float  base_diffuse_roughness;
    float  _pad0[2];

    float4 specular_color;
    float  specular_roughness;
    float  specular_ior;
    float  specular_anisotropy;
    float  _pad1;

    float  transmission_weight;
    float  transmission_depth;
    float  transmission_scatter_anisotropy;
    float  geometry_opacity;
    float4 transmission_color;
    float4 transmission_scatter;

    float4 emission_color;
    float  emission_luminance;
    float  geometry_thin_walled;
    float  transmission_dispersion_scale;
    float  transmission_dispersion_abbe_number;

    float  coat_weight;
    float  coat_roughness;
    float  coat_ior;
    float  _pad4;
    float4 coat_color;

    float  fuzz_weight;
    float  fuzz_roughness;
    float  _pad5[2];
    float4 fuzz_color;

    float  thin_film_weight;
    float  thin_film_ior;
    float  thin_film_thickness;
    float  _pad6;

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
static const int   MAX_BOUNCES = 6;
static const float T_MIN       = 1e-3f;
static const float T_MAX       = 1e30f;
static const uint  NO_TEXTURE  = 0xFFFFFFFFu;

// ─── Light ────────────────────────────────────────────────────────────────────
static float3       g_sun_dir       = normalize(float3(0.f, 1.0f, 0.f));
static const float3 g_sun_color     = float3(1.0f, 0.95f, 0.85f);
static const float  g_sun_intensity = 0.5f;

// ─── Types ────────────────────────────────────────────────────────────────────
struct Ray { float3 origin; float3 dir; };
struct Hit {
    float  t;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
    bool   front_face;
    int    matIndex;
    int    instanceID;
};


// ─── Cosine-weighted hemisphere sampling ───────────────────────────────────────
float3 cosineHemisphere(inout RNG rng, float3 n) {
    float u1 = rng.nextFloat(), u2 = rng.nextFloat();
    float r = sqrt(u1);
    float phi = 2.f * PI * u2;
    float3 local = float3(r * cos(phi), r * sin(phi), sqrt(max(0.f, 1.f - u1)));
    float3 up = abs(n.y) < 0.9f ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t = normalize(cross(up, n));
    float3 b = cross(n, t);
    return normalize(t * local.x + b * local.y + n * local.z);
}


/// ─── Halton sequence for stable sampling in probe pass ─────────────────────────
float halton(uint index, uint base)
{
    float f = 1.0f;
    float r = 0.0f;

    while (index > 0u)
    {
        f = f / float(base);
        r = r + f * float(index % base);
        index = index / base;
    }

    return r;
}

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
struct VertexAttribs { float3 normal; float2 uv; float4 tangent; };

VertexAttribs fetchAttribs(uint instanceID, uint primID, float2 bary) {
    uint voff = (uint)meshes_buf[NonUniformResourceIndex(instanceID)].vertexOffset;
    uint ioff = (uint)meshes_buf[NonUniformResourceIndex(instanceID)].indexOffset;
    uint i0 = indices_buf[ioff + primID * 3 + 0];
    uint i1 = indices_buf[ioff + primID * 3 + 1];
    uint i2 = indices_buf[ioff + primID * 3 + 2];
    float w0 = 1.f - bary.x - bary.y;
    VertexAttribs a;
    a.normal  = normalize(vertices_buf[voff+i0].normal  * w0 + vertices_buf[voff+i1].normal  * bary.x + vertices_buf[voff+i2].normal  * bary.y);
    a.tangent =            vertices_buf[voff+i0].tangent * w0 + vertices_buf[voff+i1].tangent * bary.x + vertices_buf[voff+i2].tangent * bary.y;
    a.uv      =            vertices_buf[voff+i0].uv      * w0 + vertices_buf[voff+i1].uv      * bary.x + vertices_buf[voff+i2].uv      * bary.y;
    return a;
}

// ─── Texture sampling ─────────────────────────────────────────────────────────
float4 sample_tex(uint index, float2 uv, float4 fallback) {
    if (index == NO_TEXTURE) return fallback;
    return material_textures[NonUniformResourceIndex(index)].SampleLevel(g_material_sampler, uv, 0.f);
}

// ─── Shadow helper ────────────────────────────────────────────────────────────
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
    bool frontFace = q.CommittedTriangleFrontFace();

    VertexAttribs attr = fetchAttribs(instID, primID, bary);
    hit.uv         = attr.uv;
    hit.front_face = frontFace;
    hit.instanceID = int(instID);
    hit.matIndex   = meshes_buf[NonUniformResourceIndex(instID)].matIndex;
    mat            = materials_buf[NonUniformResourceIndex(hit.matIndex)];

    float3 N = normalize(attr.normal);
    float3 T = normalize(attr.tangent.xyz);
    float  handedness = attr.tangent.w;
    T = normalize(T - N * dot(N, T));
    float3 B = handedness * cross(N, T);

    float3 normal_tex = sample_tex(mat.normal_tex_index, attr.uv,
                                   float4(0.5, 0.5, 1.0, 1.0)).xyz * 2.0 - 1.0;
    float3 N_world = normalize(T * normal_tex.x + B * normal_tex.y + N * normal_tex.z);

    hit.normal    = frontFace ? N_world : -N_world;
    hit.tangent   = T;
    hit.bitangent = B;
    return true;
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

// ─── Fill OpenPBR inputs ──────────────────────────────────────────────────────
OpenPBR_ResolvedInputs material_to_openpbr(
    GPUMaterial m, float2 uv, float3 n, float3 t, float3 b)
{
    OpenPBR_ResolvedInputs inp = openpbr_make_default_resolved_inputs();

    float4 alb = sample_tex(m.albedo_tex_index, uv, float4(1.f, 1.f, 1.f, 1.f));
    float4 orm = sample_tex(m.roughness_tex_index, uv, float4(1.f, 1.f, 1.f, 1.f));

    inp.base_color             = alb.rgb * m.base_color.rgb;
    inp.base_weight            = alb.a * m.base_color.a;
    inp.base_metalness         = saturate(orm.b * m.base_metalness);
    inp.base_diffuse_roughness = m.base_diffuse_roughness;

    inp.specular_color                = m.specular_color.rgb;
    inp.specular_weight               = m.specular_color.a;
    inp.specular_roughness            = clamp(orm.g * m.specular_roughness, 1e-3f, 1.f);
    inp.specular_ior                  = m.specular_ior;
    inp.specular_roughness_anisotropy = m.specular_anisotropy;

    inp.transmission_weight                 = m.transmission_weight;
    float3 trans_albedo = m.transmission_color.rgb;
    bool transIsWhite = (trans_albedo.r + trans_albedo.g + trans_albedo.b > 2.99f);
    if (transIsWhite) {
        trans_albedo = inp.base_color;
    }
    inp.transmission_color                  = trans_albedo;
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

// ─── Debug helpers ────────────────────────────────────────────────────────────
uint hash_u32(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float3 debugColorForObjectId(int object_id) {
    if (object_id < 0) return float3(0.02f, 0.02f, 0.02f);
    uint hashed = hash_u32(uint(object_id + 1));
    float3 rgb = float3(
        float(hashed & 0xFFu),
        float((hashed >> 8) & 0xFFu),
        float((hashed >> 16) & 0xFFu)) / 255.0f;
    return 0.15f + 0.85f * rgb;
}

// ─── Outline helper ───────────────────────────────────────────────────────────
bool isSelectionOutline(uint2 pixel, uint2 dims, int center_instance) {
    if (push.selected_mesh_index < 0) return false;
    if (center_instance != push.selected_mesh_index) return false;

    const int  outline_width = max(int(push.outline_width), 1);
    const int2 dims_i        = int2(dims.x, dims.y);
    const int2 pixel_i       = int2(pixel.x, pixel.y);
    const int2 offsets[4]    = { int2(-1,0), int2(1,0), int2(0,-1), int2(0,1) };

    [unroll]
    for (int i = 0; i < 4; ++i) {
        int2 neighbor = pixel_i + offsets[i] * outline_width;
        if (neighbor.x < 0 || neighbor.y < 0 ||
            neighbor.x >= dims_i.x || neighbor.y >= dims_i.y)
            return true;
        int neighbor_instance = traceVisibleInstance(
            makeRay(uint2(neighbor.x, neighbor.y), dims, float2(0.5f, 0.5f)));
        if (neighbor_instance != push.selected_mesh_index)
            return true;
    }
    return false;
}

// ─── Camera ───────────────────────────────────────────────────────────────────
Ray makeRay(uint2 pixel, uint2 dims, float2 jitter) {
    GPUCamera cam    = camera_buf[NonUniformResourceIndex(0)];
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

// ── Stage 0: probe pass ───────────────────────────────────────────────────────
void stage0(uint2 pixel, uint2 dims, uint local_idx, uint3 group_id, inout RNG rng) {
    // Zero histograms — each thread clears its own slot
    gs_prim_hist[local_idx] = 0;
    gs_sec_hist[local_idx] = 0;
    GroupMemoryBarrierWithGroupSync();

    // Primary ray (centre of pixel — probe is stable, no jitter)
    Ray ray = makeRay(pixel, dims, float2(0.5f, 0.5f));
    Hit hit; GPUMaterial mat;
    int prim_obj = -1, sec_obj = -1;

    if (intersectScene(ray, hit, mat)) {
        prim_obj = hit.instanceID;

        // One secondary bounce: cosine-weighted hemisphere with fixed seed per pixel
        // so the probe map is stable across frames for a static scene
        RNG probe_rng = RNG::make(pixel, 99991u);
        float3 hit_pos = ray.origin + hit.t * ray.dir;
        float3 sec_dir = cosineHemisphere(probe_rng, hit.normal);
        Ray sec_ray;
        sec_ray.origin = hit_pos + hit.normal * T_MIN;
        sec_ray.dir = sec_dir;
        Hit hit2; GPUMaterial mat2;
        if (intersectScene(sec_ray, hit2, mat2))
            sec_obj = hit2.instanceID;
    }

    object_id_image[pixel] = prim_obj;

    // Accumulate into histograms — only valid object IDs (avoids out-of-bounds)
    if (prim_obj >= 0 && prim_obj < 256)
        InterlockedAdd(gs_prim_hist[prim_obj], 1u);
    if (sec_obj >= 0 && sec_obj < 256 && sec_obj != prim_obj)
        InterlockedAdd(gs_sec_hist[sec_obj], 1u);

    GroupMemoryBarrierWithGroupSync();

    // Thread 0 writes the TileData summary
    if (local_idx == 0) {
        uint tiles_x = (dims.x + 15u) / 16u;
        uint tile_idx = group_id.y * tiles_x + group_id.x;

        // Find top primary
        int best_prim = -1;
        uint best_count = 0u;
        for (uint i = 0u; i < 256u; ++i) {
            if (gs_prim_hist[i] > best_count) {
                best_count = gs_prim_hist[i];
                best_prim = int(i);
            }
        }

        // Find top-3 secondary objects (excluding primary)
        int top3[3] = { -1, -1, -1 };
        uint top3c[3] = { 0, 0, 0 };
        for (uint i = 0u; i < 256u; ++i) {
            uint c = gs_sec_hist[i];
            if (c == 0u || int(i) == best_prim) continue;
            if (c > top3c[0]) {
                top3[2] = top3[1]; top3c[2] = top3c[1];
                top3[1] = top3[0]; top3c[1] = top3c[0];
                top3[0] = int(i); top3c[0] = c;
            } else if (c > top3c[1]) {
                top3[2] = top3[1]; top3c[2] = top3c[1];
                top3[1] = int(i); top3c[1] = c;
            } else if (c > top3c[2]) {
                top3[2] = int(i); top3c[2] = c;
            }
        }

        TileData td;
        td.primaryObject = best_prim;
        td.nextObjects[0] = top3[0];
        td.nextObjects[1] = top3[1];
        td.nextObjects[2] = top3[2];
        tile_buffer[tile_idx] = td;
    }

    // Write display output (show existing accum while probe builds)
    float3 display = (push.debug_view_mode == 1)
                         ? debugColorForObjectId(prim_obj)
                         : accum_image[pixel].rgb;
    if (isSelectionOutline(pixel, dims, prim_obj))
        display = lerp(display, push.outline_color.rgb, saturate(push.outline_color.a));
    output_image[pixel] = float4(display, 1.f);
}

// ── Stage 1: path trace ───────────────────────────────────────────────────────
void stage1(uint2 pixel, uint2 dims, uint tile_idx, inout RNG rng) {
    // Workgroup-uniform early exit: skip tiles not in the current BFS frontier
    if (tile_render_flags[tile_idx] == 0u) return;

    float2 jitter = float2(
        frac(halton(push.frame + 1u, 2u) + float(pixel.x * 1619u + pixel.y * 1021u) * 0.000001f),
        frac(halton(push.frame + 1u, 3u) + float(pixel.x * 2357u + pixel.y * 1637u) * 0.000001f)
    );
    Ray ray = makeRay(pixel, dims, jitter);
    float3 throughput = float3(1, 1, 1);
    float3 radiance = float3(0, 0, 0);

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
        Hit hit; GPUMaterial mat;
        if (!intersectScene(ray, hit, mat)) {
            radiance += throughput * skyColor(ray.dir);
            break;
        }

        float3 emissive_sample =
            sample_tex(mat.emissive_tex_index, hit.uv, float4(1, 1, 1, 1)).rgb;
        radiance += throughput * emissive_sample * mat.emission_color.rgb * mat.emission_luminance;

        float3 hitPos = ray.origin + hit.t * ray.dir;
        float3 view_dir = -ray.dir;
        float3 n_pbr = hit.front_face ? hit.normal : -hit.normal;

        OpenPBR_ResolvedInputs inputs = material_to_openpbr(mat, hit.uv, n_pbr, hit.tangent, hit.bitangent);
        OpenPBR_PreparedBsdf prepared = openpbr_prepare_bsdf_and_volume(
            inputs, throughput,
            OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, view_dir);

        // Direct lighting (NEE)
        bool sun_front = dot(hit.normal, g_sun_dir) > 0.f;
        bool sun_back = dot(hit.normal, -g_sun_dir) > 0.f && inputs.transmission_weight > 0.f;
        if (sun_front && !is_occluded(hitPos, hit.normal, g_sun_dir, T_MAX))
            radiance += throughput * openpbr_get_sum_of_diffuse_specular(openpbr_eval(prepared, g_sun_dir)) * g_sun_color * g_sun_intensity;
        if (sun_back && !is_occluded(hitPos, -hit.normal, -g_sun_dir, T_MAX))
            radiance += throughput * openpbr_get_sum_of_diffuse_specular(openpbr_eval(prepared, -g_sun_dir)) * g_sun_color * g_sun_intensity;

        // BSDF sample
        float3 light_dir; OpenPBR_DiffuseSpecular weight; float pdf; OpenPBR_BsdfLobeType sampled_type;
        openpbr_sample(prepared, rng.next3(), light_dir, weight, pdf, sampled_type);
        if (pdf <= 0.f) break;

        throughput *= openpbr_get_sum_of_diffuse_specular(weight);

        // Russian roulette
        if (bounce >= 3) {
            float p = max(throughput.r, max(throughput.g, throughput.b));
            if (rng.nextFloat() > p) break;
            throughput /= p;
        }

        float3 offset_n = dot(light_dir, hit.normal) >= 0.f ? hit.normal : -hit.normal;
        ray.origin = hitPos + offset_n * T_MIN;
        ray.dir = light_dir;
    }

    // Accumulate
    float3 prev = accum_image[pixel].rgb;
    float blend = 1.f / float(push.frame + 1u);
    accum_image[pixel] = float4(lerp(prev, radiance, blend), 1.f);

    // Write display
    int vis_id = object_id_image[pixel];
    float3 display = (push.debug_view_mode == 1)
                         ? debugColorForObjectId(vis_id)
                         : accum_image[pixel].rgb;
    if (isSelectionOutline(pixel, dims, vis_id))
        display = lerp(display, push.outline_color.rgb, saturate(push.outline_color.a));
    output_image[pixel] = float4(display, 1.f);
}

// ── Entry point ───────────────────────────────────────────────────────────────
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_id: SV_DispatchThreadID,
          uint3 group_id: SV_GroupID,
          uint group_index: SV_GroupIndex)
{
    uint2 pixel = dispatch_id.xy;
    uint2 dims;
    output_image.GetDimensions(dims.x, dims.y);
    if (any(pixel >= dims)) return;

    uint tiles_x = (dims.x + 15u) / 16u;
    uint tile_idx = group_id.y * tiles_x + group_id.x;

    RNG rng = RNG::make(pixel, push.frame);

    if (push.stage == 0u)
        stage0(pixel, dims, group_index, group_id, rng);
    else
        stage1(pixel, dims, tile_idx, rng);
}