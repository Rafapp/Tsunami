#include "pathtracer_shared_prefix.slang"

[[vk::binding(14, 0)]] RWStructuredBuffer<uint>       hipr_visible_counts;
[[vk::binding(15, 0)]] RWStructuredBuffer<uint>       hipr_secondary_counts;
[[vk::binding(16, 0)]] RWStructuredBuffer<uint>       hipr_shadow_counts;
[[vk::binding(17, 0)]] RWStructuredBuffer<float>      hipr_scores;
[[vk::binding(18, 0)]] RWStructuredBuffer<int>        hipr_order;
[[vk::binding(19, 0)]] StructuredBuffer<WaterSurfaceParams> water_surface_params_buf;
[[vk::binding(20, 0)]] RWTexture2D<float>                  water_height_image;
[[vk::binding(21, 0)]] RWTexture2D<float>                  water_prev_height_image;
[[vk::binding(22, 0)]] RWTexture2D<float4>                 accum_history_image;
[[vk::binding(23, 0)]] RWTexture2D<int>                    object_id_history_image;

// --- GPU structs: must match C++ layout exactly ---
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

// --- Constants ---
static const float PI          = 3.14159265358979f;
static const float T_MIN       = 1e-3f;
static const float T_MAX       = 1e30f;
static const uint  NO_TEXTURE  = 0xFFFFFFFFu;
static const float WATER_HEIGHT_SENTINEL = -1.0e6f;
static const float WATER_HEIGHT_INVALID_THRESHOLD = -5.0e5f;
static const uint WATER_PAUSE_BIT = 0x80000000u;
static const uint SCENE_DYNAMIC_BIT = 0x40000000u;
static const uint WATER_SPP_MASK  = 0x0000FFFFu;

bool waterPaused() {
    return (push.hipr_reserved0 & WATER_PAUSE_BIT) != 0u;
}

uint waterSppOverride() {
    return push.hipr_reserved0 & WATER_SPP_MASK;
}

bool sceneHasDynamicUpdates() {
    return (push.hipr_reserved0 & SCENE_DYNAMIC_BIT) != 0u;
}

// --- Light: directional sun ---
// Direction and intensity come from push constants (UI-controlled).
// g_sun_dir is the direction FROM the surface TO the light (positive Y = above).
static const float3 g_sun_color = float3(1.0f, 0.85f, 0.5f);

// --- Types ---
struct Ray { float3 origin; float3 dir; };
struct Hit {
    float  t;
    float3 position;
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

// --- Halton low-discrepancy jitter ---
// Radical inverse base 2 via bit reversal - O(1), exact.
float halton2(uint n) { return float(reversebits(n)) * (1.f / 4294967296.f); }

// Radical inverse base 3 - converges in <=12 iterations for n < 3^12 ~ 530k.
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

// --- RNG ---
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

// --- Geometry helpers ---
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

float powerHeuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / max(a2 + b2, 1.0e-8f);
}

float3 clampRadianceContribution(float3 value) {
    const float clamp_value = 28.0f;
    return min(max(value, float3(0.0f, 0.0f, 0.0f)),
               float3(clamp_value, clamp_value, clamp_value));
}

float3 clampPathThroughput(float3 value) {
    const float clamp_value = 10.0f;
    return min(max(value, float3(0.0f, 0.0f, 0.0f)),
               float3(clamp_value, clamp_value, clamp_value));
}

float3 sampleUniformCone(float2 u, float3 axis, float cos_theta_max) {
    float phi = 2.0f * PI * u.x;
    float cos_theta = lerp(cos_theta_max, 1.0f, u.y);
    float sin_theta = sqrt(max(1.0f - cos_theta * cos_theta, 0.0f));

    float3 tangent = buildOrthoTangent(axis);
    float3 bitangent = normalize(cross(axis, tangent));
    return normalize(tangent * (cos(phi) * sin_theta) +
                     bitangent * (sin(phi) * sin_theta) +
                     axis * cos_theta);
}

bool directionInCone(float3 dir, float3 axis, float cos_theta_max) {
    return dot(normalize(dir), normalize(axis)) >= cos_theta_max;
}

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

WaterSurfaceParams getWaterSurfaceParams() {
    return water_surface_params_buf[0];
}

bool waterEnabled() {
    WaterSurfaceParams water = getWaterSurfaceParams();
    return water.water_enabled != 0 &&
           water.axis_u_half_extent.w > 1e-4f &&
           water.axis_v_half_extent.w > 1e-4f &&
           water.water_height_to_world_scale > 1e-5f;
}

float3 worldToWater(float3 world_pos) {
    WaterSurfaceParams water = getWaterSurfaceParams();
    float3 delta = world_pos - water.center_trace_half_height.xyz;
    return float3(
        dot(delta, water.axis_u_half_extent.xyz) / max(water.axis_u_half_extent.w, 1e-4f),
        dot(delta, water.normal.xyz),
        dot(delta, water.axis_v_half_extent.xyz) / max(water.axis_v_half_extent.w, 1e-4f));
}

float3 waterToWorld(float3 water_pos) {
    WaterSurfaceParams water = getWaterSurfaceParams();
    return water.center_trace_half_height.xyz +
           water.axis_u_half_extent.xyz * (water_pos.x * water.axis_u_half_extent.w) +
           water.normal.xyz * water_pos.y +
           water.axis_v_half_extent.xyz * (water_pos.z * water.axis_v_half_extent.w);
}

float readWaterHeightRaw(int2 coord, int2 dims) {
    return water_height_image[clamp(coord, int2(0, 0), dims - 1)];
}

bool isValidWaterHeight(float value) {
    return value > WATER_HEIGHT_INVALID_THRESHOLD;
}

float sanitizeWaterHeight(float value, float fallback) {
    return isValidWaterHeight(value) ? value : fallback;
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

    float w00 = (1.0f - f.x) * (1.0f - f.y);
    float w10 = f.x * (1.0f - f.y);
    float w01 = (1.0f - f.x) * f.y;
    float w11 = f.x * f.y;

    float weight = 0.0f;
    float value = 0.0f;
    if (isValidWaterHeight(h00)) { value += h00 * w00; weight += w00; }
    if (isValidWaterHeight(h10)) { value += h10 * w10; weight += w10; }
    if (isValidWaterHeight(h01)) { value += h01 * w01; weight += w01; }
    if (isValidWaterHeight(h11)) { value += h11 * w11; weight += w11; }
    if (weight <= 1.0e-6f) {
        return WATER_HEIGHT_SENTINEL;
    }
    return value / weight;
}

float sampleWaterHeight(float2 water_xz, int2 dims) {
    float2 uv = water_xz * 0.5f + 0.5f;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        return WATER_HEIGHT_SENTINEL;
    }

    float sampled = sampleWaterHeightRaw(uv, dims);
    if (!isValidWaterHeight(sampled)) {
        return WATER_HEIGHT_SENTINEL;
    }
    return sampled * getWaterSurfaceParams().water_height_to_world_scale;
}

float3 waterSurfaceNormal(float2 water_xz, int2 dims) {
    float dx = 2.0f / max(float(dims.x), 1.0f);
    float dz = 2.0f / max(float(dims.y), 1.0f);

    float center = sampleWaterHeight(water_xz, dims);
    if (!isValidWaterHeight(center)) {
        return getWaterSurfaceParams().normal.xyz;
    }
    float hL = sanitizeWaterHeight(sampleWaterHeight(water_xz - float2(dx, 0.0f), dims), center);
    float hR = sanitizeWaterHeight(sampleWaterHeight(water_xz + float2(dx, 0.0f), dims), center);
    float hD = sanitizeWaterHeight(sampleWaterHeight(water_xz - float2(0.0f, dz), dims), center);
    float hU = sanitizeWaterHeight(sampleWaterHeight(water_xz + float2(0.0f, dz), dims), center);

    float3 pL = waterToWorld(float3(water_xz.x - dx, hL, water_xz.y));
    float3 pR = waterToWorld(float3(water_xz.x + dx, hR, water_xz.y));
    float3 pD = waterToWorld(float3(water_xz.x, hD, water_xz.y - dz));
    float3 pU = waterToWorld(float3(water_xz.x, hU, water_xz.y + dz));

    float3 normal = normalize(cross(pR - pL, pU - pD));
    if (dot(normal, getWaterSurfaceParams().normal.xyz) < 0.0f) {
        normal = -normal;
    }
    return normal;
}

float readPreviousWaterHeightRaw(int2 coord, int2 dims) {
    return water_prev_height_image[clamp(coord, int2(0, 0), dims - 1)];
}

float samplePreviousWaterHeightRaw(float2 uv, int2 dims) {
    uv = clamp(uv, float2(0.0f, 0.0f), float2(1.0f, 1.0f));

    float2 texel = uv * float2(dims - 1);
    int2 p0 = int2(floor(texel));
    int2 p1 = min(p0 + int2(1, 1), dims - 1);
    float2 f = texel - float2(p0);

    float h00 = readPreviousWaterHeightRaw(p0, dims);
    float h10 = readPreviousWaterHeightRaw(int2(p1.x, p0.y), dims);
    float h01 = readPreviousWaterHeightRaw(int2(p0.x, p1.y), dims);
    float h11 = readPreviousWaterHeightRaw(p1, dims);

    float w00 = (1.0f - f.x) * (1.0f - f.y);
    float w10 = f.x * (1.0f - f.y);
    float w01 = (1.0f - f.x) * f.y;
    float w11 = f.x * f.y;

    float weight = 0.0f;
    float value = 0.0f;
    if (isValidWaterHeight(h00)) { value += h00 * w00; weight += w00; }
    if (isValidWaterHeight(h10)) { value += h10 * w10; weight += w10; }
    if (isValidWaterHeight(h01)) { value += h01 * w01; weight += w01; }
    if (isValidWaterHeight(h11)) { value += h11 * w11; weight += w11; }
    if (weight <= 1.0e-6f) {
        return WATER_HEIGHT_SENTINEL;
    }
    return value / weight;
}

float samplePreviousWaterHeight(float2 water_xz, int2 dims) {
    float2 uv = water_xz * 0.5f + 0.5f;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        return WATER_HEIGHT_SENTINEL;
    }

    float sampled = samplePreviousWaterHeightRaw(uv, dims);
    if (!isValidWaterHeight(sampled)) {
        return WATER_HEIGHT_SENTINEL;
    }
    return sampled * getWaterSurfaceParams().water_height_to_world_scale;
}

float3 previousWaterSurfaceNormal(float2 water_xz, int2 dims) {
    float dx = 2.0f / max(float(dims.x), 1.0f);
    float dz = 2.0f / max(float(dims.y), 1.0f);

    float center = samplePreviousWaterHeight(water_xz, dims);
    if (!isValidWaterHeight(center)) {
        return getWaterSurfaceParams().normal.xyz;
    }
    float hL =
        sanitizeWaterHeight(samplePreviousWaterHeight(water_xz - float2(dx, 0.0f), dims), center);
    float hR =
        sanitizeWaterHeight(samplePreviousWaterHeight(water_xz + float2(dx, 0.0f), dims), center);
    float hD =
        sanitizeWaterHeight(samplePreviousWaterHeight(water_xz - float2(0.0f, dz), dims), center);
    float hU =
        sanitizeWaterHeight(samplePreviousWaterHeight(water_xz + float2(0.0f, dz), dims), center);

    float3 pL = waterToWorld(float3(water_xz.x - dx, hL, water_xz.y));
    float3 pR = waterToWorld(float3(water_xz.x + dx, hR, water_xz.y));
    float3 pD = waterToWorld(float3(water_xz.x, hD, water_xz.y - dz));
    float3 pU = waterToWorld(float3(water_xz.x, hU, water_xz.y + dz));

    float3 normal = normalize(cross(pR - pL, pU - pD));
    if (dot(normal, getWaterSurfaceParams().normal.xyz) < 0.0f) {
        normal = -normal;
    }
    return normal;
}

bool isFiniteColor(float3 color) {
    return all(!isnan(color)) && all(!isinf(color));
}

float3 sanitizeColor(float3 color, float3 fallback) {
    if (!isFiniteColor(color)) {
        return max(fallback, 0.0f.xxx);
    }
    return max(color, 0.0f.xxx);
}

void waterTemporalInfoForHit(Hit water_hit, out float motion, out float blend) {
    uint width = 0;
    uint height = 0;
    water_height_image.GetDimensions(width, height);
    int2 dims = int2(int(width), int(height));
    if (dims.x <= 1 || dims.y <= 1) {
        motion = 1.0f;
        blend = 1.0f;
        return;
    }
    if (waterPaused()) {
        motion = 0.0f;
        blend = 1.0f / float(push.frame + 1u);
        return;
    }
    float2 water_xz = water_hit.uv * 2.0f - 1.0f;
    float current_height = sampleWaterHeight(water_xz, dims);
    float previous_height = samplePreviousWaterHeight(water_xz, dims);
    if (!isValidWaterHeight(current_height) || !isValidWaterHeight(previous_height)) {
        motion = 1.0f;
        blend = 1.0f;
        return;
    }
    float height_delta = abs(current_height - previous_height);

    float3 current_normal = water_hit.front_face ? water_hit.normal : -water_hit.normal;
    float3 previous_normal = previousWaterSurfaceNormal(water_xz, dims);
    float normal_delta = 1.0f - saturate(dot(current_normal, previous_normal));
    float flatness = 1.0f - saturate(dot(current_normal, getWaterSurfaceParams().normal.xyz));

    motion = saturate(height_delta * 8.0f + normal_delta * 3.2f + flatness * 1.2f);
    float history_blend = lerp(0.045f, 0.22f, motion);
    float startup_blend = 1.0f / float(min(push.frame + 1u, 12u));
    blend = max(history_blend, startup_blend);
}

bool projectWorldToScreenUv(GPUCamera cam, float3 world_position, uint2 dims, out float2 uv) {
    uv = float2(0.0f, 0.0f);
    if (dims.x == 0u || dims.y == 0u) {
        return false;
    }

    float3 origin = cam.position.xyz;
    float3 target = cam.target.xyz;
    float3 up_vec = safeNormalize(cam.up.xyz, float3(0.0f, 1.0f, 0.0f));
    float3 forward = safeNormalize(target - origin, float3(0.0f, 0.0f, 1.0f));
    float3 right = safeNormalize(cross(forward, up_vec), buildOrthoTangent(forward));
    float3 up = safeNormalize(cross(right, forward), up_vec);

    float3 rel = world_position - origin;
    float z = dot(rel, forward);
    float near_clip = max(cam.fov_near_far.y, 1.0e-4f);
    if (z <= near_clip) {
        return false;
    }

    float aspect = float(dims.x) / max(float(dims.y), 1.0f);
    float half_h = tan(cam.fov_near_far.x * (PI / 180.f) * 0.5f);
    float half_w = aspect * half_h;
    if (half_h <= 1.0e-6f || half_w <= 1.0e-6f) {
        return false;
    }

    float ndc_x = dot(rel, right) / (z * half_w);
    float ndc_y = dot(rel, up) / (z * half_h);

    uv = float2(0.5f * (ndc_x + 1.0f), 0.5f * (1.0f - ndc_y));
    return uv.x >= 0.0f && uv.x <= 1.0f && uv.y >= 0.0f && uv.y <= 1.0f;
}

float3 sampleAccumHistory(float2 uv, uint2 dims) {
    int2 dims_i = int2(int(dims.x), int(dims.y));
    uv = clamp(uv, float2(0.0f, 0.0f), float2(1.0f, 1.0f));
    float2 texel = uv * float2(dims_i - 1);
    int2 p0 = int2(floor(texel));
    int2 p1 = min(p0 + int2(1, 1), dims_i - 1);
    float2 f = texel - float2(p0);

    float3 c00 = accum_history_image[p0].rgb;
    float3 c10 = accum_history_image[int2(p1.x, p0.y)].rgb;
    float3 c01 = accum_history_image[int2(p0.x, p1.y)].rgb;
    float3 c11 = accum_history_image[p1].rgb;

    float3 cx0 = lerp(c00, c10, f.x);
    float3 cx1 = lerp(c01, c11, f.x);
    return sanitizeColor(lerp(cx0, cx1, f.y), c00);
}

int sampleObjectIdHistory(float2 uv, uint2 dims) {
    int2 dims_i = int2(int(dims.x), int(dims.y));
    uv = clamp(uv, float2(0.0f, 0.0f), float2(1.0f, 1.0f));
    int2 pixel = int2(floor(uv * float2(dims_i - 1) + 0.5f));
    pixel = clamp(pixel, int2(0, 0), dims_i - 1);
    return object_id_history_image[pixel];
}

bool sampleReprojectedWaterHistory(Hit water_hit, uint2 dims, out float3 history_color,
                                   out int history_object_id, out float confidence) {
    history_color = float3(0.0f, 0.0f, 0.0f);
    history_object_id = -1;
    confidence = 0.0f;

    if (push.frame == 0u || waterPaused() || !waterEnabled()) {
        return false;
    }

    uint water_width = 0u;
    uint water_height = 0u;
    water_height_image.GetDimensions(water_width, water_height);
    int2 water_dims = int2(int(water_width), int(water_height));
    if (water_dims.x <= 1 || water_dims.y <= 1) {
        return false;
    }

    float2 water_xz = water_hit.uv * 2.0f - 1.0f;
    float current_height = sampleWaterHeight(water_xz, water_dims);
    float previous_height = samplePreviousWaterHeight(water_xz, water_dims);

    float3 previous_world = water_hit.position;
    if (isValidWaterHeight(current_height) && isValidWaterHeight(previous_height)) {
        previous_world = waterToWorld(float3(water_xz.x, previous_height, water_xz.y));
        confidence = 1.0f - saturate(abs(current_height - previous_height) * 14.0f);
    } else {
        confidence = 0.25f;
    }

    float2 previous_uv = float2(0.0f, 0.0f);
    GPUCamera previous_camera = camera_buf[NonUniformResourceIndex(1)];
    if (!projectWorldToScreenUv(previous_camera, previous_world, dims, previous_uv)) {
        return false;
    }

    history_object_id = sampleObjectIdHistory(previous_uv, dims);
    if (history_object_id != getWaterSurfaceParams().water_object_id) {
        return false;
    }

    history_color = sampleAccumHistory(previous_uv, dims);
    return isFiniteColor(history_color);
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
    const int water_material_index = getWaterSurfaceParams().water_material_index;
    if (water_material_index >= 0 && uint(water_material_index) < push.material_count) {
        water = materials_buf[NonUniformResourceIndex(water_material_index)];
    }

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

    WaterSurfaceParams water = getWaterSurfaceParams();
    float3 local_origin = worldToWater(ray.origin);
    float3 local_dir = float3(
        dot(ray.dir, water.axis_u_half_extent.xyz) / max(water.axis_u_half_extent.w, 1e-4f),
        dot(ray.dir, water.normal.xyz),
        dot(ray.dir, water.axis_v_half_extent.xyz) / max(water.axis_v_half_extent.w, 1e-4f));

    float t_min;
    float t_max;
    if (!intersectAABB(local_origin, local_dir,
                       float3(-1.04f, -water.center_trace_half_height.w, -1.04f),
                       float3(1.04f, water.center_trace_half_height.w, 1.04f),
                       t_min, t_max)) {
        return false;
    }

    t_min = max(t_min, T_MIN);
    float grazing = 1.0f - saturate(abs(local_dir.y));
    int coarse_steps = clamp(int(round(lerp(18.0f, 44.0f, grazing))), 18, 44);

    float prev_t = t_min;
    float3 prev_pos = ray.origin + ray.dir * prev_t;
    float3 prev_local = worldToWater(prev_pos);
    float prev_height = sampleWaterHeight(prev_local.xz, dims);
    bool prev_valid = isValidWaterHeight(prev_height);
    float prev_delta = prev_local.y - prev_height;

    [loop]
    for (int step = 1; step <= coarse_steps; ++step) {
        float t = lerp(t_min, t_max, float(step) / float(coarse_steps));
        float3 pos = ray.origin + ray.dir * t;
        float3 local = worldToWater(pos);
        float sampled_height = sampleWaterHeight(local.xz, dims);
        bool valid = isValidWaterHeight(sampled_height);
        float delta = local.y - sampled_height;

        if (prev_valid && valid &&
            (prev_delta == 0.0f || delta == 0.0f || prev_delta * delta <= 0.0f)) {
            float lo = prev_t;
            float hi = t;
            float lo_delta = prev_delta;

            [unroll]
            for (int refine = 0; refine < 5; ++refine) {
                float mid = 0.5f * (lo + hi);
                float3 mid_local = worldToWater(ray.origin + ray.dir * mid);
                float mid_height = sampleWaterHeight(mid_local.xz, dims);
                if (!isValidWaterHeight(mid_height)) {
                    hi = mid;
                    continue;
                }
                float mid_delta = mid_local.y - mid_height;
                if (lo_delta == 0.0f || lo_delta * mid_delta <= 0.0f) {
                    hi = mid;
                } else {
                    lo = mid;
                    lo_delta = mid_delta;
                }
            }

            hit.t = 0.5f * (lo + hi);
            float3 hit_local = worldToWater(ray.origin + ray.dir * hit.t);
            float hit_height = sampleWaterHeight(hit_local.xz, dims);
            if (!isValidWaterHeight(hit_height)) {
                break;
            }
            hit_local.y = hit_height;
            hit.position = waterToWorld(hit_local);

            float3 geometric_normal = waterSurfaceNormal(hit_local.xz, dims);
            float3 tangent = normalize(
                water.axis_u_half_extent.xyz -
                geometric_normal * dot(geometric_normal, water.axis_u_half_extent.xyz));
            float3 bitangent = normalize(cross(geometric_normal, tangent));
            if (dot(bitangent, water.axis_v_half_extent.xyz) < 0.0f) {
                bitangent = -bitangent;
            }

            bool front_face = dot(ray.dir, geometric_normal) < 0.0f;
            hit.normal = front_face ? geometric_normal : -geometric_normal;
            hit.geom_normal = hit.normal;
            hit.tangent = tangent;
            hit.bitangent = bitangent;
            hit.uv = hit_local.xz * 0.5f + 0.5f;
            hit.front_face = front_face;
            hit.matIndex = water.water_material_index;
            hit.instanceID = water.water_object_id;
            hit.primID = -1;
            mat = loadWaterMaterial();
            return true;
        }

        prev_t = t;
        prev_delta = delta;
        prev_valid = valid;
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
    a.p0 = vertices_buf[voff+i0].position;
    a.p1 = vertices_buf[voff+i1].position;
    a.p2 = vertices_buf[voff+i2].position;
    return a;
}

// --- Shadow helper ---
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

bool traceVisibleMeshHit(Ray ray, out float hit_t, out int instance_id) {
    hit_t = T_MAX;
    instance_id = -1;

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
    hit_t = q.CommittedRayT();
    instance_id = int(q.CommittedInstanceID());
    return true;
}

int traceVisibleInstance(Ray ray, out bool visible_water, out Hit visible_water_hit) {
    float mesh_hit_t = T_MAX;
    int mesh_instance_id = -1;
    bool hit_mesh = traceVisibleMeshHit(ray, mesh_hit_t, mesh_instance_id);

    Hit water_hit;
    GPUMaterial water_mat;
    bool hit_water = intersectWaterSurface(ray, water_hit, water_mat);
    visible_water = false;
    visible_water_hit = (Hit)0;
    const int water_mesh_object_id = getWaterSurfaceParams().water_material_index;
    const bool mesh_masks_water_hit =
        hit_mesh && mesh_instance_id == water_mesh_object_id;
    if (hit_water && mesh_masks_water_hit) {
        visible_water = true;
        visible_water_hit = water_hit;
        return water_hit.instanceID;
    }

    return hit_mesh ? mesh_instance_id : -1;
}

int traceVisibleInstance(Ray ray) {
    bool visible_water;
    Hit visible_water_hit;
    return traceVisibleInstance(ray, visible_water, visible_water_hit);
}

// --- Scene intersection ---
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
    bool queryFrontFace = q.CommittedTriangleFrontFace();

    VertexAttribs attr = fetchAttribs(instID, primID, bary);

    hit.position   = ray.origin + hit.t * ray.dir;
    hit.uv         = attr.uv;
    hit.instanceID = int(instID);
    hit.primID     = int(primID);
    hit.matIndex   = meshes_buf[NonUniformResourceIndex(instID)].matIndex;
    mat = materials_buf[NonUniformResourceIndex(hit.matIndex)];

    // Vertex normals/tangents are in local (BLAS) space.
    // The TLAS instance transform moves positions to world space, so we must
    // bring the TBN basis to world space too:
    //   normals  -> multiply by inverse-transpose of model matrix
    //             = mul(n_local, inverseTransform)  [row-vec x mat in Slang]
    //   tangents -> multiply by the forward model matrix (no inverse needed)
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
    if (mat.normal_tex_index != NO_TEXTURE) {
        float3 normal_tex = sample_tex(
            mat.normal_tex_index,
            attr.uv,
            float4(0.5, 0.5, 1.0, 1.0)
        ).xyz;

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

// --- Texture sampling ---
bool intersectScene(Ray ray, out Hit hit, out GPUMaterial mat) {
    Hit mesh_hit;
    GPUMaterial mesh_mat;
    bool hit_mesh = intersectMeshScene(ray, mesh_hit, mesh_mat);

    Hit water_hit;
    GPUMaterial water_mat;
    bool hit_water = intersectWaterSurface(ray, water_hit, water_mat);

    const int water_mesh_object_id = getWaterSurfaceParams().water_material_index;
    const bool mesh_masks_water_hit =
        hit_mesh && mesh_hit.instanceID == water_mesh_object_id;
    if (hit_water && mesh_masks_water_hit) {
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

float4 sample_tex(uint index, float2 uv, float4 fallback) {
    if (index == NO_TEXTURE) return fallback;
    return material_textures[NonUniformResourceIndex(index)].SampleLevel(
        g_material_sampler, uv, 0.f);
}

// --- Fill OpenPBR inputs ---
OpenPBR_ResolvedInputs material_to_openpbr(
    GPUMaterial m, float2 uv, float3 n, float3 t, float3 b)
{
    OpenPBR_ResolvedInputs inp = openpbr_make_default_resolved_inputs();

    float4 alb = sample_tex(m.albedo_tex_index, uv, float4(1.f, 1.f, 1.f, 1.f));
    // glTF ORM texture: R=occlusion, G=roughness, B=metalness.
    // The CPU values act as live multipliers so GUI edits still affect
    // textured materials.
    float4 orm = sample_tex(m.roughness_tex_index, uv, float4(1.f, 1.f, 1.f, 1.f));

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

// --- Environment ---
float3 skyColor(float3 dir) {
    float t = 0.5f * (normalize(dir).y + 1.f);
    return lerp(float3(1, 1, 1), float3(0.5f, 0.7f, 1.f), t);
}

// --- Path tracing ---
float3 tracePath(Ray ray, inout RNG rng, bool collect_influence) {
    float3 throughput = float3(1, 1, 1);
    float3 radiance   = float3(0, 0, 0);
    float  last_bsdf_pdf   = 0.0f;
    bool   has_bsdf_sample = false;

    const float sun_half_angle_rad = 0.004675f; // ~0.268 degrees apparent solar radius.
    const float sun_cos_theta_max = cos(sun_half_angle_rad);
    const float sun_solid_angle =
        2.0f * PI * max(1.0f - sun_cos_theta_max, 1.0e-6f);
    const float sun_pdf = rcp(sun_solid_angle);
    const float3 g_sun_dir = normalize(float3(push.sun_dir_x, push.sun_dir_y, push.sun_dir_z));
    // Convert directional-light intensity to finite-disc radiance for cone MIS.
    const float3 sun_radiance = g_sun_color * (push.sun_intensity / sun_solid_angle);

    const uint bounce_count = max(push.max_bounces, 1u);
    [loop]
    for (uint bounce = 0u; bounce < bounce_count; ++bounce) {
        Hit         hit;
        GPUMaterial mat;

        if (!intersectScene(ray, hit, mat)) {
            if (push.skybox_enabled != 0u) {
                radiance += throughput * clampRadianceContribution(skyColor(ray.dir));
            }
            if (push.directional_light_enabled != 0u &&
                (directionInCone(ray.dir, g_sun_dir, sun_cos_theta_max) ||
                 directionInCone(ray.dir, -g_sun_dir, sun_cos_theta_max))) {
                float mis = has_bsdf_sample ?
                    powerHeuristic(max(last_bsdf_pdf, 0.0f), sun_pdf) :
                    1.0f;
                radiance += throughput * clampRadianceContribution(sun_radiance * mis);
            }
            break;
        }

        if (collect_influence && bounce > 0 && hit.instanceID >= 0 &&
            uint(hit.instanceID) < push.hipr_object_count) {
            InterlockedAdd(hipr_secondary_counts[uint(hit.instanceID)], 1);
        }

        float3 emissive_sample =
            sample_tex(mat.emissive_tex_index, hit.uv, float4(1.f, 1.f, 1.f, 1.f)).rgb;
        float3 emissive =
            emissive_sample * mat.emission_color.rgb * mat.emission_luminance;
        radiance += throughput * clampRadianceContribution(emissive);

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

        // Direct sun lighting: finite cone sample + MIS with BSDF PDF.
        // Keep transmissive back-hemisphere behavior for backlit glass.
        if (push.directional_light_enabled != 0u) {
            bool sun_front = dot(hit.normal,  g_sun_dir) > 0.f;
            bool sun_back  = dot(hit.normal, -g_sun_dir) > 0.f &&
                             inputs.transmission_weight > 0.f;

            if (sun_front) {
                float2 sun_rand = float2(rng.nextFloat(), rng.nextFloat());
                float3 sun_dir = sampleUniformCone(sun_rand, g_sun_dir, sun_cos_theta_max);
                int occluder_front = traceOccluderInstance(hitPos, hit.normal, sun_dir, T_MAX);
                if (collect_influence && occluder_front >= 0 &&
                    uint(occluder_front) < push.hipr_object_count) {
                    InterlockedAdd(hipr_shadow_counts[uint(occluder_front)], 1);
                }
                if (occluder_front < 0) {
                    float3 f = openpbr_get_sum_of_diffuse_specular(openpbr_eval(prepared, sun_dir));
                    float bsdf_pdf = max(openpbr_pdf(prepared, sun_dir), 0.0f);
                    float mis = powerHeuristic(sun_pdf, bsdf_pdf);
                    float3 direct = f * sun_radiance * (mis / sun_pdf);
                    radiance += throughput * clampRadianceContribution(direct);
                }
            }
            // For back-hemisphere shadow rays, offset the origin toward the sun side
            // so the ray does not self-intersect with the surface it just hit.
            if (sun_back) {
                float2 sun_rand = float2(rng.nextFloat(), rng.nextFloat());
                float3 sun_dir = sampleUniformCone(sun_rand, -g_sun_dir, sun_cos_theta_max);
                int occluder_back = traceOccluderInstance(hitPos, -hit.normal, sun_dir, T_MAX);
                if (collect_influence && occluder_back >= 0 &&
                    uint(occluder_back) < push.hipr_object_count) {
                    InterlockedAdd(hipr_shadow_counts[uint(occluder_back)], 1);
                }
                if (occluder_back < 0) {
                    float3 f = openpbr_get_sum_of_diffuse_specular(openpbr_eval(prepared, sun_dir));
                    float bsdf_pdf = max(openpbr_pdf(prepared, sun_dir), 0.0f);
                    float mis = powerHeuristic(sun_pdf, bsdf_pdf);
                    float3 direct = f * sun_radiance * (mis / sun_pdf);
                    radiance += throughput * clampRadianceContribution(direct);
                }
            }
        }

        float3               light_dir;
        OpenPBR_DiffuseSpecular weight;
        float                pdf;
        OpenPBR_BsdfLobeType sampled_type;
        openpbr_sample(prepared, rng.next3(), light_dir, weight, pdf, sampled_type);

        if (pdf <= 0.f) break;

        throughput *= openpbr_get_sum_of_diffuse_specular(weight);
        throughput = clampPathThroughput(throughput);
        last_bsdf_pdf = pdf;
        has_bsdf_sample = true;

        if (bounce >= 2u) {
            float p = max(throughput.r, max(throughput.g, throughput.b));
            if (rng.nextFloat() > p) break;
            throughput /= p;
            throughput = clampPathThroughput(throughput);
        }

        float  ray_eps  = max(T_MIN, hit.t * 1.0e-4f);
        float3 offset_n =
            dot(light_dir, hit.geom_normal) >= 0.0f ? hit.geom_normal : -hit.geom_normal;
        ray.origin = hitPos + offset_n * ray_eps;
        ray.dir    = light_dir;
    }

    return radiance;
}
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
