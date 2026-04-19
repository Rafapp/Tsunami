struct PushConstants {
    float time_seconds;
    float delta_time;
    float propagation;
    float damping;
    float restoring_force;
    float audio_level;
    float height_scale;
    float ripple_radius;
    float impulse_strength;
    float impulse_frequency_hz;
    float emitter_u;
    float emitter_v;
    uint floating_object_count;
    uint floating_object_interaction_count;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

struct FloatingObjectGpu
{
    float4 center;
    float4 axis_x;
    float4 axis_y;
    float4 axis_z;
    float4 half_extents;
    float4 color;
};

struct FloatingObjectInteractionGpu
{
    float4 center_radius_strength;
    float4 wake_direction_length;
};

[[vk::binding(0, 0)]] RWTexture2D<float4> output_image;
[[vk::binding(1, 0)]] RWTexture2D<float> current_height;
[[vk::binding(2, 0)]] RWTexture2D<float> previous_height;
[[vk::binding(3, 0)]] StructuredBuffer<FloatingObjectGpu> floating_objects;
[[vk::binding(4, 0)]] StructuredBuffer<FloatingObjectInteractionGpu> floating_object_interactions;
[[vk::binding(5, 0)]] StructuredBuffer<float> water_domain_mask;

static const float PI = 3.14159265358979323846f;
static const float KELVIN_WAKE_HALF_ANGLE = 19.47f * (PI / 180.0f);
static const float MAX_STABLE_WORLD_HEIGHT = 0.24f;
static const float MAX_STABLE_WORLD_VELOCITY = 0.18f;
static const float MAX_STABLE_WORLD_SLOPE = 2.25f;
static const float HULL_WATER_CLEARANCE = 0.008f;
static const float WATER_HEIGHT_SENTINEL = -1.0e6f;
static const float WATER_HEIGHT_INVALID_THRESHOLD = -5.0e5f;
static const float WATER_DOMAIN_ACTIVE_THRESHOLD = 0.5f;

int2 clampCoord(int2 coord, int2 dims)
{
    return clamp(coord, int2(0, 0), dims - 1);
}

uint domainMaskIndex(int2 coord, int2 dims)
{
    return uint(coord.y * dims.x + coord.x);
}

float readDomainMask(int2 coord, int2 dims)
{
    int2 clamped = clampCoord(coord, dims);
    return water_domain_mask[domainMaskIndex(clamped, dims)];
}

bool isDomainActive(int2 coord, int2 dims)
{
    return readDomainMask(coord, dims) > WATER_DOMAIN_ACTIVE_THRESHOLD;
}

bool isValidWaterHeight(float value)
{
    return value > WATER_HEIGHT_INVALID_THRESHOLD;
}

float sanitizeWaterHeight(float value, float fallback)
{
    return isValidWaterHeight(value) ? value : fallback;
}

float readCurrentHeight(int2 coord, int2 dims)
{
    int2 clamped = clampCoord(coord, dims);
    if (!isDomainActive(clamped, dims)) {
        return WATER_HEIGHT_SENTINEL;
    }

    float value = current_height[clamped];
    return isValidWaterHeight(value) ? value : WATER_HEIGHT_SENTINEL;
}

float readCurrentHeightForSolver(int2 coord, int2 dims, float fallback)
{
    float sampled = readCurrentHeight(coord, dims);
    return sanitizeWaterHeight(sampled, fallback);
}

float sampleCurrentHeight(float2 uv, int2 dims)
{
    uv = clamp(uv, float2(0.0f, 0.0f), float2(1.0f, 1.0f));

    float2 texel = uv * float2(dims - 1);
    int2 p0 = int2(floor(texel));
    int2 p1 = min(p0 + int2(1, 1), dims - 1);
    float2 f = texel - float2(p0);

    int2 c00 = p0;
    int2 c10 = int2(p1.x, p0.y);
    int2 c01 = int2(p0.x, p1.y);
    int2 c11 = p1;

    float h00 = readCurrentHeight(c00, dims);
    float h10 = readCurrentHeight(c10, dims);
    float h01 = readCurrentHeight(c01, dims);
    float h11 = readCurrentHeight(c11, dims);

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

float gaussian(float2 delta, float radius)
{
    float r2 = max(radius * radius, 1e-6f);
    return exp(-dot(delta, delta) / (2.0f * r2));
}

float impactKernel(float2 delta, float radius)
{
    float inner = gaussian(delta, radius * 0.55f);
    float outer = gaussian(delta, radius * 1.30f);
    return outer * 0.62f - inner;
}

float2 safeNormalize2D(float2 value, float2 fallback)
{
    float len = length(value);
    return len > 1e-5f ? value / len : fallback;
}

struct HorizontalFootprintGpu
{
    float2 axis_u;
    float2 axis_v;
    float2 half_extents;
};

HorizontalFootprintGpu computeHorizontalFootprint(FloatingObjectGpu object_data)
{
    float2 axis_x_xz = object_data.axis_x.xz;
    float2 axis_y_xz = object_data.axis_y.xz;
    float2 axis_z_xz = object_data.axis_z.xz;
    float3 half_extents = object_data.half_extents.xyz;

    HorizontalFootprintGpu footprint;
    footprint.axis_u = safeNormalize2D(axis_x_xz, float2(1.0f, 0.0f));
    float2 axis_v_candidate = axis_z_xz - footprint.axis_u * dot(axis_z_xz, footprint.axis_u);
    if (length(axis_v_candidate) <= 1e-5f) {
        axis_v_candidate = float2(-footprint.axis_u.y, footprint.axis_u.x);
    }
    footprint.axis_v = safeNormalize2D(axis_v_candidate, float2(0.0f, 1.0f));
    footprint.half_extents.x = abs(dot(axis_x_xz, footprint.axis_u)) * half_extents.x +
                               abs(dot(axis_y_xz, footprint.axis_u)) * half_extents.y +
                               abs(dot(axis_z_xz, footprint.axis_u)) * half_extents.z;
    footprint.half_extents.y = abs(dot(axis_x_xz, footprint.axis_v)) * half_extents.x +
                               abs(dot(axis_y_xz, footprint.axis_v)) * half_extents.y +
                               abs(dot(axis_z_xz, footprint.axis_v)) * half_extents.z;
    return footprint;
}

float objectHullDisplacementAtXZ(float2 xz)
{
    float displacement = 0.0f;

    [loop]
    for (uint object_index = 0; object_index < push.floating_object_count; ++object_index) {
        FloatingObjectGpu object_data = floating_objects[object_index];
        float hull_displacement = max(object_data.center.w, 0.0f);
        if (hull_displacement <= 1e-5f) {
            continue;
        }

        HorizontalFootprintGpu footprint = computeHorizontalFootprint(object_data);
        if (footprint.half_extents.x <= 1e-5f || footprint.half_extents.y <= 1e-5f) {
            continue;
        }

        float2 delta = xz - object_data.center.xz;
        float2 local = float2(dot(delta, footprint.axis_u), dot(delta, footprint.axis_v));
        float2 q = abs(local) - footprint.half_extents;
        float2 outside = max(q, float2(0.0f, 0.0f));
        float outside_distance = length(outside);
        float inside_distance = min(max(q.x, q.y), 0.0f);
        float signed_distance = outside_distance + inside_distance;
        float edge_softness = clamp(min(footprint.half_extents.x, footprint.half_extents.y) * 0.30f,
                                    0.03f, 0.09f);
        float inside_weight =
            1.0f - smoothstep(-edge_softness, edge_softness, signed_distance);
        float positive_distance = max(signed_distance, 0.0f);
        float rim_sigma = edge_softness * 1.35f;
        float rim_center = edge_softness * 1.15f;
        float rim_delta = positive_distance - rim_center;
        float rim_weight = exp(-(rim_delta * rim_delta) /
                               max(2.0f * rim_sigma * rim_sigma, 1e-6f));
        float shoulder_sigma = edge_softness * 2.40f;
        float shoulder_center = edge_softness * 2.80f;
        float shoulder_delta = positive_distance - shoulder_center;
        float shoulder_weight = exp(-(shoulder_delta * shoulder_delta) /
                                    max(2.0f * shoulder_sigma * shoulder_sigma, 1e-6f));
        float rim = (hull_displacement * 0.62f * rim_weight +
                     hull_displacement * 0.18f * shoulder_weight) *
                    (1.0f - inside_weight);
        displacement += -hull_displacement * inside_weight + rim;
    }

    return clamp(displacement, -0.18f, 0.18f);
}

float objectDisturbanceAtXZ(float2 xz)
{
    float disturbance = 0.0f;
    float kelvin_tan = tan(KELVIN_WAKE_HALF_ANGLE);

    [loop]
    for (uint object_index = 0; object_index < push.floating_object_interaction_count;
         ++object_index) {
        FloatingObjectInteractionGpu interaction = floating_object_interactions[object_index];
        float2 center = interaction.center_radius_strength.xy;
        float radius = max(interaction.center_radius_strength.z, 1e-4f);
        float strength = interaction.center_radius_strength.w;
        if (strength <= 1e-5f) {
            continue;
        }

        float2 wake_dir = interaction.wake_direction_length.xy;
        float wake_dir_length = length(wake_dir);
        if (wake_dir_length <= 1e-5f) {
            continue;
        }
        wake_dir /= wake_dir_length;
        float2 wake_right = float2(-wake_dir.y, wake_dir.x);
        float wake_length = max(interaction.wake_direction_length.z, radius * 2.0f);
        float planar_speed = max(interaction.wake_direction_length.w, 0.0f);
        float speed_factor = saturate(planar_speed * 3.5f);
        float local_strength = strength * (0.65f + speed_factor * 0.90f);

        disturbance += impactKernel(xz - (center + wake_dir * radius * 0.55f), radius * 1.05f) *
                       local_strength * 0.55f;
        disturbance += impactKernel(xz - (center - wake_dir * radius * 0.40f), radius * 0.92f) *
                       local_strength * 0.34f;

        if (planar_speed <= 0.004f) {
            continue;
        }

        int sample_count =
            clamp(int(ceil(wake_length / max(radius * 1.35f, 0.05f))), 3, 7);
        [loop]
        for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
            float t = float(sample_index + 1) / float(sample_count);
            float along = lerp(radius * 1.15f, wake_length, pow(t, 0.88f));
            float arm_offset = along * kelvin_tan;
            float2 wake_center = center - wake_dir * along;
            float sample_radius = radius * lerp(0.82f, 1.42f, t);
            float decay = exp(-t * 1.45f);
            float arm_strength = local_strength * decay * (0.22f + speed_factor * 0.24f);
            float2 left_arm_center = wake_center + wake_right * arm_offset;
            float2 right_arm_center = wake_center - wake_right * arm_offset;
            disturbance += impactKernel(xz - left_arm_center, sample_radius) * arm_strength;
            disturbance += impactKernel(xz - right_arm_center, sample_radius) * arm_strength;

            float centerline_strength =
                local_strength * decay * ((sample_index & 1) == 0 ? 0.12f : -0.09f);
            disturbance += impactKernel(xz - (center - wake_dir * along * 0.82f),
                                        sample_radius * 0.82f) *
                           centerline_strength;
        }
    }

    return disturbance;
}

float heightToWorld(float value)
{
    return value * (push.height_scale * 0.045f);
}

float surfaceHeightAtXZ(float2 xz, int2 dims)
{
    float2 uv = xz * 0.5f + 0.5f;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        return WATER_HEIGHT_SENTINEL;
    }

    float sampled = sampleCurrentHeight(uv, dims);
    if (!isValidWaterHeight(sampled)) {
        return WATER_HEIGHT_SENTINEL;
    }
    return heightToWorld(sampled);
}

float3 surfaceNormalAtXZ(float2 xz, int2 dims)
{
    float dx = 2.0f / max(float(dims.x), 1.0f);
    float dz = 2.0f / max(float(dims.y), 1.0f);

    float center = surfaceHeightAtXZ(xz, dims);
    if (!isValidWaterHeight(center)) {
        return float3(0.0f, 1.0f, 0.0f);
    }
    float hL = sanitizeWaterHeight(surfaceHeightAtXZ(xz - float2(dx, 0.0f), dims), center);
    float hR = sanitizeWaterHeight(surfaceHeightAtXZ(xz + float2(dx, 0.0f), dims), center);
    float hD = sanitizeWaterHeight(surfaceHeightAtXZ(xz - float2(0.0f, dz), dims), center);
    float hU = sanitizeWaterHeight(surfaceHeightAtXZ(xz + float2(0.0f, dz), dims), center);

    float dhdx = (hR - hL) / max(2.0f * dx, 1e-4f);
    float dhdz = (hU - hD) / max(2.0f * dz, 1e-4f);
    return normalize(float3(-dhdx, 1.0f, -dhdz));
}

float rippleEnergyAtXZ(float2 xz, int2 dims)
{
    float2 uv = xz * 0.5f + 0.5f;
    float du = 1.0f / max(float(dims.x), 1.0f);
    float dv = 1.0f / max(float(dims.y), 1.0f);

    float center = sampleCurrentHeight(uv, dims);
    if (!isValidWaterHeight(center)) {
        return 0.0f;
    }
    float left = sanitizeWaterHeight(sampleCurrentHeight(uv + float2(-du, 0.0f), dims), center);
    float right = sanitizeWaterHeight(sampleCurrentHeight(uv + float2(du, 0.0f), dims), center);
    float down = sanitizeWaterHeight(sampleCurrentHeight(uv + float2(0.0f, -dv), dims), center);
    float up = sanitizeWaterHeight(sampleCurrentHeight(uv + float2(0.0f, dv), dims), center);
    float down_left =
        sanitizeWaterHeight(sampleCurrentHeight(uv + float2(-du, -dv), dims), center);
    float down_right =
        sanitizeWaterHeight(sampleCurrentHeight(uv + float2(du, -dv), dims), center);
    float up_left = sanitizeWaterHeight(sampleCurrentHeight(uv + float2(-du, dv), dims), center);
    float up_right = sanitizeWaterHeight(sampleCurrentHeight(uv + float2(du, dv), dims), center);
    float laplacian = (((left + right + down + up) * 4.0f) +
                       (down_left + down_right + up_left + up_right) -
                       (20.0f * center)) / 6.0f;
    float2 gradient = float2(right - left, up - down);

    return saturate(length(gradient) * 8.5f + abs(laplacian) * 2.2f + abs(center) * 7.0f +
                    push.audio_level * 0.18f);
}

float3 skyColor(float3 dir)
{
    float t = saturate(dir.y * 0.5f + 0.5f);
    float3 horizon = float3(0.76f, 0.88f, 0.98f);
    float3 zenith = float3(0.08f, 0.19f, 0.33f);
    return lerp(horizon, zenith, t);
}

bool traceFloatingObject(float3 ray_origin, float3 ray_dir, FloatingObjectGpu object_data,
                         out float hit_t, out float3 hit_pos, out float3 hit_normal,
                         out float3 hit_color)
{
    float3 delta = ray_origin - object_data.center.xyz;
    float3 local_origin = float3(dot(delta, object_data.axis_x.xyz),
                                 dot(delta, object_data.axis_y.xyz),
                                 dot(delta, object_data.axis_z.xyz));
    float3 local_dir = float3(dot(ray_dir, object_data.axis_x.xyz),
                              dot(ray_dir, object_data.axis_y.xyz),
                              dot(ray_dir, object_data.axis_z.xyz));

    float t_min;
    float t_max;
    if (!intersectAABB(local_origin, local_dir, -object_data.half_extents.xyz,
                       object_data.half_extents.xyz, t_min, t_max)) {
        hit_t = 0.0f;
        hit_pos = float3(0.0f, 0.0f, 0.0f);
        hit_normal = float3(0.0f, 1.0f, 0.0f);
        hit_color = float3(0.0f, 0.0f, 0.0f);
        return false;
    }

    hit_t = max(t_min, 0.0f);
    float3 local_hit = local_origin + local_dir * hit_t;
    float3 face_distance = abs(abs(local_hit) - object_data.half_extents.xyz);

    float3 local_normal = float3(local_hit.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
    if (face_distance.y < face_distance.x && face_distance.y <= face_distance.z) {
        local_normal = float3(0.0f, local_hit.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
    } else if (face_distance.z < face_distance.x && face_distance.z < face_distance.y) {
        local_normal = float3(0.0f, 0.0f, local_hit.z >= 0.0f ? 1.0f : -1.0f);
    }

    hit_pos = ray_origin + ray_dir * hit_t;
    hit_normal = normalize(local_normal.x * object_data.axis_x.xyz +
                           local_normal.y * object_data.axis_y.xyz +
                           local_normal.z * object_data.axis_z.xyz);
    hit_color = object_data.color.xyz;
    return true;
}

bool traceFloatingObjects(float3 ray_origin, float3 ray_dir, out float hit_t, out float3 hit_pos,
                          out float3 hit_normal, out float3 hit_color)
{
    bool found_hit = false;
    hit_t = 0.0f;
    hit_pos = float3(0.0f, 0.0f, 0.0f);
    hit_normal = float3(0.0f, 1.0f, 0.0f);
    hit_color = float3(0.0f, 0.0f, 0.0f);

    [loop]
    for (uint object_index = 0; object_index < push.floating_object_count; ++object_index) {
        float candidate_t;
        float3 candidate_pos;
        float3 candidate_normal;
        float3 candidate_color;
        if (!traceFloatingObject(ray_origin, ray_dir, floating_objects[object_index], candidate_t,
                                 candidate_pos, candidate_normal, candidate_color)) {
            continue;
        }

        if (!found_hit || candidate_t < hit_t) {
            found_hit = true;
            hit_t = candidate_t;
            hit_pos = candidate_pos;
            hit_normal = candidate_normal;
            hit_color = candidate_color;
        }
    }

    return found_hit;
}

float3 shadeFloatingObject(float3 hit_pos, float3 hit_normal, float3 view_dir, float3 base_color)
{
    float3 light_dir = normalize(float3(-0.35f, 0.86f, 0.38f));
    float3 reflected_light = reflect(-light_dir, hit_normal);
    float diffuse = saturate(dot(hit_normal, light_dir));
    float specular = pow(saturate(dot(reflected_light, view_dir)), 42.0f);
    float sky_wrap = saturate(hit_normal.y * 0.5f + 0.5f);

    float3 ambient = lerp(float3(0.09f, 0.10f, 0.11f), float3(0.25f, 0.27f, 0.30f), sky_wrap);
    float3 color = base_color * (ambient + diffuse * 0.78f);
    color += specular * float3(1.0f, 0.95f, 0.90f) * (0.12f + push.audio_level * 0.16f);

    float wetness = saturate((0.04f - hit_pos.y) * 4.0f);
    color = lerp(color, color * float3(0.65f, 0.76f, 0.84f), wetness * 0.28f);
    return color;
}

bool intersectAABB(float3 ray_origin, float3 ray_dir, float3 box_min, float3 box_max, out float t_min,
                   out float t_max)
{
    float3 inv_dir = 1.0f / ray_dir;
    float3 t0 = (box_min - ray_origin) * inv_dir;
    float3 t1 = (box_max - ray_origin) * inv_dir;

    float3 t_small = min(t0, t1);
    float3 t_large = max(t0, t1);

    t_min = max(max(t_small.x, t_small.y), t_small.z);
    t_max = min(min(t_large.x, t_large.y), t_large.z);
    return t_max >= max(t_min, 0.0f);
}

bool objectBottomHeightAtXZ(float2 xz, FloatingObjectGpu object_data, out float bottom_height)
{
    HorizontalFootprintGpu footprint = computeHorizontalFootprint(object_data);
    float2 delta = xz - object_data.center.xz;
    float2 local = float2(dot(delta, footprint.axis_u), dot(delta, footprint.axis_v));
    const float footprint_pad = 0.02f;
    if (abs(local.x) > footprint.half_extents.x + footprint_pad ||
        abs(local.y) > footprint.half_extents.y + footprint_pad) {
        bottom_height = 0.0f;
        return false;
    }

    float3 ray_origin = float3(xz.x, -1.6f, xz.y);
    float3 ray_dir = float3(0.0f, 1.0f, 0.0f);
    float3 ray_delta = ray_origin - object_data.center.xyz;
    float3 local_origin = float3(dot(ray_delta, object_data.axis_x.xyz),
                                 dot(ray_delta, object_data.axis_y.xyz),
                                 dot(ray_delta, object_data.axis_z.xyz));
    float3 local_dir = float3(dot(ray_dir, object_data.axis_x.xyz),
                              dot(ray_dir, object_data.axis_y.xyz),
                              dot(ray_dir, object_data.axis_z.xyz));

    float t_min;
    float t_max;
    if (!intersectAABB(local_origin, local_dir, -object_data.half_extents.xyz,
                       object_data.half_extents.xyz, t_min, t_max) || t_max < 0.0f) {
        bottom_height = 0.0f;
        return false;
    }

    bottom_height = ray_origin.y + max(t_min, 0.0f);
    return true;
}

float objectWaterCeilingAtXZ(float2 xz, out float mask)
{
    float ceiling = 1e6f;
    mask = 0.0f;

    [loop]
    for (uint object_index = 0; object_index < push.floating_object_count; ++object_index) {
        float bottom_height;
        if (!objectBottomHeightAtXZ(xz, floating_objects[object_index], bottom_height)) {
            continue;
        }

        ceiling = min(ceiling, bottom_height - HULL_WATER_CLEARANCE);
        mask = 1.0f;
    }

    return ceiling;
}

bool traceWater(float3 ray_origin, float3 ray_dir, int2 dims, out float hit_t, out float3 hit_pos,
                out float3 hit_normal, out float ripple_energy)
{
    float t_min;
    float t_max;
    if (!intersectAABB(ray_origin, ray_dir, float3(-1.05f, -0.7f, -1.05f),
                       float3(1.05f, 0.9f, 1.05f), t_min, t_max)) {
        hit_t = 0.0f;
        hit_pos = float3(0.0f, 0.0f, 0.0f);
        hit_normal = float3(0.0f, 1.0f, 0.0f);
        ripple_energy = 0.0f;
        return false;
    }

    t_min = max(t_min, 0.0f);

    float prev_t = t_min;
    float3 prev_pos = ray_origin + ray_dir * prev_t;
    float prev_height = surfaceHeightAtXZ(prev_pos.xz, dims);
    bool prev_valid = isValidWaterHeight(prev_height);
    float prev_delta = prev_pos.y - prev_height;

    [loop]
    for (int step = 1; step <= 84; ++step) {
        float t = lerp(t_min, t_max, float(step) / 84.0f);
        float3 pos = ray_origin + ray_dir * t;
        float height = surfaceHeightAtXZ(pos.xz, dims);
        bool valid = isValidWaterHeight(height);
        float delta = pos.y - height;

        if (prev_valid && valid && (prev_delta == 0.0f || delta == 0.0f || prev_delta * delta <= 0.0f)) {
            float lo = prev_t;
            float hi = t;
            float lo_delta = prev_delta;

            [unroll]
            for (int refine = 0; refine < 6; ++refine) {
                float mid = 0.5f * (lo + hi);
                float3 mid_pos = ray_origin + ray_dir * mid;
                float mid_height = surfaceHeightAtXZ(mid_pos.xz, dims);
                if (!isValidWaterHeight(mid_height)) {
                    hi = mid;
                    continue;
                }

                float mid_delta = mid_pos.y - mid_height;
                if (lo_delta == 0.0f || lo_delta * mid_delta <= 0.0f) {
                    hi = mid;
                } else {
                    lo = mid;
                    lo_delta = mid_delta;
                }
            }

            float final_hit_t = 0.5f * (lo + hi);
            hit_t = final_hit_t;
            hit_pos = ray_origin + ray_dir * final_hit_t;
            float final_height = surfaceHeightAtXZ(hit_pos.xz, dims);
            if (!isValidWaterHeight(final_height)) {
                break;
            }
            hit_pos.y = final_height;
            hit_normal = surfaceNormalAtXZ(hit_pos.xz, dims);
            ripple_energy = rippleEnergyAtXZ(hit_pos.xz, dims);
            return true;
        }

        prev_t = t;
        prev_delta = delta;
        prev_valid = valid;
    }

    hit_t = 0.0f;
    hit_pos = float3(0.0f, 0.0f, 0.0f);
    hit_normal = float3(0.0f, 1.0f, 0.0f);
    ripple_energy = 0.0f;
    return false;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint2 pixel = dispatch_id.xy;
    uint2 dims;
    output_image.GetDimensions(dims.x, dims.y);
    if (any(pixel >= dims)) return;

    int2 coord = int2(pixel);
    int2 dims_i = int2(dims);

    if (!isDomainActive(coord, dims_i)) {
        previous_height[coord] = WATER_HEIGHT_SENTINEL;
        output_image[coord] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    float center = sanitizeWaterHeight(current_height[coord], 0.0f);
    float left = readCurrentHeightForSolver(coord + int2(-1, 0), dims_i, center);
    float right = readCurrentHeightForSolver(coord + int2(1, 0), dims_i, center);
    float down = readCurrentHeightForSolver(coord + int2(0, -1), dims_i, center);
    float up = readCurrentHeightForSolver(coord + int2(0, 1), dims_i, center);
    float down_left = readCurrentHeightForSolver(coord + int2(-1, -1), dims_i, center);
    float down_right = readCurrentHeightForSolver(coord + int2(1, -1), dims_i, center);
    float up_left = readCurrentHeightForSolver(coord + int2(-1, 1), dims_i, center);
    float up_right = readCurrentHeightForSolver(coord + int2(1, 1), dims_i, center);
    float previous = sanitizeWaterHeight(previous_height[coord], center);

    float laplacian = (((left + right + down + up) * 4.0f) +
                       (down_left + down_right + up_left + up_right) -
                       (20.0f * center)) / 6.0f;
    float neighbor_average = (((left + right + down + up) * 2.0f) +
                              (down_left + down_right + up_left + up_right)) / 12.0f;
    float world_height_scale = max(heightToWorld(1.0f), 1e-4f);
    float world_cell_x = 2.0f / max(float(dims.x - 1), 1.0f);
    float world_cell_z = 2.0f / max(float(dims.y - 1), 1.0f);
    float2 world_gradient = float2(
        (right - left) * world_height_scale / max(2.0f * world_cell_x, 1e-4f),
        (up - down) * world_height_scale / max(2.0f * world_cell_z, 1e-4f));
    float world_height = abs(center) * world_height_scale;
    float raw_velocity = center - previous;
    float world_velocity = abs(raw_velocity) * world_height_scale;
    float world_curvature = abs(laplacian) * world_height_scale;
    float solver_stress = saturate(max(world_height / MAX_STABLE_WORLD_HEIGHT,
                                       max(world_velocity / MAX_STABLE_WORLD_VELOCITY,
                                           length(world_gradient) / MAX_STABLE_WORLD_SLOPE)));
    solver_stress = max(solver_stress, saturate(world_curvature * 2.6f));
    float adaptive_damping = saturate(push.damping + solver_stress * 0.08f);
    float velocity = raw_velocity * (1.0f - adaptive_damping);
    float collapse_smoothing = saturate(push.damping * 2.0f + solver_stress * 0.55f) * 0.04f;

    float2 uv = (float2(pixel) + 0.5f) / float2(dims);
    float2 world_xz = uv * 2.0f - 1.0f;
    float hull_target = objectHullDisplacementAtXZ(world_xz);
    float hull_mask = saturate(abs(hull_target) * 16.0f);
    float2 emitter = float2(push.emitter_u, push.emitter_v);
    float disturbance = impactKernel(uv - emitter, push.ripple_radius) * push.impulse_strength;
    float max_sim_height = clamp(MAX_STABLE_WORLD_HEIGHT / world_height_scale, 0.08f, 0.42f);
    float max_sim_velocity = max(max_sim_height * (0.42f - solver_stress * 0.14f),
                                 max_sim_height * 0.20f);
    float max_disturbance = max(max_sim_height * (0.22f - solver_stress * 0.08f),
                                max_sim_height * 0.08f);

    float roughness = push.audio_level * push.audio_level * (1.0f - solver_stress * 0.70f);
    float splash_angle = push.time_seconds * (6.0f + push.audio_level * 8.0f);
    float2 splash_dir0 = float2(cos(splash_angle), sin(splash_angle));
    float2 splash_dir1 = float2(-splash_dir0.y, splash_dir0.x);
    disturbance += roughness *
                   impactKernel(uv - (emitter + splash_dir0 * push.ripple_radius * 1.85f),
                                push.ripple_radius * 0.56f) *
                   push.impulse_strength * 0.34f;
    disturbance += roughness *
                   impactKernel(uv - (emitter - splash_dir1 * push.ripple_radius * 1.55f),
                                push.ripple_radius * 0.46f) *
                   push.impulse_strength * 0.22f;
    disturbance += objectDisturbanceAtXZ(world_xz);
    disturbance *= (1.0f - solver_stress * 0.45f);
    disturbance = clamp(disturbance, -max_disturbance, max_disturbance);

    float acceleration = (push.propagation * laplacian) - (center * push.restoring_force);
    acceleration *= lerp(1.0f, 0.65f, solver_stress);
    velocity += disturbance + acceleration;
    velocity += (hull_target - center) * (0.12f + hull_mask * 0.40f);
    velocity = clamp(velocity, -max_sim_velocity, max_sim_velocity);
    float next_height = center + velocity;
    next_height = lerp(next_height, hull_target, hull_mask * (0.22f + solver_stress * 0.10f));
    next_height = lerp(next_height, neighbor_average, collapse_smoothing);
    next_height = clamp(next_height, -max_sim_height, max_sim_height);
    previous_height[coord] = next_height;

    float2 screen_uv = (float2(pixel) + 0.5f) / float2(dims);
    float2 ndc = float2(screen_uv.x * 2.0f - 1.0f, 1.0f - screen_uv.y * 2.0f);
    ndc.x *= float(dims.x) / max(float(dims.y), 1.0f);

    float3 camera_pos = float3(0.0f, 0.92f, 1.58f);
    float3 camera_target = float3(0.0f, 0.03f, 0.0f);
    float3 world_up = float3(0.0f, 1.0f, 0.0f);
    float3 forward = normalize(camera_target - camera_pos);
    float3 right_axis = normalize(cross(forward, world_up));
    float3 up_axis = normalize(cross(right_axis, forward));
    float fov_scale = tan((32.0f * PI / 180.0f) * 0.5f);
    float3 ray_dir = normalize(forward + ndc.x * right_axis * fov_scale + ndc.y * up_axis * fov_scale);

    float water_hit_t;
    float3 water_hit_pos;
    float3 water_hit_normal;
    float water_ripple_energy;
    float object_hit_t;
    float3 object_hit_pos;
    float3 object_hit_normal;
    float3 object_hit_color;
    float3 color = skyColor(ray_dir);

    bool hit_water = traceWater(camera_pos, ray_dir, dims_i, water_hit_t, water_hit_pos,
                                water_hit_normal, water_ripple_energy);
    bool hit_object = traceFloatingObjects(camera_pos, ray_dir, object_hit_t, object_hit_pos,
                                           object_hit_normal, object_hit_color);

    if (hit_object && (!hit_water || object_hit_t < water_hit_t)) {
        float3 view_dir = normalize(camera_pos - object_hit_pos);
        color = shadeFloatingObject(object_hit_pos, object_hit_normal, view_dir, object_hit_color);
    } else if (hit_water) {
        float3 view_dir = normalize(camera_pos - water_hit_pos);
        float3 light_dir = normalize(float3(-0.35f, 0.86f, 0.38f));
        float3 reflected_dir = reflect(-view_dir, water_hit_normal);

        float fresnel = pow(1.0f - saturate(dot(water_hit_normal, view_dir)), 5.0f);
        float diffuse = saturate(dot(water_hit_normal, light_dir));
        float specular = pow(saturate(dot(reflect(-light_dir, water_hit_normal), view_dir)), 78.0f);

        float height_band = saturate(water_hit_pos.y * 2.8f + 0.5f);
        float3 deep_water = float3(0.02f, 0.10f, 0.18f);
        float3 shallow_water = float3(0.08f, 0.38f, 0.55f);
        float3 transmitted = lerp(deep_water, shallow_water, height_band);

        float3 reflection = skyColor(reflected_dir);
        color = lerp(transmitted, reflection, 0.22f + 0.78f * fresnel);
        color += diffuse * float3(0.02f, 0.03f, 0.04f);
        color += specular * float3(1.0f, 0.97f, 0.92f) * (0.45f + push.audio_level * 0.85f);
        color = lerp(color, float3(0.92f, 0.97f, 1.0f),
                     water_ripple_energy * (0.08f + 0.14f * push.audio_level));
    }

    output_image[coord] = float4(pow(max(color, 0.0f), 1.0f / 2.2f), 1.0f);
}
