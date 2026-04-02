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
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

[[vk::binding(0, 0)]] RWTexture2D<float4> output_image;
[[vk::binding(1, 0)]] RWTexture2D<float> current_height;
[[vk::binding(2, 0)]] RWTexture2D<float> previous_height;

static const float PI = 3.14159265358979323846f;

int2 clampCoord(int2 coord, int2 dims)
{
    return clamp(coord, int2(0, 0), dims - 1);
}

float readCurrentHeight(int2 coord, int2 dims)
{
    return current_height[clampCoord(coord, dims)];
}

float sampleCurrentHeight(float2 uv, int2 dims)
{
    uv = clamp(uv, float2(0.0f, 0.0f), float2(1.0f, 1.0f));

    float2 texel = uv * float2(dims - 1);
    int2 p0 = int2(floor(texel));
    int2 p1 = min(p0 + int2(1, 1), dims - 1);
    float2 f = texel - float2(p0);

    float h00 = readCurrentHeight(p0, dims);
    float h10 = readCurrentHeight(int2(p1.x, p0.y), dims);
    float h01 = readCurrentHeight(int2(p0.x, p1.y), dims);
    float h11 = readCurrentHeight(p1, dims);

    return lerp(lerp(h00, h10, f.x), lerp(h01, h11, f.x), f.y);
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

float heightToWorld(float value)
{
    return value * (push.height_scale * 0.045f);
}

float surfaceHeightAtXZ(float2 xz, int2 dims)
{
    float2 uv = xz * 0.5f + 0.5f;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        return -1000.0f;
    }

    return heightToWorld(sampleCurrentHeight(uv, dims));
}

float3 surfaceNormalAtXZ(float2 xz, int2 dims)
{
    float dx = 2.0f / max(float(dims.x), 1.0f);
    float dz = 2.0f / max(float(dims.y), 1.0f);

    float hL = surfaceHeightAtXZ(xz - float2(dx, 0.0f), dims);
    float hR = surfaceHeightAtXZ(xz + float2(dx, 0.0f), dims);
    float hD = surfaceHeightAtXZ(xz - float2(0.0f, dz), dims);
    float hU = surfaceHeightAtXZ(xz + float2(0.0f, dz), dims);

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
    float left = sampleCurrentHeight(uv + float2(-du, 0.0f), dims);
    float right = sampleCurrentHeight(uv + float2(du, 0.0f), dims);
    float down = sampleCurrentHeight(uv + float2(0.0f, -dv), dims);
    float up = sampleCurrentHeight(uv + float2(0.0f, dv), dims);
    float down_left = sampleCurrentHeight(uv + float2(-du, -dv), dims);
    float down_right = sampleCurrentHeight(uv + float2(du, -dv), dims);
    float up_left = sampleCurrentHeight(uv + float2(-du, dv), dims);
    float up_right = sampleCurrentHeight(uv + float2(du, dv), dims);
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

bool traceWater(float3 ray_origin, float3 ray_dir, int2 dims, out float3 hit_pos, out float3 hit_normal,
                out float ripple_energy)
{
    float t_min;
    float t_max;
    if (!intersectAABB(ray_origin, ray_dir, float3(-1.05f, -0.7f, -1.05f),
                       float3(1.05f, 0.9f, 1.05f), t_min, t_max)) {
        hit_pos = float3(0.0f, 0.0f, 0.0f);
        hit_normal = float3(0.0f, 1.0f, 0.0f);
        ripple_energy = 0.0f;
        return false;
    }

    t_min = max(t_min, 0.0f);

    float prev_t = t_min;
    float3 prev_pos = ray_origin + ray_dir * prev_t;
    float prev_delta = prev_pos.y - surfaceHeightAtXZ(prev_pos.xz, dims);

    [loop]
    for (int step = 1; step <= 84; ++step) {
        float t = lerp(t_min, t_max, float(step) / 84.0f);
        float3 pos = ray_origin + ray_dir * t;
        float delta = pos.y - surfaceHeightAtXZ(pos.xz, dims);

        if (delta <= 0.0f) {
            float lo = prev_t;
            float hi = t;

            [unroll]
            for (int refine = 0; refine < 6; ++refine) {
                float mid = 0.5f * (lo + hi);
                float3 mid_pos = ray_origin + ray_dir * mid;
                float mid_delta = mid_pos.y - surfaceHeightAtXZ(mid_pos.xz, dims);
                if (mid_delta > 0.0f) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }

            float hit_t = 0.5f * (lo + hi);
            hit_pos = ray_origin + ray_dir * hit_t;
            hit_pos.y = surfaceHeightAtXZ(hit_pos.xz, dims);
            hit_normal = surfaceNormalAtXZ(hit_pos.xz, dims);
            ripple_energy = rippleEnergyAtXZ(hit_pos.xz, dims);
            return true;
        }

        prev_t = t;
        prev_delta = delta;
    }

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

    float center = readCurrentHeight(coord, dims_i);
    float left = readCurrentHeight(coord + int2(-1, 0), dims_i);
    float right = readCurrentHeight(coord + int2(1, 0), dims_i);
    float down = readCurrentHeight(coord + int2(0, -1), dims_i);
    float up = readCurrentHeight(coord + int2(0, 1), dims_i);
    float down_left = readCurrentHeight(coord + int2(-1, -1), dims_i);
    float down_right = readCurrentHeight(coord + int2(1, -1), dims_i);
    float up_left = readCurrentHeight(coord + int2(-1, 1), dims_i);
    float up_right = readCurrentHeight(coord + int2(1, 1), dims_i);
    float previous = previous_height[coord];

    float laplacian = (((left + right + down + up) * 4.0f) +
                       (down_left + down_right + up_left + up_right) -
                       (20.0f * center)) / 6.0f;
    float neighbor_average = (((left + right + down + up) * 2.0f) +
                              (down_left + down_right + up_left + up_right)) / 12.0f;
    float velocity = (center - previous) * (1.0f - push.damping);
    float collapse_smoothing = saturate(push.damping * 2.5f) * 0.05f;

    float2 uv = (float2(pixel) + 0.5f) / float2(dims);
    float2 emitter = float2(push.emitter_u, push.emitter_v);
    float disturbance = impactKernel(uv - emitter, push.ripple_radius) * push.impulse_strength;

    float roughness = push.audio_level * push.audio_level;
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

    float acceleration = (push.propagation * laplacian) - (center * push.restoring_force);
    velocity += disturbance + acceleration;
    float next_height = center + velocity;
    next_height = lerp(next_height, neighbor_average, collapse_smoothing);
    next_height = clamp(next_height, -0.75f, 0.75f);
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

    float3 hit_pos;
    float3 hit_normal;
    float ripple_energy;
    float3 color = skyColor(ray_dir);

    if (traceWater(camera_pos, ray_dir, dims_i, hit_pos, hit_normal, ripple_energy)) {
        float3 view_dir = normalize(camera_pos - hit_pos);
        float3 light_dir = normalize(float3(-0.35f, 0.86f, 0.38f));
        float3 reflected_dir = reflect(-view_dir, hit_normal);

        float fresnel = pow(1.0f - saturate(dot(hit_normal, view_dir)), 5.0f);
        float diffuse = saturate(dot(hit_normal, light_dir));
        float specular = pow(saturate(dot(reflect(-light_dir, hit_normal), view_dir)), 78.0f);

        float height_band = saturate(hit_pos.y * 2.8f + 0.5f);
        float3 deep_water = float3(0.02f, 0.10f, 0.18f);
        float3 shallow_water = float3(0.08f, 0.38f, 0.55f);
        float3 transmitted = lerp(deep_water, shallow_water, height_band);

        float3 reflection = skyColor(reflected_dir);
        color = lerp(transmitted, reflection, 0.22f + 0.78f * fresnel);
        color += diffuse * float3(0.02f, 0.03f, 0.04f);
        color += specular * float3(1.0f, 0.97f, 0.92f) * (0.45f + push.audio_level * 0.85f);
        color = lerp(color, float3(0.92f, 0.97f, 1.0f), ripple_energy * (0.08f + 0.14f * push.audio_level));
    }

    output_image[coord] = float4(pow(max(color, 0.0f), 1.0f / 2.2f), 1.0f);
}
