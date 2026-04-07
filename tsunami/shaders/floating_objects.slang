struct PushConstants
{
    float time_seconds;
    float delta_time;
    float height_to_world_scale;
    uint object_count;
    uint reset_requested;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

struct FloatingObjectSettingsGpu
{
    float4 anchor_base_height_yaw;
    float4 size_mass;
    float4 color_buoyancy;
    float4 buoyancy_linear_angular;
    float4 motion_planar;
    float4 anchor_waterline_yaw;
};

struct FloatingObjectStateGpu
{
    float4 position_hull;
    float4 rotation_pad;
    float4 linear_velocity_pad;
    float4 angular_velocity_pad;
};

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

[[vk::binding(0, 0)]] RWTexture2D<float> current_height;
[[vk::binding(1, 0)]] StructuredBuffer<FloatingObjectSettingsGpu> object_settings;
[[vk::binding(2, 0)]] RWStructuredBuffer<FloatingObjectStateGpu> object_states;
[[vk::binding(3, 0)]] RWStructuredBuffer<FloatingObjectGpu> floating_objects;
[[vk::binding(4, 0)]] RWStructuredBuffer<FloatingObjectInteractionGpu> floating_object_interactions;

static const uint MAX_FLOATING_OBJECTS = 8u;
static const float GRAVITY = 4.5f;
static const float SURFACE_BOUNDS = 0.94f;

groupshared FloatingObjectGpu g_previous_objects[MAX_FLOATING_OBJECTS];
groupshared FloatingObjectStateGpu g_previous_states[MAX_FLOATING_OBJECTS];
groupshared FloatingObjectSettingsGpu g_previous_settings[MAX_FLOATING_OBJECTS];

static const float4 BUOYANCY_POINT_PATTERN[13] = {
    float4(-0.78f, -1.0f, -0.78f, 0.85f),
    float4(0.0f, -1.0f, -0.78f, 1.00f),
    float4(0.78f, -1.0f, -0.78f, 0.85f),
    float4(-0.78f, -1.0f, 0.0f, 1.00f),
    float4(0.0f, -1.0f, 0.0f, 1.30f),
    float4(0.78f, -1.0f, 0.0f, 1.00f),
    float4(-0.78f, -1.0f, 0.78f, 0.85f),
    float4(0.0f, -1.0f, 0.78f, 1.00f),
    float4(0.78f, -1.0f, 0.78f, 0.85f),
    float4(-1.0f, -0.35f, 0.0f, 0.55f),
    float4(1.0f, -0.35f, 0.0f, 0.55f),
    float4(0.0f, -0.35f, -1.0f, 0.55f),
    float4(0.0f, -0.35f, 1.0f, 0.55f),
};
static const float BUOYANCY_FORCE_SCALE = 5.0f / 10.9f;
static const float TOTAL_BUOYANCY_WEIGHT = 5.0f;

float clamp01(float value)
{
    return clamp(value, 0.0f, 1.0f);
}

float2 safeNormalize2D(float2 value, float2 fallback)
{
    float len = length(value);
    return len > 1e-5f ? value / len : fallback;
}

float wrapAngle(float angle_radians)
{
    static const float PI = 3.14159265358979323846f;
    static const float TWO_PI = PI * 2.0f;
    while (angle_radians > PI) {
        angle_radians -= TWO_PI;
    }
    while (angle_radians < -PI) {
        angle_radians += TWO_PI;
    }
    return angle_radians;
}

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

float4 quatFromEuler(float3 euler)
{
    float3 c = cos(euler * 0.5f);
    float3 s = sin(euler * 0.5f);
    return float4(
        s.x * c.y * c.z - c.x * s.y * s.z,
        c.x * s.y * c.z + s.x * c.y * s.z,
        c.x * c.y * s.z - s.x * s.y * c.z,
        c.x * c.y * c.z + s.x * s.y * s.z);
}

float3 rotateByQuat(float4 quat, float3 value)
{
    return value + 2.0f * cross(quat.xyz, cross(quat.xyz, value) + quat.w * value);
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

float objectHullDisplacementAtXZ(float2 xz, uint ignored_object_index)
{
    float displacement = 0.0f;

    [unroll]
    for (uint object_index = 0; object_index < MAX_FLOATING_OBJECTS; ++object_index) {
        if (object_index >= push.object_count) {
            break;
        }
        if (object_index == ignored_object_index) {
            continue;
        }

        FloatingObjectGpu object_data = g_previous_objects[object_index];
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
        float inside_weight = 1.0f - smoothstep(-edge_softness, edge_softness, signed_distance);
        float positive_distance = max(signed_distance, 0.0f);
        float rim_sigma = edge_softness * 1.35f;
        float rim_center = edge_softness * 1.15f;
        float rim_delta = positive_distance - rim_center;
        float rim_weight =
            exp(-(rim_delta * rim_delta) / max(2.0f * rim_sigma * rim_sigma, 1e-6f));
        float shoulder_sigma = edge_softness * 2.40f;
        float shoulder_center = edge_softness * 2.80f;
        float shoulder_delta = positive_distance - shoulder_center;
        float shoulder_weight =
            exp(-(shoulder_delta * shoulder_delta) / max(2.0f * shoulder_sigma * shoulder_sigma, 1e-6f));
        float rim = (hull_displacement * 0.62f * rim_weight +
                     hull_displacement * 0.18f * shoulder_weight) *
                    (1.0f - inside_weight);
        displacement += -hull_displacement * inside_weight + rim;
    }

    return clamp(displacement, -0.18f, 0.18f);
}

float sampleWaterHeightAtXZ(float2 xz, int2 dims, uint ignored_object_index)
{
    float2 uv = xz * 0.5f + 0.5f;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        return -1000.0f;
    }

    float raw_height = sampleCurrentHeight(uv, dims) * push.height_to_world_scale;
    float hull_displacement =
        objectHullDisplacementAtXZ(xz, ignored_object_index) * push.height_to_world_scale;
    return raw_height - hull_displacement;
}

[shader("compute")]
[numthreads(8, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uint3 group_thread_id : SV_GroupThreadID)
{
    uint object_index = group_thread_id.x;
    if (object_index < MAX_FLOATING_OBJECTS) {
        g_previous_objects[object_index] = floating_objects[object_index];
        g_previous_states[object_index] = object_states[object_index];
        g_previous_settings[object_index] = object_settings[object_index];
    }
    GroupMemoryBarrierWithGroupSync();

    if (object_index >= push.object_count) {
        return;
    }

    int2 dims;
    current_height.GetDimensions(dims.x, dims.y);

    FloatingObjectSettingsGpu settings = g_previous_settings[object_index];
    FloatingObjectStateGpu state = g_previous_states[object_index];

    float2 anchor_xz = settings.anchor_base_height_yaw.xy;
    float base_height = settings.anchor_base_height_yaw.z;
    float base_yaw = settings.anchor_base_height_yaw.w;
    float3 size = settings.size_mass.xyz;
    float mass = max(settings.size_mass.w, 0.001f);
    float3 color = settings.color_buoyancy.xyz;
    float buoyancy_strength = settings.color_buoyancy.w;
    float buoyancy_damping = settings.buoyancy_linear_angular.x;
    float linear_damping = settings.buoyancy_linear_angular.y;
    float angular_strength = settings.buoyancy_linear_angular.z;
    float angular_damping = settings.buoyancy_linear_angular.w;
    float self_righting = settings.motion_planar.x;
    float max_tilt_radians = settings.motion_planar.y;
    float planar_drift_strength = settings.motion_planar.z;
    float planar_damping = settings.motion_planar.w;
    float anchor_pull_strength = settings.anchor_waterline_yaw.x;
    float drift_radius = settings.anchor_waterline_yaw.y;
    float waterline_offset = settings.anchor_waterline_yaw.z;
    float yaw_follow_strength = settings.anchor_waterline_yaw.w;

    if (push.reset_requested != 0u) {
        state.position_hull = float4(anchor_xz.x, base_height, anchor_xz.y, 0.0f);
        state.rotation_pad = float4(0.0f, base_yaw, 0.0f, 0.0f);
        state.linear_velocity_pad = float4(0.0f, 0.0f, 0.0f, 0.0f);
        state.angular_velocity_pad = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float3 half_extents = size * 0.5f;
    float3 previous_position = state.position_hull.xyz;
    float3 previous_rotation = state.rotation_pad.xyz;
    float4 orientation = quatFromEuler(previous_rotation);
    float3 axis_x_prev = normalize(rotateByQuat(orientation, float3(1.0f, 0.0f, 0.0f)));
    float3 axis_y_prev = normalize(rotateByQuat(orientation, float3(0.0f, 1.0f, 0.0f)));
    float3 axis_z_prev = normalize(rotateByQuat(orientation, float3(0.0f, 0.0f, 1.0f)));
    float2 previous_center_xz = previous_position.xz;

    float3 linear_velocity = state.linear_velocity_pad.xyz;
    float2 angular_velocity = state.angular_velocity_pad.xy;
    float3 total_force = float3(0.0f, -GRAVITY * mass, 0.0f);
    float3 total_torque = float3(0.0f, 0.0f, 0.0f);

    [unroll]
    for (int point_index = 0; point_index < 13; ++point_index) {
        float4 pattern = BUOYANCY_POINT_PATTERN[point_index];
        float sample_weight = pattern.w * BUOYANCY_FORCE_SCALE;
        float3 local_point = float3(pattern.x * half_extents.x, pattern.y * half_extents.y,
                                    pattern.z * half_extents.z);
        float3 rotated_point = axis_x_prev * local_point.x + axis_y_prev * local_point.y +
                               axis_z_prev * local_point.z;
        float3 world_point = previous_position + rotated_point;
        float water_height = sampleWaterHeightAtXZ(world_point.xz, dims, object_index);
        float submersion = water_height - world_point.y;
        if (submersion <= 0.0f) {
            continue;
        }

        float3 angular_velocity_world = float3(angular_velocity.x, 0.0f, angular_velocity.y);
        float3 point_velocity = linear_velocity + cross(angular_velocity_world, rotated_point);
        float upward_force =
            max(submersion * buoyancy_strength * sample_weight -
                (point_velocity.y * buoyancy_damping * sample_weight), 0.0f);
        float3 buoyancy_force = float3(0.0f, upward_force, 0.0f);
        total_force += buoyancy_force;
        total_torque += cross(rotated_point, buoyancy_force);
    }

    float slope_sample_distance = max(size.x, size.z) * 0.55f;
    float2 center_xz = previous_position.xz;
    float water_left =
        sampleWaterHeightAtXZ(center_xz + float2(-slope_sample_distance, 0.0f), dims, object_index);
    float water_right =
        sampleWaterHeightAtXZ(center_xz + float2(slope_sample_distance, 0.0f), dims, object_index);
    float water_down =
        sampleWaterHeightAtXZ(center_xz + float2(0.0f, -slope_sample_distance), dims, object_index);
    float water_up =
        sampleWaterHeightAtXZ(center_xz + float2(0.0f, slope_sample_distance), dims, object_index);
    float2 water_gradient = float2(
        (water_right - water_left) / max(slope_sample_distance * 2.0f, 1.0e-4f),
        (water_up - water_down) / max(slope_sample_distance * 2.0f, 1.0e-4f));
    float2 anchor_offset = center_xz - anchor_xz;
    float2 planar_force = -water_gradient * planar_drift_strength * mass;

    float anchor_distance = length(anchor_offset);
    if (anchor_distance > drift_radius) {
        float2 overflow =
            (anchor_offset / max(anchor_distance, 1.0e-4f)) * (anchor_distance - drift_radius);
        planar_force += -overflow * anchor_pull_strength * mass;
    }

    float collision_radius = max(size.x, size.z) * 0.42f;
    [unroll]
    for (uint other_index = 0; other_index < MAX_FLOATING_OBJECTS; ++other_index) {
        if (other_index >= push.object_count || other_index == object_index) {
            continue;
        }

        FloatingObjectStateGpu other_state = g_previous_states[other_index];
        FloatingObjectSettingsGpu other_settings = g_previous_settings[other_index];
        float3 other_position = other_state.position_hull.xyz;
        float2 separation = previous_position.xz - other_position.xz;
        float distance = length(separation);
        float other_radius = max(other_settings.size_mass.x, other_settings.size_mass.z) * 0.42f;
        float min_separation = collision_radius + other_radius + 0.035f;
        float vertical_overlap_limit = (size.y + other_settings.size_mass.y) * 0.55f;
        if (abs(previous_position.y - other_position.y) > vertical_overlap_limit ||
            distance >= min_separation) {
            continue;
        }

        float2 separation_dir = distance > 1.0e-4f ?
            (separation / distance) :
            safeNormalize2D(anchor_xz - other_position.xz,
                            float2(cos(push.time_seconds + object_index),
                                   sin(push.time_seconds + object_index)));
        float overlap = min_separation - distance;
        float2 other_planar_velocity = other_state.linear_velocity_pad.xz;
        float closing_speed = dot(linear_velocity.xz - other_planar_velocity, separation_dir);
        planar_force += separation_dir * overlap * mass * 42.0f;
        if (closing_speed < 0.0f) {
            planar_force += separation_dir * (-closing_speed) * mass * 7.0f;
        }
    }

    float horizontal_bound = SURFACE_BOUNDS - max(half_extents.x, half_extents.z) - 0.02f;
    if (abs(previous_position.x) > horizontal_bound) {
        planar_force.x += (-previous_position.x +
                           clamp(previous_position.x, -horizontal_bound, horizontal_bound)) *
                          8.0f * mass;
    }
    if (abs(previous_position.z) > horizontal_bound) {
        planar_force.y += (-previous_position.z +
                           clamp(previous_position.z, -horizontal_bound, horizontal_bound)) *
                          8.0f * mass;
    }

    float inverse_mass = 1.0f / mass;
    linear_velocity.y += (total_force.y * inverse_mass) * push.delta_time;
    linear_velocity.x += (planar_force.x * inverse_mass) * push.delta_time;
    linear_velocity.z += (planar_force.y * inverse_mass) * push.delta_time;
    linear_velocity.x *= exp(-planar_damping * push.delta_time);
    linear_velocity.y *= exp(-linear_damping * push.delta_time);
    linear_velocity.z *= exp(-planar_damping * push.delta_time);
    float max_upward_speed = max(0.20f, size.y * 2.0f + 0.12f);
    float max_downward_speed = max(0.28f, size.y * 2.8f + 0.18f);
    linear_velocity.y = clamp(linear_velocity.y, -max_downward_speed, max_upward_speed);

    float3 position = previous_position;
    position += linear_velocity * push.delta_time;
    position.x = clamp(position.x, -horizontal_bound, horizontal_bound);
    position.z = clamp(position.z, -horizontal_bound, horizontal_bound);

    float2 position_correction = float2(0.0f, 0.0f);
    [unroll]
    for (uint other_index = 0; other_index < MAX_FLOATING_OBJECTS; ++other_index) {
        if (other_index >= push.object_count || other_index == object_index) {
            continue;
        }

        FloatingObjectStateGpu other_state = g_previous_states[other_index];
        FloatingObjectSettingsGpu other_settings = g_previous_settings[other_index];
        float2 separation = position.xz - other_state.position_hull.xz;
        float distance = length(separation);
        float other_radius = max(other_settings.size_mass.x, other_settings.size_mass.z) * 0.42f;
        float min_separation = collision_radius + other_radius + 0.035f;
        if (distance >= min_separation) {
            continue;
        }

        float2 separation_dir = distance > 1.0e-4f ?
            (separation / distance) :
            safeNormalize2D(anchor_xz - other_state.position_hull.xz,
                            float2(cos(push.time_seconds + object_index),
                                   sin(push.time_seconds + object_index)));
        position_correction += separation_dir * (min_separation - distance) * 0.55f;
    }

    position.xz += position_correction;
    linear_velocity.xz += position_correction / max(push.delta_time, 1.0e-4f) * 0.30f;
    position.x = clamp(position.x, -horizontal_bound, horizontal_bound);
    position.z = clamp(position.z, -horizontal_bound, horizontal_bound);

    float2 angular_acceleration = float2(total_torque.x * angular_strength,
                                         total_torque.z * angular_strength);
    angular_velocity += angular_acceleration * push.delta_time;

    float3 rotation = previous_rotation;
    angular_velocity += float2(-rotation.x, -rotation.z) * self_righting * push.delta_time;
    angular_velocity *= exp(-angular_damping * push.delta_time);

    rotation.x += angular_velocity.x * push.delta_time;
    rotation.z += angular_velocity.y * push.delta_time;
    rotation.x = clamp(rotation.x, -max_tilt_radians, max_tilt_radians);
    rotation.z = clamp(rotation.z, -max_tilt_radians, max_tilt_radians);

    float2 planar_velocity = linear_velocity.xz;
    if (length(planar_velocity) > 0.01f) {
        float desired_yaw = atan2(planar_velocity.x, planar_velocity.y);
        float yaw_delta = wrapAngle(desired_yaw - rotation.y);
        rotation.y += yaw_delta * yaw_follow_strength * push.delta_time;
    } else {
        float yaw_delta = wrapAngle(base_yaw - rotation.y);
        rotation.y += yaw_delta * 0.8f * push.delta_time;
    }

    float4 orientation_now = quatFromEuler(rotation);
    float3 axis_x = normalize(rotateByQuat(orientation_now, float3(1.0f, 0.0f, 0.0f)));
    float3 axis_y = normalize(rotateByQuat(orientation_now, float3(0.0f, 1.0f, 0.0f)));
    float3 axis_z = normalize(rotateByQuat(orientation_now, float3(0.0f, 0.0f, 1.0f)));

    float local_water_height = sampleWaterHeightAtXZ(position.xz, dims, object_index);
    float previous_water_height = sampleWaterHeightAtXZ(previous_position.xz, dims, object_index);
    float equilibrium_draft =
        clamp((GRAVITY * mass) / max(buoyancy_strength * TOTAL_BUOYANCY_WEIGHT, 1.0e-4f),
              size.y * 0.12f, size.y * 0.58f);
    float target_center_y = local_water_height + half_extents.y - equilibrium_draft + waterline_offset;
    float float_height_error = target_center_y - position.y;
    float height_stabilization =
        float_height_error * buoyancy_strength * 0.28f - linear_velocity.y * buoyancy_damping * 0.18f;
    linear_velocity.y += height_stabilization * push.delta_time;
    linear_velocity.y = clamp(linear_velocity.y, -max_downward_speed, max_upward_speed);
    position.y = max(position.y, target_center_y - size.y * 0.035f);

    float vertical_correction = 0.0f;
    const float max_allowed_sample_submersion = equilibrium_draft + size.y * 0.08f;
    [unroll]
    for (int point_index = 0; point_index < 13; ++point_index) {
        float4 pattern = BUOYANCY_POINT_PATTERN[point_index];
        float3 local_point = float3(pattern.x * half_extents.x, pattern.y * half_extents.y,
                                    pattern.z * half_extents.z);
        float3 rotated_point = axis_x * local_point.x + axis_y * local_point.y + axis_z * local_point.z;
        float3 world_point = position + rotated_point;
        float water_height = sampleWaterHeightAtXZ(world_point.xz, dims, object_index);
        vertical_correction = max(vertical_correction,
                                  water_height - world_point.y - max_allowed_sample_submersion);
    }
    if (vertical_correction > 0.0f) {
        float capped_vertical_correction = min(vertical_correction, size.y * 0.12f);
        position.y += capped_vertical_correction;
        linear_velocity.y = min(max(linear_velocity.y, 0.0f), max_upward_speed * 0.18f);
    }

    local_water_height = sampleWaterHeightAtXZ(position.xz, dims, object_index);
    float keel_height = position.y - half_extents.y;
    float submersion_depth = max(local_water_height - keel_height, 0.0f);
    float previous_keel_height = previous_position.y - half_extents.y;
    float previous_submersion_depth = max(previous_water_height - previous_keel_height, 0.0f);
    float submersion_ratio =
        clamp(submersion_depth / max(size.y, 1.0e-4f), 0.0f, 1.5f);
    float2 current_center_xz = position.xz;
    float frame_travel_distance = length(current_center_xz - previous_center_xz);
    float planar_speed = length(planar_velocity);
    float center_planar_speed = frame_travel_distance / max(push.delta_time, 1.0e-4f);
    float draft_raw = submersion_depth / max(push.height_to_world_scale, 1.0e-4f);
    float hull_displacement = clamp(draft_raw * 0.82f + submersion_ratio * 0.018f +
                                    center_planar_speed * 0.010f, 0.0f, 0.14f);
    float submersion_speed =
        abs(submersion_depth - previous_submersion_depth) / max(push.delta_time, 1.0e-4f);
    float angular_surface_speed = length(angular_velocity) * max(size.x, size.z) * 0.35f;
    float2 forward_from_yaw = float2(sin(rotation.y), cos(rotation.y));
    float2 wake_direction = center_planar_speed > 1.0e-4f ?
        (current_center_xz - previous_center_xz) / max(frame_travel_distance, 1.0e-4f) :
        forward_from_yaw;
    float wake_drive = max(center_planar_speed - 0.0015f, 0.0f);
    float heave_drive = max(submersion_speed + angular_surface_speed - 0.008f, 0.0f);
    float ripple_strength =
        clamp(submersion_ratio * (wake_drive * 0.40f + heave_drive * 0.16f), 0.0f, 0.024f);
    float sample_radius = clamp(max(size.x, size.z) * 0.30f, 0.060f, 0.16f);
    float wake_length =
        clamp(max(size.x, size.z) * (1.8f + center_planar_speed * 8.0f + submersion_ratio * 1.4f),
              sample_radius * 2.0f, 0.95f);

    FloatingObjectGpu render_data;
    render_data.center = float4(position, hull_displacement);
    render_data.axis_x = float4(axis_x, 0.0f);
    render_data.axis_y = float4(axis_y, 0.0f);
    render_data.axis_z = float4(axis_z, 0.0f);
    render_data.half_extents = float4(half_extents, 0.0f);
    render_data.color = float4(color, 0.0f);

    FloatingObjectInteractionGpu interaction_data;
    interaction_data.center_radius_strength = float4(0.0f, 0.0f, 0.0f, 0.0f);
    interaction_data.wake_direction_length = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (submersion_depth > 1.0e-4f && ripple_strength > 1.0e-4f) {
        interaction_data.center_radius_strength =
            float4(current_center_xz.x, current_center_xz.y, sample_radius, ripple_strength);
        interaction_data.wake_direction_length =
            float4(wake_direction.x, wake_direction.y, wake_length, max(planar_speed, center_planar_speed));
    }

    state.position_hull = float4(position, hull_displacement);
    state.rotation_pad = float4(rotation, 0.0f);
    state.linear_velocity_pad = float4(linear_velocity, 0.0f);
    state.angular_velocity_pad = float4(angular_velocity.x, angular_velocity.y, 0.0f, 0.0f);

    object_states[object_index] = state;
    floating_objects[object_index] = render_data;
    floating_object_interactions[object_index] = interaction_data;
}
