struct PushConstants {
    uint   frame;
    uint   material_count;
    int    selected_mesh_index;
    uint   outline_width;
    int    debug_view_mode;
    uint   stage;
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

[[vk::binding(0, 0)]]  RWTexture2D<float4>             output_image;
[[vk::binding(2, 0)]]  StructuredBuffer<GPUCamera>     camera_buf;
[[vk::binding(5, 0)]]  RaytracingAccelerationStructure tlas;
[[vk::binding(13, 0)]] RWTexture2D<int>                object_id_image;

struct GPUCamera {
    float4 position;
    float4 target;
    float4 up;
    float4 fov_near_far;
};

struct Ray {
    float3 origin;
    float3 dir;
};

static const float T_MIN = 1e-3f;
static const float T_MAX = 1e30f;

float3 safeNormalize(float3 v, float3 fallback) {
    float len2 = dot(v, v);
    if (len2 > 1.0e-12f) {
        return v * rsqrt(len2);
    }
    return fallback;
}

Ray makeRay(uint2 pixel, uint2 dims, float2 jitter) {
    GPUCamera cam = camera_buf[0];

    float3 camPos = cam.position.xyz;
    float3 forward = safeNormalize(cam.target.xyz - camPos, float3(0.f, 0.f, -1.f));
    float3 upInput = safeNormalize(cam.up.xyz, float3(0.f, 1.f, 0.f));
    float3 right = safeNormalize(cross(forward, upInput), float3(1.f, 0.f, 0.f));
    float3 up = safeNormalize(cross(right, forward), float3(0.f, 1.f, 0.f));

    float aspect = float(dims.x) / max(float(dims.y), 1.f);
    float fovY = radians(cam.fov_near_far.x);
    float tanHalfFov = tan(0.5f * fovY);

    float2 uv = (float2(pixel) + jitter) / float2(dims);
    float2 ndc = uv * 2.f - 1.f;
    ndc.y = -ndc.y;

    float3 dir = safeNormalize(
        forward + ndc.x * aspect * tanHalfFov * right + ndc.y * tanHalfFov * up,
        forward);

    Ray r;
    r.origin = camPos;
    r.dir = dir;
    return r;
}

int traceVisibleInstance(Ray ray) {
    RayDesc ray_desc;
    ray_desc.Origin = ray.origin;
    ray_desc.Direction = ray.dir;
    ray_desc.TMin = T_MIN;
    ray_desc.TMax = T_MAX;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray_desc);
    while (query.Proceed()) {}

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        return int(query.CommittedInstanceID());
    }
    return -1;
}

bool isSelectionOutline(uint2 pixel, uint2 dims, int visible_object_id) {
    if (push.selected_mesh_index < 0 || visible_object_id != push.selected_mesh_index) {
        return false;
    }

    int outline_width = int(max(push.outline_width, 1u));
    int2 pixel_i = int2(pixel);
    int2 dims_i = int2(dims);

    int2 neighbor_offsets[4] = {
        int2(-1, 0),
        int2(1, 0),
        int2(0, -1),
        int2(0, 1)
    };

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

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 pixel = dispatch_id.xy;
    uint2 dims;
    output_image.GetDimensions(dims.x, dims.y);

    if (any(pixel >= dims)) return;

    // Stage 0: build object_id_image.
    if (push.stage == 0) {
        int visible_id = traceVisibleInstance(makeRay(pixel, dims, float2(0.5f, 0.5f)));
        object_id_image[pixel] = visible_id;
        return;
    }

    // ObjectID mode writes only in stage 2.
    if (push.stage != 2) return;

    int visible_object_id = object_id_image[pixel];
    float3 display = debugColorForObjectId(visible_object_id);

    if (isSelectionOutline(pixel, dims, visible_object_id)) {
        display = lerp(display, push.outline_color.rgb, saturate(push.outline_color.a));
    }

    output_image[pixel] = float4(display, 1.0f);
}
