// ─── push constants & bindings ────────────────────────────────────────────────
struct PushConstants { uint frame; uint shape_count; uint material_count; };
[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

[[vk::binding(0, 0)]] RWTexture2D<float4> output_image;
[[vk::binding(1, 0)]] RWTexture2D<float4> accum_image;

// ─── scene buffers ────────────────────────────────────────────────────────────
struct GPUCamera {
    float4   position;
    float4   target;
    float4   up;
    float4   fov_near_far; // x: fov, y: near, z: far
};

struct GPUShape {
    float4x4 transform;
    float4x4 inverseTransform;
    int      type;      // 0: quad, 1: box, 2: sphere
    int      matIndex;
    float    _pad[2];
};

struct GPUMaterial {
    float4 albedo_type;    // rgb: albedo, a: type
    float4 emission;       // rgb: emission, w: intensity
};

[[vk::binding(2, 0)]] StructuredBuffer<GPUCamera>   camera_buf;
[[vk::binding(3, 0)]] StructuredBuffer<GPUShape>    shapes_buf;
[[vk::binding(4, 0)]] StructuredBuffer<GPUMaterial> materials_buf;

// ─── constants ────────────────────────────────────────────────────────────────
static const float PI         = 3.14159265358979f;
static const int   MAX_BOUNCES = 4;
static const float T_MIN      = 0.00001f;
static const float T_MAX      = 1e30f;

// ─── types ────────────────────────────────────────────────────────────────────
struct Ray { float3 origin; float3 dir; };
struct Hit { float t; float3 normal; bool front_face; int matIndex; };

// ─── RNG (unchanged) ──────────────────────────────────────────────────────────
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
    [mutating] float nextFloat() { return float(next()) * (1.0f / 4294967296.0f); }
    [mutating] float3 cosineHemisphere(float3 n) {
        float  u1  = nextFloat(), u2 = nextFloat();
        float  r   = sqrt(u1);
        float  phi = 2.0f * PI * u2;
        float3 up  = abs(n.y) < 0.999f ? float3(0,1,0) : float3(1,0,0);
        float3 T   = normalize(cross(up, n));
        float3 B   = cross(n, T);
        return normalize(T * (r * cos(phi)) + B * (r * sin(phi)) + n * sqrt(1.0f - u1));
    }
};

// ─── material ─────────────────────────────────────────────────────────────────
struct MaterialSample {
    float3 dir;
    float3 weight;
    bool   absorbed;
};

float D_GGX(float3 n, float3 h, float alpha) {
    float cosH  = max(dot(n, h), 0.0f);
    float a2    = alpha * alpha;
    float cos2  = cosH * cosH;
    float denom = cos2 * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * denom * denom + 1e-20f);
}

float G1_GGX(float3 n, float3 v, float alpha) {
    float cosV = max(dot(n, v), 0.0f);
    if (cosV <= 0.0f) return 0.0f;
    float tanV = sqrt(max(0.0f, 1.0f - cosV * cosV)) / cosV;
    float a    = 1.0f / (alpha * tanV);
    if (a < 1.6f)
        return (3.535f * a + 2.181f * a * a) / (1.0f + 2.276f * a + 2.577f * a * a);
    return 1.0f;
}

float3 fresnel_schlick(float cosTheta, float3 F0v) {
    return F0v + (1.0f - F0v) * pow(1.0f - cosTheta, 5.0f);
}

MaterialSample sampleCookTorrance(float3 inDir, float3 normal, inout RNG rng, float3 albedo) {
    MaterialSample s;
    float3 wi    = -inDir;
    float  cos_i = max(dot(normal, wi), 0.0f);
    if (cos_i <= 0.0f) { s.absorbed = true; s.weight = float3(0,0,0); return s; }

    float  roughness = 0.2f;
    float  alpha     = roughness * roughness;
    float3 F0        = float3(0.04f, 0.04f, 0.04f);
    float  p_spec    = 0.5f;

    float3 up       = abs(normal.y) < 0.999f ? float3(0,1,0) : float3(1,0,0);
    float3 tangent  = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    float r_choice = rng.nextFloat();
    if (r_choice < p_spec) {
        float u1 = rng.nextFloat(), u2 = rng.nextFloat();
        float phi    = 2.0f * PI * u1;
        float tan2   = alpha * alpha * u2 / max(1.0f - u2, 1e-6f);
        float cosThetaH = 1.0f / sqrt(1.0f + tan2);
        float sinThetaH = sqrt(max(0.0f, 1.0f - cosThetaH * cosThetaH));

        float3 h_t = float3(cos(phi) * sinThetaH, sin(phi) * sinThetaH, cosThetaH);
        float3 h   = normalize(tangent * h_t.x + bitangent * h_t.y + normal * h_t.z);
        float3 wo  = reflect(-wi, h);

        float cos_o = max(dot(normal, wo), 0.0f);
        if (cos_o <= 0.0f) { s.absorbed = true; s.weight = float3(0,0,0); return s; }

        float  D       = D_GGX(normal, h, alpha);
        float  G       = G1_GGX(normal, wi, alpha) * G1_GGX(normal, wo, alpha);
        float3 F       = fresnel_schlick(max(dot(h, wi), 0.0f), F0);
        float  n_dot_h  = max(dot(normal, h), 1e-6f);
        float  wo_dot_h = max(dot(wo, h), 1e-6f);

        s.dir      = wo;
        s.weight   = (F * G * wo_dot_h) / (p_spec * cos_i * n_dot_h);
        s.absorbed = false;
        return s;
    } else {
        float u1 = rng.nextFloat(), u2 = rng.nextFloat();
        float r   = sqrt(u1);
        float phi = 2.0f * PI * u2;
        float x   = r * cos(phi), y = r * sin(phi);
        float z   = sqrt(max(0.0f, 1.0f - x*x - y*y));
        float3 wo = normalize(tangent * x + bitangent * y + normal * z);

        float cos_o = max(dot(normal, wo), 0.0f);
        if (cos_o <= 0.0f) { s.absorbed = true; s.weight = float3(0,0,0); return s; }

        float3 F_at_normal = fresnel_schlick(cos_i, F0);
        float3 brdf_diff   = albedo / PI * (1.0f - max(F_at_normal.x, max(F_at_normal.y, F_at_normal.z)));
        float  full_pdf    = (1.0f - p_spec) * (cos_o / PI) + 1e-20f;

        s.dir      = wo;
        s.weight   = brdf_diff * cos_o / full_pdf;
        s.absorbed = false;
        return s;
    }
}

// ─── intersection ─────────────────────────────────────────────────────────────
bool intersectSphere(Ray ray, float4x4 transform, float4x4 invTransform, float tmin, float tmax, out Hit hit)
{
    hit = (Hit)0;
    float3 o = mul(invTransform, float4(ray.origin, 1.0f)).xyz;
    float3 d = mul(invTransform, float4(ray.dir,    0.0f)).xyz;

    float a    = dot(d, d);
    float hb   = dot(o, d);
    float c    = dot(o, o) - 1.0f;
    float disc = hb * hb - a * c;
    if (disc < 0.0f) return false;

    float sqrtd = sqrt(disc);
    float t = (-hb - sqrtd) / a;
    if (t < tmin || t > tmax) {
        t = (-hb + sqrtd) / a;
        if (t < tmin || t > tmax) return false;
    }

    float3 p    = o + t * d;
    float3 objN = normalize(p);
    float3 wN   = normalize(mul((float3x3)transpose(invTransform), objN));
    bool   front = dot(ray.dir, wN) < 0.0f;

    float3 pWorld = mul(transform, float4(p, 1.0)).xyz;
    float worldT  = dot(pWorld - ray.origin, ray.dir); // assuming ray.dir is normalized
    hit.t = worldT;
    hit.normal     = front ? wN : -wN;
    hit.front_face = front;
    return true;
}

bool intersectAABB(Ray ray, float4x4 transform, float4x4 invTransform, float tmin, float tmax, out Hit hit)
{
    hit = (Hit)0;
    float3 o = mul(invTransform, float4(ray.origin, 1.0f)).xyz;
    float3 d = mul(invTransform, float4(ray.dir,    0.0f)).xyz;

    float3 tMin3 = (-1.0f - o) / d;
    float3 tMax3 = ( 1.0f - o) / d;
    float3 t1    = min(tMin3, tMax3);
    float3 t2    = max(tMin3, tMax3);
    float  tNear = max(max(t1.x, t1.y), t1.z);
    float  tFar  = min(min(t2.x, t2.y), t2.z);

    if (tNear > tFar || tFar < tmin || tNear > tmax) return false;
    float t = tNear > tmin ? tNear : tFar;
    if (t > tmax) return false;

    float3 p    = o + t * d;
    float3 absP = abs(p);
    float3 objN;
    if      (absP.x > absP.y && absP.x > absP.z) objN = float3(sign(p.x), 0, 0);
    else if (absP.y > absP.z)                     objN = float3(0, sign(p.y), 0);
    else                                          objN = float3(0, 0, sign(p.z));

    float3 wN    = normalize(mul((float3x3)transpose(invTransform), objN));
    bool   front = dot(ray.dir, wN) < 0.0f;

    float3 pWorld = mul(transform, float4(p, 1.0)).xyz;
    float worldT  = dot(pWorld - ray.origin, ray.dir); // assuming ray.dir is normalized
    hit.t = worldT;
    hit.normal     = front ? wN : -wN;
    hit.front_face = front;
    return true;
}

bool intersectQuad(Ray ray, float4x4 transform, float4x4 invTransform, float tmin, float tmax, out Hit hit)
{
    hit = (Hit)0;
    float3 o = mul(invTransform, float4(ray.origin, 1.0f)).xyz;
    float3 d = mul(invTransform, float4(ray.dir,    0.0f)).xyz;

    if (abs(d.y) < 1e-6f) return false;
    float t = -o.y / d.y;
    if (t < tmin || t > tmax) return false;

    float3 p = o + t * d;
    if (abs(p.x) > 1.0f || abs(p.z) > 1.0f) return false;

    float3 objN  = float3(0, 1, 0);
    float3 wN    = normalize(mul((float3x3)transpose(invTransform), objN));
    bool   front = dot(ray.dir, wN) < 0.0f;

    float3 pWorld = mul(transform, float4(p, 1.0)).xyz;
    float worldT  = dot(pWorld - ray.origin, ray.dir); // assuming ray.dir is normalized
    hit.t = worldT;
    hit.normal     = front ? wN : -wN;
    hit.front_face = front;
    return true;
}

// ─── scene intersection ───────────────────────────────────────────────────────
bool intersectScene(Ray ray, out Hit best, out GPUMaterial mat)
{
    best.t    = T_MAX;
    bool didHit = false;

    for (uint i = 0; i < push.shape_count; ++i) {
        GPUShape shape = shapes_buf[i];
        Hit h;
        bool hit = false;

        if      (shape.type == 0) hit = intersectQuad  (ray, shape.transform, shape.inverseTransform, T_MIN, best.t, h);
        else if (shape.type == 1) hit = intersectAABB  (ray, shape.transform, shape.inverseTransform, T_MIN, best.t, h);
        else if (shape.type == 2) hit = intersectSphere(ray, shape.transform, shape.inverseTransform, T_MIN, best.t, h);

        if (hit) {
            best          = h;
            best.matIndex = shape.matIndex;
            didHit        = true;
        }
    }

    if (didHit) mat = materials_buf[best.matIndex];
    return didHit;
}

// ─── environment ──────────────────────────────────────────────────────────────
float3 skyColor(float3 dir) {
    float t = 0.5f * (normalize(dir).y + 1.0f);
    return lerp(float3(1,1,1), float3(0.5f, 0.7f, 1.0f), t);
}

// ─── path tracing ─────────────────────────────────────────────────────────────
float3 tracePath(Ray ray, inout RNG rng)
{
    float3 throughput = float3(1,1,1);
    float3 output     = float3(0,0,0);

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
        Hit        best;
        GPUMaterial mat;

        if (!intersectScene(ray, best, mat)) {
            output += throughput * skyColor(ray.dir);
            return output;
        }

        float3 albedo   = mat.albedo_type.rgb;
        float3 emission = mat.emission.rgb * mat.emission.w;

        // emissive contribution
        output += throughput * emission;

        float3 hitPos = ray.origin + best.t * ray.dir;

        MaterialSample ms = sampleCookTorrance(ray.dir, best.normal, rng, albedo);
        if (ms.absorbed) return output;

        throughput *= ms.weight;
        ray.origin  = hitPos + best.normal * T_MIN;
        ray.dir     = ms.dir;
    }

    return output;
}

// ─── camera ───────────────────────────────────────────────────────────────────
static float offset = 0.0f;
static float offset2 = 0.0f;
Ray makeRay(uint2 pixel, uint2 dims, float2 jitter)
{
    GPUCamera cam     = camera_buf[0];
    float3    origin  = cam.position.xyz;
    offset = sin(push.frame * 0.1f) * 1.0;
    offset2 = cos(push.frame * 0.1f) * 1.0;
    origin.x += offset;
    origin.y += offset2;
    origin.z += offset2;
    float3    target  = cam.target.xyz;
    float3    worldUp = cam.up.xyz;
    float     fov     = cam.fov_near_far.x;
    float     aspect  = float(dims.x) / float(dims.y);

    float  half_h = tan(fov * (PI / 180.0f) * 0.5f);
    float  half_w = aspect * half_h;

    float3 fwd   = normalize(target - origin);
    float3 right = normalize(cross(fwd, worldUp));
    float3 up    = cross(right, fwd);

    float2 uv = (float2(pixel) + jitter) / float2(dims);
    float  u  =  (2.0f * uv.x - 1.0f) * half_w;
    float  v  =  (1.0f - 2.0f * uv.y) * half_h;

    Ray ray;
    ray.origin = origin;
    ray.dir    = normalize(fwd + u * right + v * up);
    return ray;
}

// ─── entry point ──────────────────────────────────────────────────────────────
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint2 pixel = dispatch_id.xy;
    uint2 dims;
    output_image.GetDimensions(dims.x, dims.y);
    if (any(pixel >= dims)) return;

    RNG    rng    = RNG::make(pixel, push.frame);
    float2 jitter = float2(rng.nextFloat(), rng.nextFloat());

    float3 colorAccum = float3(0,0,0);

    for (int i = 0; i < 8; ++i) {
        jitter = float2(rng.nextFloat(), rng.nextFloat());
        colorAccum += tracePath(makeRay(pixel, dims, jitter), rng);
    }

    float3 color = colorAccum / 8.0f;
    float3 prev  = accum_image[pixel].rgb;
    float3 accum = lerp(prev, color, 1.0f / float(push.frame + 1u));

    accum_image[pixel] = float4(accum, 1.0f);
    // output_image[pixel] = float4(accum, 1.0f);
    output_image[pixel] = float4(color, 1.0f);
}