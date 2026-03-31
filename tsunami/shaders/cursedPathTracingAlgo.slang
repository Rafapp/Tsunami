// ─── push constants & bindings ────────────────────────────────────────────────
struct PushConstants { uint frame; };
[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

[[vk::binding(0, 0)]] RWTexture2D<float4> output_image;
[[vk::binding(1, 0)]] RWTexture2D<float4> accum_image;

// ─── scene constants ──────────────────────────────────────────────────────────
static const float PI = 3.14159265358979f;

// camera
static const float3 CAM_ORIGIN  = float3(0.0f, 0.0f,  1.0f);
static const float3 CAM_TARGET  = float3(0.0f, 0.0f, -1.0f);
static const float3 WORLD_UP    = float3(0.0f, 1.0f,  0.0f);
static const float  CAM_FOV_DEG = 60.0f;
static const float  CAM_ASPECT  = 16.0f / 9.0f;

// sphere
static const float3 SPHERE_CENTER = float3(0.0f,  0.0f,  -1.0f);
static const float  SPHERE_RADIUS = 0.5f;
static const float3 SPHERE_ALBEDO = float3(0.8f,  0.3f,   0.1f);

// ground
static const float3 GROUND_CENTER = float3(0.0f, -100.5f, -1.0f);
static const float  GROUND_RADIUS = 100.0f;
static const float3 GROUND_ALBEDO = float3(0.5f,  0.5f,   0.5f);

// path tracing
static const int   MAX_BOUNCES = 8;
static const float T_MIN       = 0.00001f;
static const float T_MAX       = 1e30f;

// point light
static const float3 LIGHT_POS   = float3(2.0f, 2.0f, 0.0f);
static const float3 LIGHT_COLOR = float3(10.0f, 10.0f, 10.0f);

// ─── types ────────────────────────────────────────────────────────────────────
struct Ray { float3 origin; float3 dir; };
struct Hit { float  t; float3 normal; bool front_face; };

// ─── PCG RNG ──────────────────────────────────────────────────────────────────
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
        float  u1  = nextFloat();
        float  u2  = nextFloat();
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
    float3 weight;   // throughput multiplier
    bool   absorbed;
};

float D_GGX(float3 n, float3 h, float alpha)
{
    float cosH = max(dot(n, h), 0.0f);
    float a2   = alpha * alpha;
    float cos2 = cosH * cosH;
    float denom = cos2 * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * denom * denom + 1e-20f);
}

float G1_GGX(float3 n, float3 v, float alpha)
{
    float cosV = max(dot(n, v), 0.0f);
    if (cosV <= 0.0f) return 0.0f;

    float tanV = sqrt(max(0.0f, 1.0f - cosV * cosV)) / cosV;
    float a = 1.0f / (alpha * tanV);

    if (a < 1.6f)
        return (3.535f * a + 2.181f * a * a) / (1.0f + 2.276f * a + 2.577f * a * a);

    return 1.0f;
}

float3 fresnel_schlick(float cosTheta, float3 F0v)
{
    return F0v + (1.0f - F0v) * pow(1.0f - cosTheta, 5.0f);
}

MaterialSample sampleCookTorrance(float3 inDir, float3 normal, inout RNG rng, float3 albedo)
{
    // Implements a two-lobe material: cosine-weighted diffuse + GGX microfacet specular
    MaterialSample s;

    float3 wi = -inDir; // incident direction toward the surface
    float cos_i = max(dot(normal, wi), 0.0f);
    if (cos_i <= 0.0f) { s.absorbed = true; s.weight = float3(0,0,0); return s; }

    // parameters
    float roughness = 0.2f;
    float alpha = roughness * roughness;
    float3 F0 = float3(0.04f, 0.04f, 0.04f); // dielectric base reflectance
    float p_spec = 0.5f; // mixture weight between specular and diffuse

    // build TBN
    float3 up = abs(normal.y) < 0.999f ? float3(0,1,0) : float3(1,0,0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    // choose lobe
    float r_choice = rng.nextFloat();
    if (r_choice < p_spec) {
        // --- Specular: sample GGX half-vector ---
        float u1 = rng.nextFloat();
        float u2 = rng.nextFloat();

        float phi = 2.0f * PI * u1;
        float tan2 = alpha * alpha * u2 / max(1.0f - u2, 1e-6f);
        float cosThetaH = 1.0f / sqrt(1.0f + tan2);
        float sinThetaH = sqrt(max(0.0f, 1.0f - cosThetaH * cosThetaH));

        float3 h_t = float3(cos(phi) * sinThetaH, sin(phi) * sinThetaH, cosThetaH);
        float3 h = normalize(tangent * h_t.x + bitangent * h_t.y + normal * h_t.z);

        float3 wo = reflect(-wi, h);
        float cos_o = max(dot(normal, wo), 0.0f);
        if (cos_o <= 0.0f) { s.absorbed = true; s.weight = float3(0,0,0); return s; }

        float D = D_GGX(normal, h, alpha);
        float G = G1_GGX(normal, wi, alpha) * G1_GGX(normal, wo, alpha);
        float3 F = fresnel_schlick(max(dot(h, wi), 0.0f), F0);

        // BRDF (microfacet specular)
        // use algebraically-simplified throughput when sampling the half-vector:
        // weight = (F * G * (wo.h)) / (p_spec * cos_i * (n.h))
        float n_dot_h = max(dot(normal, h), 1e-6f);
        float wo_dot_h = max(dot(wo, h), 1e-6f);

        s.dir = wo;
        s.weight = (F * G * wo_dot_h) / (p_spec * cos_i * n_dot_h);
        s.absorbed = false;
        return s;
    } else {
        // --- Diffuse: sample cosine-weighted hemisphere ---
        float u1 = rng.nextFloat();
        float u2 = rng.nextFloat();
        float r = sqrt(u1);
        float phi = 2.0f * PI * u2;
        float x = r * cos(phi);
        float y = r * sin(phi);
        float z = sqrt(max(0.0f, 1.0f - x*x - y*y));
        float3 wo = normalize(tangent * x + bitangent * y + normal * z);

        float cos_o = max(dot(normal, wo), 0.0f);
        if (cos_o <= 0.0f) { s.absorbed = true; s.weight = float3(0,0,0); return s; }

        // Fresnel at normal to reduce diffuse energy where specular reflects
        float3 F_at_normal = fresnel_schlick(cos_i, F0);

        float3 brdf_diff = albedo / PI * (1.0f - max(F_at_normal.x, max(F_at_normal.y, F_at_normal.z)));

        float full_pdf = (1.0f - p_spec) * (cos_o / PI) + 1e-20f;

        s.dir = wo;
        s.weight = brdf_diff * cos_o / full_pdf;
        s.absorbed = false;
        return s;
    }
}

// ─── intersection ─────────────────────────────────────────────────────────────
bool intersectSphere(Ray ray, float3 center, float radius,
                     float tmin, float tmax, out Hit hit)
{
    hit = Hit();
    float3 oc   = ray.origin - center;
    float  a    = dot(ray.dir, ray.dir);
    float  hb   = dot(oc, ray.dir);
    float  c    = dot(oc, oc) - radius * radius;
    float  disc = hb * hb - a * c;
    if (disc < 0.0f) return false;

    float sqrtd = sqrt(disc);
    float t     = (-hb - sqrtd) / a;
    if (t < tmin || t > tmax) {
        t = (-hb + sqrtd) / a;
        if (t < tmin || t > tmax) return false;
    }

    float3 p    = ray.origin + t * ray.dir;
    float3 outN = normalize(p - center);
    bool   front = dot(ray.dir, outN) < 0.0f;

    hit.t          = t;
    hit.normal     = front ? outN : -outN;
    hit.front_face = front;
    return true;
}

// ─── environment ──────────────────────────────────────────────────────────────
float3 skyColor(float3 dir) {
    float t_anim = push.frame * 0.01f;

    float3 ColorAnimated = float3(
        0.5f + 0.5f * sin(t_anim),
        0.5f + 0.5f * sin(t_anim + 2.0f),
        0.5f + 0.5f * sin(t_anim + 4.0f)
    );

    float t = 0.5f * (normalize(dir).y + 1.0f);
    return lerp(float3(1, 1, 1), ColorAnimated, t);
}

// ─── path tracing ─────────────────────────────────────────────────────────────
float3 tracePath(Ray ray, inout RNG rng)
{
    float3 throughput = float3(1,1,1);
    float3 output     = float3(0,0,0);

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce)
    {
        Hit    best;
        float3 albedo;
        bool   didHit = false;
        float  tBest  = T_MAX;

        Hit h;
        if (intersectSphere(ray, SPHERE_CENTER, SPHERE_RADIUS, T_MIN, tBest, h)) {
            best = h; tBest = h.t; albedo = SPHERE_ALBEDO; didHit = true;
        }
        if (intersectSphere(ray, GROUND_CENTER, GROUND_RADIUS, T_MIN, tBest, h)) {
            best = h; albedo = GROUND_ALBEDO; didHit = true;
        }

        if (!didHit) {
            output += throughput * skyColor(ray.dir);
            return output;
        }

        float3 hitPos = ray.origin + best.t * ray.dir;

        // ── direct light ──────────────────────────────────────────────────────
        float3 toLight    = LIGHT_POS - hitPos;
        float  distToLight = length(toLight);
        float3 lightDir   = toLight / distToLight;

        Ray shadowRay;
        shadowRay.origin = hitPos + (best.front_face ? best.normal : -best.normal) * T_MIN;
        shadowRay.dir    = lightDir;

        Hit shadowHit;
        bool inShadow = intersectSphere(shadowRay, SPHERE_CENTER, SPHERE_RADIUS, T_MIN, distToLight, shadowHit)
                     || intersectSphere(shadowRay, GROUND_CENTER, GROUND_RADIUS, T_MIN, distToLight, shadowHit);

        if (!inShadow && best.front_face) {
            float  cosTheta = max(dot(best.normal, lightDir), 0.001f);
            if (cosTheta <= 0.0f) continue;   
            float3 diffuse  = albedo * cosTheta;

            float3 viewDir  = -ray.dir;
            float3 halfVec  = normalize(viewDir + lightDir);
            float  spec     = pow(max(dot(best.normal, halfVec), 0.0f), 64.0f);
            float3 specular = albedo * spec;

            output += throughput * (diffuse + specular) * LIGHT_COLOR / (distToLight * distToLight);
        }

        // ── scatter ───────────────────────────────────────────────────────────
        MaterialSample ms = sampleCookTorrance(ray.dir, best.normal, rng, albedo);
        if (ms.absorbed) return output;
        throughput *= ms.weight;
        ray.origin  = hitPos + best.normal * T_MIN;
        ray.dir     = ms.dir;
    }

    return output;
}

// ─── camera ───────────────────────────────────────────────────────────────────
Ray makeRay(uint2 pixel, uint2 dims, float2 jitter)
{
    float  half_h = tan(CAM_FOV_DEG * (PI / 180.0f) * 0.5f);
    float  half_w = CAM_ASPECT * half_h;

    float3 fwd   = normalize(CAM_TARGET - CAM_ORIGIN);
    float3 right = normalize(cross(fwd, WORLD_UP));
    float3 up    = cross(right, fwd);

    float2 uv = (float2(pixel) + jitter) / float2(dims);
    float  u  =  (2.0f * uv.x - 1.0f) * half_w;
    float  v  =  (1.0f - 2.0f * uv.y) * half_h;

    Ray ray;
    ray.origin = CAM_ORIGIN;
    ray.dir    = normalize(fwd + u * right + v * up);
    return ray;
}

// ─── entry point ─────────────────────────────────────────────────────────────
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
    float3 color  = tracePath(makeRay(pixel, dims, jitter), rng);

    float3 prev  = accum_image[pixel].rgb;
    float3 accum = lerp(prev, color, 1.0f / float(push.frame + 1u));

    accum_image[pixel] = float4(color, 1.0f);
    output_image[pixel] = float4(sqrt(max(color, 0.0f)), 1.0f);
}