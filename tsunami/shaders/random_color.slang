struct PushConstants {
    uint frame;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

[[vk::binding(0, 0)]]
RWTexture2D<float4> output_image;

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint seed = dispatch_id.x
              ^ (dispatch_id.y << 16)
              ^ (push.frame * 719393u);

    float r = float(pcg(seed))      / float(0xFFFFFFFFu);
    float g = float(pcg(seed + 1u)) / float(0xFFFFFFFFu);
    float b = float(pcg(seed + 2u)) / float(0xFFFFFFFFu);

    output_image[dispatch_id.xy] = float4(r, g, b, 1.0);
}