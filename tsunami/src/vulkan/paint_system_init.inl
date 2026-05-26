struct PaintState {
    std::vector<uint8_t> cpu_buffer;
    uint32_t             width       = 512;
    uint32_t             height      = 512;
    int32_t              mesh_index  = -1;
    int32_t              tex_index   = -1;
    VkDescriptorSet      imgui_ds    = VK_NULL_HANDLE;
    bool                 window_open = false;
    bool                 prev_p      = false;
    float                pen_radius  = 12.f;
    bool                 created_tex = false;
    bool                 needs_upload = false;
} paint;

const auto audienceToStainedGlassColor = [](float loudness) -> glm::vec4 {
    static const glm::vec4 colorOptions[] = { // a more neon-ish/ vibrant palette for stained glass 
        {0.85f, 0.07f, 0.07f, 1.f}, //red
        {0.95f, 0.55f, 0.02f, 1.f}, //orange
        {0.05f, 0.65f, 0.15f, 1.f}, //green
        {0.05f, 0.20f, 0.85f, 1.f}, //blue 
        {0.50f, 0.05f, 0.80f, 1.f}, //purple
    };
    constexpr int n = 5;
    const int idx = static_cast<int>(std::clamp(loudness, 0.f, 1.f) * (n - 1) + 0.5f);
    return colorOptions[std::clamp(idx, 0, n - 1)];
};