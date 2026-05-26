//Helper lambda to upload cpu_buffer to the GPU texture
//This function is called every frame while painting, and also does one more call on mouse release, is a bottleneck, so we might want to use
//it only on simpler scenes. 
auto paint_do_upload = [&]() {
    if (paint.tex_index < 0 ||
        paint.tex_index >= static_cast<int>(render_target_ctx.mat_images.size()) ||
        render_target_ctx.mat_images[paint.tex_index] == VK_NULL_HANDLE) return;
    VkImage up_img = render_target_ctx.mat_images[paint.tex_index];
    VmaAllocator up_alloc = render_target_ctx.allocator;
    VmaAllocation up_sa;
    VkBuffer up_stg = create_and_upload_buffer(
        up_alloc, static_cast<VkDeviceSize>(paint.cpu_buffer.size()),
        paint.cpu_buffer.data(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, up_sa);
    VkCommandBuffer up_cmd = begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);
    transition_layout(up_cmd, up_img,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT,            VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy up_region{};
    up_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    up_region.imageSubresource.layerCount = 1;
    up_region.imageExtent                 = {paint.width, paint.height, 1};
    vkCmdCopyBufferToImage(up_cmd, up_stg, up_img,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &up_region);
    transition_layout(up_cmd, up_img,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT,   VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool,
                     vulkan_ctx.graphics_queue, up_cmd);
    vmaDestroyBuffer(up_alloc, up_stg, up_sa);
    frame_number = 0;
    needs_visibility_pass = true;
    reset_hipr_object_sampling();
};


//Input reading and paint window opening logic
{
    const bool p_down = glfwGetKey(m_window->handle(), GLFW_KEY_P) == GLFW_PRESS;

    if (p_down && !paint.prev_p && ui::selection_ctx.selected_mesh_index >= 0) {
        const int32_t mesh_idx = ui::selection_ctx.selected_mesh_index;

        if (mesh_idx >= 0 && mesh_idx < static_cast<int>(m_scene->m_meshes.size())) {
            auto& open_mesh = m_scene->m_meshes[mesh_idx];

            if (open_mesh && open_mesh->m_material) {
                paint.mesh_index  = mesh_idx;
                paint.created_tex = false;

                //Does the mesh already have an albedo texture?
                const uint32_t ati       = open_mesh->m_material->m_gpu.albedo_tex_index;
                const int      ati_count = static_cast<int>(render_target_ctx.mat_images.size());
                const bool     has_tex   = (ati != 0xFFFFFFFFu) &&
                                           (static_cast<int>(ati) < ati_count) &&
                                           (render_target_ctx.mat_images[ati] != VK_NULL_HANDLE);

                if (has_tex) {
                    //Use the existing texture for the paint window
                    paint.tex_index = static_cast<int32_t>(ati);
                    if (paint.tex_index < static_cast<int>(m_scene->m_textures.size()) &&
                        m_scene->m_textures[paint.tex_index]) {
                        paint.width  = static_cast<uint32_t>(m_scene->m_textures[paint.tex_index]->width);
                        paint.height = static_cast<uint32_t>(m_scene->m_textures[paint.tex_index]->height);
                    }
                } else {
                    //If there isn't an existing texture, create a new one and assign it to the material
                    paint.width  = 512;
                    paint.height = 512;

                    VmaAllocator blank_alloc = render_target_ctx.allocator;

                    VkImageCreateInfo blank_ii{};
                    blank_ii.sType                = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                    blank_ii.imageType            = VK_IMAGE_TYPE_2D;
                    blank_ii.format               = VK_FORMAT_R8G8B8A8_UNORM;
                    blank_ii.extent               = {paint.width, paint.height, 1};
                    blank_ii.mipLevels            = 1;
                    blank_ii.arrayLayers          = 1;
                    blank_ii.samples              = VK_SAMPLE_COUNT_1_BIT;
                    blank_ii.tiling               = VK_IMAGE_TILING_OPTIMAL;
                    blank_ii.usage                = VK_IMAGE_USAGE_SAMPLED_BIT |
                                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                    blank_ii.initialLayout        = VK_IMAGE_LAYOUT_UNDEFINED;

                    VmaAllocationCreateInfo blank_ai{};
                    blank_ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;

                    VkImage       blank_img;
                    VmaAllocation blank_imgalloc;

                    // create a new white texture and assign it to the material
                    if (vmaCreateImage(blank_alloc, &blank_ii, &blank_ai,
                                       &blank_img, &blank_imgalloc, nullptr) == VK_SUCCESS) {

                        VkImageView blank_view = create_image_view(
                            vulkan_ctx.device, blank_img,
                            VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);

                        // Upload white pixels to initialize the texture, might change it to a clear texture if this fucks with the stained glass idea
                        std::vector<uint8_t> white_pixels(
                            static_cast<size_t>(paint.width) * paint.height * 4, 255u);
                        VmaAllocation blank_sa;
                        VkBuffer blank_stg = create_and_upload_buffer(
                            blank_alloc, static_cast<VkDeviceSize>(white_pixels.size()),
                            white_pixels.data(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, blank_sa);

                        VkCommandBuffer blank_cmd = begin_one_time_cmd(
                            vulkan_ctx.device, command_ctx.command_pool);
                        transition_layout(blank_cmd, blank_img,
                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            0, VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                        VkBufferImageCopy blank_region{};
                        blank_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        blank_region.imageSubresource.layerCount = 1;
                        blank_region.imageExtent                 = {paint.width, paint.height, 1};
                        vkCmdCopyBufferToImage(blank_cmd, blank_stg, blank_img,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blank_region);
                        transition_layout(blank_cmd, blank_img,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                        end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool,
                                         vulkan_ctx.graphics_queue, blank_cmd);
                        vmaDestroyBuffer(blank_alloc, blank_stg, blank_sa);

                        // Register the new image in the material texture arrays
                        const int32_t new_idx = static_cast<int32_t>(render_target_ctx.mat_images.size());
                        render_target_ctx.mat_images.push_back(blank_img);
                        render_target_ctx.mat_allocs.push_back(blank_imgalloc);
                        render_target_ctx.mat_views.push_back(blank_view);

                        // Update the descriptor set so the shader can sample the new texture
                        VkDescriptorImageInfo blank_img_info{};
                        blank_img_info.imageView   = blank_view;
                        blank_img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        VkWriteDescriptorSet blank_write{};
                        blank_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        blank_write.dstSet          = render_target_ctx.descriptor_set;
                        blank_write.dstBinding      = 11;
                        blank_write.dstArrayElement = static_cast<uint32_t>(new_idx);
                        blank_write.descriptorCount = 1;
                        blank_write.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                        blank_write.pImageInfo      = &blank_img_info;
                        vkUpdateDescriptorSets(vulkan_ctx.device, 1, &blank_write, 0, nullptr);

                        // Wire the new texture into the material so the shader uses it
                        paint.tex_index   = new_idx;
                        paint.created_tex = true;
                        open_mesh->m_material->m_gpu.albedo_tex_index = static_cast<uint32_t>(new_idx);
                        ui::applySelectedMaterialEditor(
                            m_scene.get(), render_target_ctx.allocator,
                            scene_ctx.material_mapped, scene_ctx.material_count,
                            scene_ctx.material_alloc);
                    }
                }

                //Initialize CPU buffer
                // For existing textures, read back from GPU so we paint over what's there.
                // For new textures, fill with white. TODO: maybe change to a clear texture if this fucks with the stained glass idea
                paint.cpu_buffer.resize(
                    static_cast<size_t>(paint.width) * paint.height * 4, 255u);

                // Read back from GPU if the texture already exists
                if (has_tex) {
                    VkImage      rb_img   = render_target_ctx.mat_images[paint.tex_index];
                    VmaAllocator rb_alloc = render_target_ctx.allocator;

                    VkBufferCreateInfo rb_bi{};
                    rb_bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    rb_bi.size  = static_cast<VkDeviceSize>(paint.cpu_buffer.size());
                    rb_bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

                    VmaAllocationCreateInfo rb_ai{};
                    rb_ai.usage = VMA_MEMORY_USAGE_CPU_ONLY;
                    rb_ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

                    VkBuffer      rb_buf;
                    VmaAllocation rb_sa;
                    VmaAllocationInfo rb_info;

                    if (vmaCreateBuffer(rb_alloc, &rb_bi, &rb_ai, &rb_buf, &rb_sa, &rb_info) == VK_SUCCESS) {
                        VkCommandBuffer rb_cmd = begin_one_time_cmd(
                            vulkan_ctx.device, command_ctx.command_pool);
                        transition_layout(rb_cmd, rb_img,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_ACCESS_SHADER_READ_BIT,            VK_ACCESS_TRANSFER_READ_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                        VkBufferImageCopy rb_region{};
                        rb_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        rb_region.imageSubresource.layerCount = 1;
                        rb_region.imageExtent                 = {paint.width, paint.height, 1};
                        vkCmdCopyImageToBuffer(rb_cmd, rb_img,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb_buf, 1, &rb_region);
                        transition_layout(rb_cmd, rb_img,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_ACCESS_TRANSFER_READ_BIT,    VK_ACCESS_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                        end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool,
                                         vulkan_ctx.graphics_queue, rb_cmd);
                        memcpy(paint.cpu_buffer.data(), rb_info.pMappedData, paint.cpu_buffer.size());
                        vmaDestroyBuffer(rb_alloc, rb_buf, rb_sa);
                    }
                }

                //Register texture with ImGui so it can be shown as a canvas
                if (paint.imgui_ds != VK_NULL_HANDLE) {
                    ImGui_ImplVulkan_RemoveTexture(paint.imgui_ds);
                    paint.imgui_ds = VK_NULL_HANDLE;
                }
                if (paint.tex_index >= 0 &&
                    paint.tex_index < static_cast<int>(render_target_ctx.mat_views.size()) &&
                    render_target_ctx.mat_views[paint.tex_index] != VK_NULL_HANDLE) {
                    paint.imgui_ds = ImGui_ImplVulkan_AddTexture(
                        render_target_ctx.material_sampler,
                        render_target_ctx.mat_views[paint.tex_index],
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                }

                paint.window_open = true;
                /*
                std::cout << "[PAINT] mesh=" << mesh_idx
                          << " tex="     << paint.tex_index
                          << " created=" << paint.created_tex
                          << " "         << paint.width << "x" << paint.height << "\n";
                */
            }
        }
    }

    paint.prev_p = p_down;
}


//Paint window UI and painting logic
if (paint.window_open) {
    ImGui::SetNextWindowSize(ImVec2(540, 600), ImGuiCond_FirstUseEver);
    ImGui::Begin("Texture Painter", &paint.window_open,
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    //Audience color swatch and brush radius slider
    const glm::vec4 pen_col = audienceToStainedGlassColor(scene_ctx.selection_voice_loudness);
    ImGui::ColorButton("##pen",
        ImVec4(pen_col.r, pen_col.g, pen_col.b, pen_col.a),
        ImGuiColorEditFlags_NoTooltip, ImVec2(40, 40));
    ImGui::SameLine();
    ImGui::Text("Audience color   Brush radius:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::SliderFloat("##r", &paint.pen_radius, 1.f, 60.f);

    if (paint.imgui_ds != VK_NULL_HANDLE) {

        //Scale canvas to fit inside the window
        const float  scale       = std::min(1.f, 500.f / static_cast<float>(paint.width));
        const ImVec2 canvas_size = ImVec2(paint.width * scale, paint.height * scale);
        const ImVec2 canvas_pos  = ImGui::GetCursorScreenPos();

        // Draw the texture as an image
        ImGui::Image((ImTextureID)(void*)paint.imgui_ds, canvas_size);

        // Invisible button overlaid on the image, absolutely fucked up
        ImGui::SetCursorScreenPos(canvas_pos);
        ImGui::InvisibleButton("##canvas_interact", canvas_size);

        const bool is_painting = ImGui::IsItemActive();
        const bool released    = ImGui::IsItemDeactivated();

        //Write brush strokes into cpu_buffer
        if (is_painting) {
            const ImVec2  mouse = ImGui::GetMousePos();
            const int     cx    = static_cast<int>((mouse.x - canvas_pos.x) / scale);
            const int     cy    = static_cast<int>((mouse.y - canvas_pos.y) / scale);
            const int     r     = static_cast<int>(paint.pen_radius);
            const uint8_t pr    = static_cast<uint8_t>(pen_col.r * 255.f);
            const uint8_t pg    = static_cast<uint8_t>(pen_col.g * 255.f);
            const uint8_t pb    = static_cast<uint8_t>(pen_col.b * 255.f);

            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    if (dx * dx + dy * dy > r * r) continue; //loop for defining a circular brush 
                    const int tx = cx + dx;
                    const int ty = cy + dy;
                    if (tx < 0 || ty < 0 || tx >= static_cast<int>(paint.width) || ty >= static_cast<int>(paint.height)) continue;
                    const size_t pidx = (static_cast<size_t>(ty) * paint.width + tx) * 4;
                    paint.cpu_buffer[pidx + 0] = pr;
                    paint.cpu_buffer[pidx + 1] = pg;
                    paint.cpu_buffer[pidx + 2] = pb;
                    paint.cpu_buffer[pidx + 3] = 255u;
                }
            }
        }

        // Was uploading on mouse release, but that was causing a delay in the paint strokes appearing on the mesh. Now uploading every frame while painting, which is more responsive but may be less efficient.
        if (is_painting || released) {
            paint_do_upload();
        }

    } else {
        ImGui::TextDisabled("No texture available.");
    }

    ImGui::End();

    //Garbage collection stuff
    if (!paint.window_open && paint.imgui_ds != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(paint.imgui_ds);
        paint.imgui_ds = VK_NULL_HANDLE;
    }
}