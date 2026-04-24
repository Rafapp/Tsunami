#include "internal/pathtracer_hipr_family_shared.slang"

float3 heatmapRamp(float t) {
    t = saturate(t);
    float3 red  = float3(0.95f, 0.20f, 0.10f);
    float3 blue = float3(0.12f, 0.18f, 0.45f);
    return lerp(red, blue, t);
}

float3 hiprInfluenceTintColor(int rank) {
    if (rank < 0) return float3(1.0f, 1.0f, 1.0f);
    if (push.hipr_vis_rainbow_tint == 0u) {
        return float3(1.0f, 0.82f, 0.22f);
    }

    float denom = max(float(max(int(push.hipr_top_k), 1) - 1), 1.0f);
    float heat  = float(rank) / denom;
    return heatmapRamp(heat);
}

int hiprRankForObject(int object_id) {
    if (object_id < 0) return -1;
    if (push.hipr_top_k == 0) return -1;
    [loop]
    for (uint rank = 0; rank < push.hipr_top_k; ++rank) {
        if (hipr_order[rank] == object_id) {
            return int(rank);
        }
    }
    return -1;
}

float3 srgbToLinear(float3 color) {
    float3 cutoff = step(float3(0.04045f, 0.04045f, 0.04045f), color);
    float3 low    = color / 12.92f;
    float3 high   = pow((color + 0.055f) / 1.055f, float3(2.4f, 2.4f, 2.4f));
    return lerp(low, high, cutoff);
}

float3 unchartedTonemapPartial(float3 color) {
    float3 x = max(float3(0.0f, 0.0f, 0.0f), color - 0.004f);
    return (x * (6.2f * x + 0.5f)) / (x * (6.2f * x + 1.7f) + 0.06f);
}

float3 unchartedTonemapFilmic(float3 color, float exposureBias) {
    float3 exposed = color * exposureBias;
    exposed        = unchartedTonemapPartial(exposed);
    float3 W       = unchartedTonemapPartial(float3(11.2f, 11.2f, 11.2f));
    float3 whiteScale = 1.0f / max(W, float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
    return exposed * whiteScale;
}

// --- Entry point ---

float luminance(float3 color) {
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

float historyMismatch(float3 previous_color, float3 current_color) {
    float previous_luma = luminance(previous_color);
    float current_luma = luminance(current_color);
    float luma_denom = max(max(previous_luma, current_luma), 0.12f);
    float relative_luma_delta = abs(current_luma - previous_luma) / luma_denom;

    float3 rgb_delta = abs(previous_color - current_color);
    float rgb_denom = max(max(max(previous_color.r, previous_color.g),
                              max(previous_color.b, max(current_color.r,
                                                        max(current_color.g, current_color.b)))),
                          0.12f);
    float relative_rgb_delta = max(rgb_delta.r, max(rgb_delta.g, rgb_delta.b)) / rgb_denom;
    float combined_delta = max(relative_luma_delta, relative_rgb_delta * 0.85f);
    return smoothstep(0.22f, 0.90f, combined_delta);
}

float3 temporalOutlierClamp(float3 previous_color, float3 current_color, float mismatch) {
    float history_luma = luminance(previous_color);
    float floor_value = 0.22f + history_luma * 1.2f;
    float relaxed_scale = lerp(2.2f, 6.5f, saturate(mismatch));
    float3 max_color = previous_color * relaxed_scale + floor_value.xxx;
    float3 guarded_max = max(max_color, current_color * 0.60f);
    return min(current_color, guarded_max);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 pixel = dispatch_id.xy;
    uint2 dims;
    output_image.GetDimensions(dims.x, dims.y);

    if (push.stage == 10) {
        if (pixel.y != 0) return;
        uint index = pixel.x;
        if (index < push.hipr_object_count) {
            hipr_visible_counts[index]   = 0u;
            hipr_secondary_counts[index] = 0u;
            hipr_shadow_counts[index]    = 0u;
            if (push.hipr_clear_order != 0u) {
                hipr_scores[index] = 0.0f;
            }
        }
        if (push.hipr_clear_order != 0u && index < push.hipr_top_k) {
            hipr_order[index] = -1;
        }
        return;
    }

    if (any(pixel >= dims)) return;

    if (push.stage == 3) {
        if (pixel.x != 0 || pixel.y != 0) return;
        if (push.selected_mesh_index < 0 || push.hipr_object_count == 0 || push.hipr_top_k == 0)
            return;

        const bool incremental = (push.hipr_incremental_sort != 0u) && (push.hipr_clear_order == 0u);
        const float score_blend = saturate(push.hipr_score_blend);

        for (uint i = 0; i < push.hipr_object_count; ++i) {
            float visible = float(max(hipr_visible_counts[i], 1u));
            float instant_score =
                (float(hipr_secondary_counts[i]) + 0.5f * float(hipr_shadow_counts[i])) / visible;
            if (incremental) {
                hipr_scores[i] = lerp(hipr_scores[i], instant_score, score_blend);
            } else {
                hipr_scores[i] = instant_score;
            }
        }

        if (!incremental) {
            for (uint i = 0; i < push.hipr_top_k; ++i) {
                hipr_order[i] = -1;
            }

            hipr_order[0] = push.selected_mesh_index;
            for (uint rank = 1; rank < push.hipr_top_k; ++rank) {
                float best_score = -1.0f;
                int   best_id    = -1;
                for (uint obj = 0; obj < push.hipr_object_count; ++obj) {
                    if (int(obj) == push.selected_mesh_index) continue;

                    bool already_chosen = false;
                    for (uint prev = 0; prev < rank; ++prev) {
                        if (hipr_order[prev] == int(obj)) {
                            already_chosen = true;
                            break;
                        }
                    }
                    if (already_chosen) continue;

                    float s = hipr_scores[obj];
                    if (s > best_score) {
                        best_score = s;
                        best_id    = int(obj);
                    }
                }

                if (best_id < 0) break;
                hipr_order[rank] = best_id;
            }
        } else {
            // Keep previous ranking as the base ordering to avoid disruptive reshuffles.
            int seen[64];
            uint seen_count = 0u;
            [unroll]
            for (uint i = 0; i < 64; ++i) {
                seen[i] = -1;
            }

            hipr_order[0] = push.selected_mesh_index;
            seen[seen_count++] = push.selected_mesh_index;

            uint write_rank = 1u;
            for (uint rank = 1; rank < push.hipr_top_k && write_rank < push.hipr_top_k; ++rank) {
                int id = hipr_order[rank];
                if (id < 0 || id == push.selected_mesh_index || uint(id) >= push.hipr_object_count) {
                    continue;
                }
                bool duplicate = false;
                for (uint i = 0; i < seen_count; ++i) {
                    if (seen[i] == id) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) continue;
                hipr_order[write_rank++] = id;
                if (seen_count < 64u) seen[seen_count++] = id;
            }

            while (write_rank < push.hipr_top_k) {
                float best_score = -1.0f;
                int   best_id    = -1;
                for (uint obj = 0; obj < push.hipr_object_count; ++obj) {
                    int id = int(obj);
                    if (id == push.selected_mesh_index) continue;
                    bool duplicate = false;
                    for (uint i = 0; i < seen_count; ++i) {
                        if (seen[i] == id) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (duplicate) continue;

                    float s = hipr_scores[obj];
                    if (s > best_score) {
                        best_score = s;
                        best_id    = id;
                    }
                }
                if (best_id < 0) {
                    hipr_order[write_rank++] = -1;
                } else {
                    hipr_order[write_rank++] = best_id;
                    if (seen_count < 64u) seen[seen_count++] = best_id;
                }
            }

            // Gentle local swaps only when score improvement is significant.
            const float swap_threshold = 1.08f;
            for (uint pass = 0; pass < push.hipr_top_k; ++pass) {
                for (uint rank = 1; rank + 1 < push.hipr_top_k; ++rank) {
                    int a = hipr_order[rank];
                    int b = hipr_order[rank + 1u];
                    if (a < 0 || b < 0 || uint(a) >= push.hipr_object_count ||
                        uint(b) >= push.hipr_object_count) {
                        continue;
                    }
                    float sa = hipr_scores[uint(a)];
                    float sb = hipr_scores[uint(b)];
                    if (sb > sa * swap_threshold) {
                        hipr_order[rank] = b;
                        hipr_order[rank + 1u] = a;
                    }
                }
            }
        }
        return;
    }

    // Stage 0: visibility pass
    // Fire a primary ray for every pixel and record which object it hit.
    // This pass only runs when the camera changes; results are cached in
    // object_id_image for use by subsequent stage-1 dispatches.
    if (push.stage == 0) {
        int visible_id = traceVisibleInstance(makeRay(pixel, dims, float2(0.5f, 0.5f)));
        object_id_image[pixel] = visible_id;
        if (push.selected_mesh_index >= 0 && visible_id >= 0 && uint(visible_id) < push.hipr_object_count) {
            InterlockedAdd(hipr_visible_counts[uint(visible_id)], 1);
        }
        return;
    }

    // Stages 1 and 2: path trace
    Hit center_water_hit = (Hit)0;
    bool center_visible_water = false;
    Ray center_ray = makeRay(pixel, dims, float2(0.5f, 0.5f));
    int visible_object_id = traceVisibleInstance(center_ray, center_visible_water, center_water_hit);
    int previous_visible_object_id = -1;
    if (push.frame > 0u) {
        previous_visible_object_id = object_id_history_image[pixel];
    }
    object_id_image[pixel] = visible_object_id;

    // Determine whether this pixel belongs to the selected object.
    bool is_selected_pixel = (push.selected_mesh_index >= 0) &&
                              (visible_object_id == push.selected_mesh_index);
    bool hipr_vis_post_reveal = false;

    // HiPR visualization mode:
    // Stage 2 is used as a compositing pass that reveals objects in ranked order over time.
    if (push.stage == 2) {
        const uint reveal_period = max(push.hipr_frames_per_object, 1u);
        const uint reveal_rank = (push.hipr_top_k > 0u) ?
                                     min(push.hipr_top_k - 1u, push.frame / reveal_period) :
                                     0u;
        const bool reveal_complete = (push.hipr_top_k == 0u) ||
                                     (reveal_rank >= (push.hipr_top_k - 1u));

        if (!reveal_complete) {
            float3 display = float3(0.0f, 0.0f, 0.0f);
            bool   revealed = false;

            if (push.selected_mesh_index >= 0 && visible_object_id >= 0 && push.hipr_top_k > 0) {
                int rank = (visible_object_id == push.selected_mesh_index) ? 0 :
                           hiprRankForObject(visible_object_id);
                revealed = (rank >= 0) && (uint(rank) <= reveal_rank);
                if (revealed) {
                    display = accum_image[pixel].rgb;
                    if (push.hipr_vis_enable_tint != 0u) {
                        float3 tint_color = hiprInfluenceTintColor(rank);
                        display = lerp(display, tint_color, saturate(push.hipr_vis_tint_strength));
                    }
                }
            }

            if (!revealed) {
                accum_image[pixel] = float4(0.0f, 0.0f, 0.0f, 1.0f);
            }
            if (isSelectionOutline(pixel, dims, visible_object_id)) {
                display = lerp(display, push.outline_color.rgb, saturate(push.outline_color.a));
            }
            output_image[pixel] = float4(display, 1.0f);
            return;
        }

        // Once all ranked objects are revealed, continue with full-scene path tracing
        // in this visualization mode (including unranked objects).
        hipr_vis_post_reveal = true;
    }

    // Stage 1 handles only the selected object's pixels.
    // Stage 2 handles everything else.
    if (push.stage == 1 && !is_selected_pixel) return;
    if (push.stage == 2 && is_selected_pixel && !hipr_vis_post_reveal) return;
    if (push.stage == 4) {
        if (push.hipr_render_rank < 0) return;
        uint rank = min(uint(push.hipr_render_rank), push.hipr_top_k - 1u);
        int target_object_id = hipr_order[rank];
        if (target_object_id < 0 || visible_object_id != target_object_id) return;
    }

    // Path trace this pixel and accumulate into the running average.
    RNG rng = RNG::make(pixel, push.frame);
    float3 colorAccum = float3(0, 0, 0);
    const uint requested_spp = max(push.spp, 1u);
    uint spp = requested_spp;
    const bool water_paused = waterPaused();
    const uint water_spp_override = waterSppOverride();
    const int water_object_id = getWaterSurfaceParams().water_object_id;
    if (water_spp_override > 0u && visible_object_id == water_object_id) {
        spp = max(requested_spp, water_spp_override);
    }
    [loop]
    for (uint i = 0u; i < spp; ++i) {
        uint sample_index = push.frame * spp + i;
        float2 jitter = haltonJitter(sample_index, pixel);
        bool collect_influence = (push.stage == 1) && is_selected_pixel && (push.selected_mesh_index >= 0);
        colorAccum += tracePath(makeRay(pixel, dims, jitter), rng, collect_influence);
    }
    float3 color = sanitizeColor(colorAccum / float(spp), float3(0.0f, 0.0f, 0.0f));

    float3 prev  = sanitizeColor(accum_image[pixel].rgb, color);
    uint accumulation_frame = push.frame;
    float center_water_motion = 0.0f;
    float center_history_mismatch = 0.0f;
    bool used_reprojected_water_history = false;
    float reprojected_history_confidence = 0.0f;

    // Special continuation mode: after per-object focused sampling is complete,
    // stage 1/2 switch to full-scene accumulation. Objects already in the HiPR
    // set keep their prior N-sample history, while never-sampled objects start
    // from the new full-scene frame index.
    if (push.hipr_render_rank == -2) {
        if (push.stage == 1) {
            accumulation_frame = push.hipr_frames_per_object + push.frame;
        } else if (push.stage == 2) {
            int rank = (visible_object_id == push.selected_mesh_index) ? 0 :
                       hiprRankForObject(visible_object_id);
            if (rank >= 0) {
                accumulation_frame = push.hipr_frames_per_object + push.frame;
            }
        }
    }

    float  blend = 1.f / float(accumulation_frame + 1u);
    if (center_visible_water) {
        waterTemporalInfoForHit(center_water_hit, center_water_motion, blend);
    }
    if (center_visible_water && !water_paused && push.frame > 0u &&
        !sceneHasDynamicUpdates()) {
        float3 reprojected_prev = float3(0.0f, 0.0f, 0.0f);
        int reprojected_prev_object_id = -1;
        if (sampleReprojectedWaterHistory(center_water_hit, dims, reprojected_prev,
                                          reprojected_prev_object_id,
                                          reprojected_history_confidence)) {
            prev = sanitizeColor(reprojected_prev, color);
            previous_visible_object_id = reprojected_prev_object_id;
            used_reprojected_water_history = true;
        }
    }
    if (push.frame > 0u && previous_visible_object_id != visible_object_id) {
        const bool previous_was_water = previous_visible_object_id == water_object_id;
        const bool suppress_reset_for_paused_water =
            water_paused && (center_visible_water || previous_was_water);
        if (!suppress_reset_for_paused_water) {
            blend = 1.0f;
        }
    }
    if (push.frame > 0u && center_visible_water) {
        center_history_mismatch = historyMismatch(prev, color);
        color = temporalOutlierClamp(prev, color, center_history_mismatch);
    }
    if (center_visible_water && !water_paused && push.frame > 0u) {
        if (used_reprojected_water_history) {
            float reprojection_guard =
                lerp(0.04f, 0.25f, 1.0f - reprojected_history_confidence);
            blend = max(blend, reprojection_guard);
            blend = max(blend, lerp(0.03f, 0.55f, center_history_mismatch));
        } else {
            blend = max(blend, lerp(0.0f, 0.72f, center_history_mismatch));
        }
    }
    if (center_visible_water && water_paused) {
        blend = 1.f / float(accumulation_frame + 1u);
    }
    if (sceneHasDynamicUpdates() && center_visible_water && push.frame > 0u) {
        // Dynamic floaters alter reflected/transmitted radiance on water even when
        // the water height field changes modestly, so push water history to refresh.
        blend = max(blend, lerp(0.18f, 0.92f, center_history_mismatch));
    }
    if (sceneHasDynamicUpdates() && !center_visible_water && push.frame > 0u &&
        visible_object_id >= 0 && previous_visible_object_id == visible_object_id) {
        int visible_rank = hiprRankForObject(visible_object_id);
        const bool in_dynamic_focus =
            (push.selected_mesh_index < 0) || (visible_rank >= 0);
        if (in_dynamic_focus) {
            float dynamic_history_mismatch = historyMismatch(prev, color);
            color = temporalOutlierClamp(prev, color, dynamic_history_mismatch);
            float dynamic_blend = smoothstep(0.10f, 0.85f, dynamic_history_mismatch);
            blend = max(blend, dynamic_blend);
        }
    }

    float3 accum = sanitizeColor(lerp(prev, color, blend), color);
    accum_image[pixel] = float4(accum, 1.f);

    float3 display = accum;
    if (push.stage == 2 && hipr_vis_post_reveal &&
        push.hipr_vis_enable_tint != 0u && visible_object_id >= 0) {
        int rank = (visible_object_id == push.selected_mesh_index) ? 0 :
                   hiprRankForObject(visible_object_id);
        int tint_rank = (rank >= 0) ? rank : int(push.hipr_top_k);
        float3 tint_color = hiprInfluenceTintColor(tint_rank);
        display = lerp(display, tint_color, saturate(push.hipr_vis_tint_strength));
    }
    if (isSelectionOutline(pixel, dims, visible_object_id)) {
        display = lerp(display, push.outline_color.rgb, saturate(push.outline_color.a));
    }
    output_image[pixel] = float4(display, 1.f);
}
