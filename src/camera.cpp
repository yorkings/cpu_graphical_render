// void camera::render(hit_list& world, std::vector<uint32_t>& screen_pixels) {
//     initialize();
//     if (screen_pixels.size() != static_cast<size_t>(image_width * image_height)) {
//         screen_pixels.resize(image_width * image_height);
//     }
//     std::mdspan pixels_2d(screen_pixels.data(), image_height, image_width);

//     const int blocks_x = (image_width + 1) / 2;
//     const int blocks_y = (image_height + 1) / 2;
//     const int total_blocks = blocks_x * blocks_y;

//     std::atomic<int> blocks_completed(0);

//     // LOOP 1: Parallel 1D loop over all 2x2 pixel quads
//     #pragma omp parallel for schedule(dynamic, 16)
//     for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
//         const int block_y = (block_idx / blocks_x) * 2;
//         const int block_x = (block_idx % blocks_x) * 2;

//         seed_rng_simd4_quad(block_x, block_y, 0);

//         vec4 accum_r = _mm_setzero_ps();
//         vec4 accum_g = _mm_setzero_ps();
//         vec4 accum_b = _mm_setzero_ps();

//         // LOOP 2: Flattened 1D sample loop over total SPP
//         for (int s = 0; s < samples_per_pixel; ++s) {
//             const int s_i = s % spp_sqrt;
//             const int s_j = s / spp_sqrt;

//             Raypackets pack = get_ray_packs(block_x, block_y, s_i, s_j);
//             vec4 pack_r, pack_g, pack_b;
//             ray_color_packet_simd(pack, world, max_depth, pack_r, pack_g, pack_b);

//             accum_r = _mm_add_ps(accum_r, pack_r);
//             accum_g = _mm_add_ps(accum_g, pack_g);
//             accum_b = _mm_add_ps(accum_b, pack_b);
//         }

//         accum_r = sanitize_simd4(accum_r);
//         accum_g = sanitize_simd4(accum_g);
//         accum_b = sanitize_simd4(accum_b);

//         uint32_t packed_pixels[4];
//         write_color_simd4(accum_r, accum_g, accum_b, pixel_sample_scale, packed_pixels);

//         pixels_2d[block_y, block_x] = packed_pixels[0];
//         if (block_x + 1 < image_width) 
//             pixels_2d[block_y, block_x + 1] = packed_pixels[1];
//         if (block_y + 1 < image_height) 
//             pixels_2d[block_y + 1, block_x] = packed_pixels[2];
//         if (block_x + 1 < image_width && block_y + 1 < image_height) 
//             pixels_2d[block_y + 1, block_x + 1] = packed_pixels[3];

//         // Lockless, low-overhead progress logging
//         int completed = blocks_completed.fetch_add(1, std::memory_order_relaxed) + 1;
//         if ((completed & 0x7F) == 0 && omp_get_thread_num() == 0) {
//             int percent = (completed * 100) / total_blocks;
//             std::cerr << "\rRendering: " << percent << "% [" << completed << "/" << total_blocks << " blocks]" << std::flush;
//         }
//     }
//     std::cerr << "\n\nRendering complete.\n";
// }

// void ray_color_packet_simd(
//     Raypackets& pack, 
//     const hit_list& world, 
//     int max_depth, 
//     __m128& out_r, __m128& out_g, __m128& out_b,
//     uint32_t* rng_state
// ) {
//     __m128 accum_r = _mm_setzero_ps();
//     __m128 accum_g = _mm_setzero_ps();
//     __m128 accum_b = _mm_setzero_ps();

//     __m128 tp_r = _mm_set1_ps(1.0f);
//     __m128 tp_g = _mm_set1_ps(1.0f);
//     __m128 tp_b = _mm_set1_ps(1.0f);

//     // Track active lanes (-1.0f float bitmask for active, 0.0f for inactive)
//     __m128 active_mask = _mm_castsi128_ps(_mm_set1_epi32(-1));

//     for (int depth = 0; depth < max_depth; ++depth) {
//         // Early exit if all 4 rays in the packet have terminated
//         if (_mm_movemask_ps(active_mask) == 0) break;

//         hit_record_simd rec;
//         __m128 hit_mask = world.hit_simd(pack, _mm_set1_ps(0.001f), _mm_set1_ps(1e30f), rec);
//         active_mask = _mm_and_ps(active_mask, hit_mask);

//         if (_mm_movemask_ps(active_mask) == 0) {
//             // Accumulate background/sky light for surviving rays that missed
//             accum_r = _mm_add_ps(accum_r, _mm_mul_ps(tp_r, get_sky_r(pack)));
//             accum_g = _mm_add_ps(accum_g, _mm_mul_ps(tp_g, get_sky_g(pack)));
//             accum_b = _mm_add_ps(accum_b, _mm_mul_ps(tp_b, get_sky_b(pack)));
//             break;
//         }

//         // Accumulate surface emission
//         accum_r = _mm_add_ps(accum_r, _mm_and_ps(active_mask, _mm_mul_ps(tp_r, rec.mat.emit_r)));
//         accum_g = _mm_add_ps(accum_g, _mm_and_ps(active_mask, _mm_mul_ps(tp_g, rec.mat.emit_g)));
//         accum_b = _mm_add_ps(accum_b, _mm_and_ps(active_mask, _mm_mul_ps(tp_b, rec.mat.emit_b)));

//         // Evaluate scattering (modifies pack rays and updates tp_r, tp_g, tp_b)
//         scatter_simd(rec, pack, tp_r, tp_g, tp_b, rng_state);

//         // --- Russian Roulette Path Termination (Start after bounce 3) ---
//         if (depth >= 3) {
//             // 1. Max color component determines ray importance per lane
//             __m128 max_tp = _mm_max_ps(tp_r, _mm_max_ps(tp_g, tp_b));

//             // 2. Clamp probability to [0.05, 0.95] to prevent division-by-zero or infinite loops
//             __m128 p_continue = _mm_min_ps(max_tp, _mm_set1_ps(0.95f));
//             p_continue = _mm_max_ps(p_continue, _mm_set1_ps(0.05f));

//             // 3. Generate uniform random floats in [0, 1) for all 4 lanes
//             __m128 xi = simd_random_float4(rng_state);

//             // 4. Create survival mask (xi < p_continue)
//             __m128 survive_mask = _mm_cmplt_ps(xi, p_continue);
//             active_mask = _mm_and_ps(active_mask, survive_mask);

//             // 5. Unbias throughput for surviving rays: throughput *= (1.0 / p_continue)
//             __m128 inv_p = _mm_div_ps(_mm_set1_ps(1.0f), p_continue);
//             tp_r = _mm_mul_ps(tp_r, inv_p);
//             tp_g = _mm_mul_ps(tp_g, inv_p);
//             tp_b = _mm_mul_ps(tp_b, inv_p);
//         }
//     }

//     out_r = accum_r;
//     out_g = accum_g;
//     out_b = accum_b;
// }