#pragma once
#include "vec3_smd8.h"

inline void load_aos_transpose8x8(const float* ptr,smd8& c0, smd8& c1, smd8& c2, smd8& c3,smd8& c4, smd8& c5, smd8& c6, smd8& c7) {
    smd8 r0 = _mm256_loadu_ps(ptr +  0);
    smd8 r1 = _mm256_loadu_ps(ptr +  8);
    smd8 r2 = _mm256_loadu_ps(ptr + 16);
    smd8 r3 = _mm256_loadu_ps(ptr + 24);
    smd8 r4 = _mm256_loadu_ps(ptr + 32);
    smd8 r5 = _mm256_loadu_ps(ptr + 40);
    smd8 r6 = _mm256_loadu_ps(ptr + 48);
    smd8 r7 = _mm256_loadu_ps(ptr + 56);

    smd8 t0 = _mm256_unpacklo_ps(r0, r1);
    smd8 t1 = _mm256_unpackhi_ps(r0, r1);
    smd8 t2 = _mm256_unpacklo_ps(r2, r3);
    smd8 t3 = _mm256_unpackhi_ps(r2, r3);
    smd8 t4 = _mm256_unpacklo_ps(r4, r5);
    smd8 t5 = _mm256_unpackhi_ps(r4, r5);
    smd8 t6 = _mm256_unpacklo_ps(r6, r7);
    smd8 t7 = _mm256_unpackhi_ps(r6, r7);

    smd8 v0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    smd8 v1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    smd8 v2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    smd8 v3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    smd8 v4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    smd8 v5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    smd8 v6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    smd8 v7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    c0 = _mm256_permute2f128_ps(v0, v4, 0x20); // Field 0 across all 8 items
    c1 = _mm256_permute2f128_ps(v1, v5, 0x20); // Field 1
    c2 = _mm256_permute2f128_ps(v2, v6, 0x20); // Field 2
    c3 = _mm256_permute2f128_ps(v3, v7, 0x20); // Field 3
    c4 = _mm256_permute2f128_ps(v0, v4, 0x31); // Field 4
    c5 = _mm256_permute2f128_ps(v1, v5, 0x31); // Field 5
    c6 = _mm256_permute2f128_ps(v2, v6, 0x31); // Field 6
    c7 = _mm256_permute2f128_ps(v3, v7, 0x31); // Field 7
}

inline void load_aos_transpose9x8(const float* ptr, smd8& c0, smd8& c1, smd8& c2, smd8& c3, smd8& c4, smd8& c5, smd8& c6, smd8& c7, smd8& c8) {   
    smd8 r0 = _mm256_loadu_ps(ptr +  0);
    smd8 r1 = _mm256_loadu_ps(ptr +  8);
    smd8 r2 = _mm256_loadu_ps(ptr + 16);
    smd8 r3 = _mm256_loadu_ps(ptr + 24);
    smd8 r4 = _mm256_loadu_ps(ptr + 32);
    smd8 r5 = _mm256_loadu_ps(ptr + 40);
    smd8 r6 = _mm256_loadu_ps(ptr + 48);
    smd8 r7 = _mm256_loadu_ps(ptr + 56);
    smd8 r8 = _mm256_loadu_ps(ptr + 64);

 
    const smd8i idx0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const smd8i idx1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    const smd8i idx2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
    const smd8i idx3 = _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2);
    const smd8i idx4 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
    const smd8i idx5 = _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4);
    const smd8i idx6 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);
    const smd8i idx7 = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);

    c0 = _mm256_blend_ps(_mm256_permutevar8x32_ps(r0, idx0), _mm256_permutevar8x32_ps(r7, idx7), 0x80);
    c1 = _mm256_blend_ps(_mm256_permutevar8x32_ps(r0, idx1), _mm256_permutevar8x32_ps(r8, idx0), 0x80);
    c2 = _mm256_blend_ps(_mm256_permutevar8x32_ps(r0, idx2), _mm256_permutevar8x32_ps(r8, idx1), 0x80);
    c3 = _mm256_blend_ps(_mm256_permutevar8x32_ps(r0, idx3), _mm256_permutevar8x32_ps(r8, idx2), 0x80);
    c4 = _mm256_blend_ps(_mm256_permutevar8x32_ps(r0, idx4), _mm256_permutevar8x32_ps(r8, idx3), 0x80);
    c5 = _mm256_blend_ps(_mm256_permutevar8x32_ps(r0, idx5), _mm256_permutevar8x32_ps(r8, idx4), 0x80);
    c6 = _mm256_blend_ps(_mm256_permutevar8x32_ps(r0, idx6), _mm256_permutevar8x32_ps(r8, idx5), 0x80);
    c7 = _mm256_blend_ps(_mm256_permutevar8x32_ps(r0, idx7), _mm256_permutevar8x32_ps(r8, idx6), 0x80);
    c8 = _mm256_blend_ps(_mm256_permutevar8x32_ps(r1, idx0), _mm256_permutevar8x32_ps(r8, idx7), 0x80);
}

