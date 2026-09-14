#pragma once
#include <immintrin.h>
#include <limits>
#include <string>
#include <memory>
#include <omp.h>
#include<algorithm>
#include <random> 
#include<cmath>
#include<cstdlib>
#include <atomic>
#include<vector>

using std::make_shared;
using std::shared_ptr;

using smd4=__m128;
using smd4i=__m128i;
using smd8=__m256;
using smd8i=__m256i;

constexpr float infinity =std::numeric_limits<float>::infinity();
constexpr float pi = 3.14159265358979323846;
constexpr float invpi = 0.31830988618379067154;
constexpr float inv2pi = 0.15915494309189533577;
constexpr float inv4pi = 0.07957747154594766788;
constexpr float piOver2 = 1.57079632679489661923;
constexpr float piOver4 = 0.78539816339744830961;
constexpr float sqrt2 = 1.41421356237309504880;

std::string get_time_current();

inline float degrees_to_radians(float theta){
    return theta*(pi/180.0f);
}

inline smd8 degrees_to_radians_simd8(smd8 degrees) {    
    return _mm256_mul_ps(degrees, _mm256_set1_ps(pi / 180.0f));
}
inline smd8 radians_to_degrees_simd8(smd8 rad) {    
    return _mm256_mul_ps(rad, _mm256_set1_ps(180.0f/pi));
}


thread_local inline smd8i rng_state_simd8 = _mm256_set_epi32(
    2463534242u, 123456789u, 362436069u, 521288629u,
    886294711u,  991827364u, 152433241u, 661283741u
);


inline smd8 random_float_simd8() {
    smd8i x = rng_state_simd8;
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
    rng_state_simd8 = x;

    smd8i mantissa   = _mm256_and_si256(x, _mm256_set1_epi32(0x007FFFFF));
    smd8i float_bits = _mm256_or_si256(mantissa, _mm256_set1_epi32(0x3F800000));
    smd8 val_1_to_2  = _mm256_castsi256_ps(float_bits);

    return _mm256_sub_ps(val_1_to_2, _mm256_set1_ps(1.0f));
}

inline void random_float_simd8(smd8& out_rand) {
    out_rand = random_float_simd8();
}

inline smd8 random_float_simd8(float min, float max) {
    smd8 rand_0_to_1 = random_float_simd8();
    smd8 range       = _mm256_set1_ps(max - min);
    smd8 min_vec     = _mm256_set1_ps(min);
    return _mm256_fmadd_ps(rand_0_to_1, range, min_vec);
}

inline void seed_rng_simd8_octa(int i, int j, int sample_idx) {
    uint32_t p[8];
    int idx = 0;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 4; ++dx) {
            p[idx++] = ((i + dx) * 1973u + (j + dy) * 9277u + sample_idx * 26699u) | 1u;
        }
    }
    rng_state_simd8 = _mm256_set_epi32(
        p[7] ^ 0x2A6B3C11u, p[6] ^ 0x18B7390Du, p[5] ^ 0x7E31A955u, p[4] ^ 0x3F982C43u,
        p[3] ^ 0x6C078965u, p[2] ^ 0x5D588B65u, p[1] ^ 0x41C64E6Du, p[0] ^ 0x997F4A7Bu
    );
}
inline smd8 lerp(smd8 a, smd8 b, smd8 t) {
    //linear interpolation: lerp(a, b, t) = a + t * (b - a)
    return _mm256_fmadd_ps(t, _mm256_sub_ps(b, a), a);
}




inline smd8 fast_div(smd8 num, smd8 den) {
    smd8 rcp = _mm256_rcp_ps(den);
    smd8 two = _mm256_set1_ps(2.0f);
    smd8 rcp_refined = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(den, rcp, two));
    return _mm256_mul_ps(num, rcp_refined);
}
inline smd8 precise_div(smd8 num, smd8 den) {
    smd8 eps     = _mm256_set1_ps(1e-9f);
    smd8 sign_bit= _mm256_set1_ps(-0.0f);
    smd8 abs_den = _mm256_andnot_ps(sign_bit, den);

    smd8 is_invalid= _mm256_cmp_ps(abs_den, eps, _CMP_NGE_UQ);
    smd8 val_sign  = _mm256_and_ps(den, sign_bit);
    smd8 signed_eps= _mm256_or_ps(eps, val_sign);
    smd8 safe_den  = _mm256_blendv_ps(den, signed_eps, is_invalid);

    // FMA3 Newton-Raphson Refinement for exact single-precision reciprocal
    smd8 x0  = _mm256_rcp_ps(safe_den); 
    smd8 one = _mm256_set1_ps(1.0f);
    smd8 e   = _mm256_fnmadd_ps(safe_den, x0, one);
    smd8 x1  = _mm256_fmadd_ps(x0, e, x0);
    return _mm256_mul_ps(num, x1);
}

inline smd8 precise_inv(smd8 dir) {
    return precise_div(_mm256_set1_ps(1.0f), dir);
}

inline smd8 precise_sqrt(smd8 x) {
    //newton-Raphson refinement for sqrt(x)
    smd8 rcp_sqrt = _mm256_rsqrt_ps(x);
    smd8 half     = _mm256_set1_ps(0.5f);
    smd8 three    = _mm256_set1_ps(3.0f);
    smd8 rcp_sqrt_sq = _mm256_mul_ps(rcp_sqrt, rcp_sqrt);
    smd8 term        = _mm256_fnmadd_ps(x, rcp_sqrt_sq, three);
    smd8 rcp_sqrt_refined = _mm256_mul_ps(_mm256_mul_ps(half, rcp_sqrt), term);
    return _mm256_mul_ps(x, rcp_sqrt_refined);
}

//TRIGONOMETRIc EQUATIONS
inline void sincos_simd8(smd8 x, smd8 &out_sin, smd8 &out_cos) {
    const smd8 inv_half_pi = _mm256_set1_ps(0.63661977236758134308f); 
    const smd8 half_pi_hi  = _mm256_set1_ps(1.570796251296997f);      
    const smd8 half_pi_lo  = _mm256_set1_ps(7.549789954891282e-8f); 

    const smd8 s1 = _mm256_set1_ps(-0.16666666666666666f); 
    const smd8 s2 = _mm256_set1_ps( 0.00833333333333333f); 
    const smd8 s3 = _mm256_set1_ps(-0.000198412698412698f); 

    const smd8 c1 = _mm256_set1_ps(-0.5f);                  
    const smd8 c2 = _mm256_set1_ps( 0.04166666666666666f); 
    const smd8 c3 = _mm256_set1_ps(-0.00138888888888888f); 

    smd8i k_int = _mm256_cvtps_epi32(_mm256_mul_ps(x, inv_half_pi));
    smd8 k      = _mm256_cvtepi32_ps(k_int);

    smd8 y = _mm256_fnmadd_ps(k, half_pi_hi, x);
    y      = _mm256_fnmadd_ps(k, half_pi_lo, y);

    smd8 y2 = _mm256_mul_ps(y, y);
    smd8 y3 = _mm256_mul_ps(y, y2);

    smd8 p_sin = _mm256_fmadd_ps(s3, y2, s2);
    p_sin      = _mm256_fmadd_ps(p_sin, y2, s1);
    smd8 sin_y = _mm256_fmadd_ps(p_sin, y3, y);

    smd8 p_cos = _mm256_fmadd_ps(c3, y2, c2);
    p_cos      = _mm256_fmadd_ps(p_cos, y2, c1);
    smd8 cos_y = _mm256_fmadd_ps(p_cos, y2, _mm256_set1_ps(1.0f));

    smd8 swap_mask = _mm256_castsi256_ps(_mm256_slli_epi32(k_int, 31));
    smd8 sin_val   = _mm256_blendv_ps(sin_y, cos_y, swap_mask);
    smd8 cos_val   = _mm256_blendv_ps(cos_y, sin_y, swap_mask);

    smd8i sin_sign_int = _mm256_slli_epi32(_mm256_and_si256(k_int, _mm256_set1_epi32(2)), 30);
    smd8 sin_sign      = _mm256_castsi256_ps(sin_sign_int);
    smd8 cos_sign      = _mm256_xor_ps(sin_sign, swap_mask);

    out_sin = _mm256_xor_ps(sin_val, sin_sign);
    out_cos = _mm256_xor_ps(cos_val, cos_sign);
}

inline smd8 simd_tan(smd8 x) {
    smd8 s, c;
    sincos_simd8(x, s, c);
    return precise_div(s, c);
}

inline smd8 simd_atan(smd8 x) {
    smd8 sign_mask = _mm256_set1_ps(-0.0f);
    smd8 sign      = _mm256_and_ps(x, sign_mask);
    smd8 abs_x     = _mm256_andnot_ps(sign_mask, x);

    smd8 cmp   = _mm256_cmp_ps(abs_x, _mm256_set1_ps(1.0f), _CMP_GT_OQ);
    smd8 x_inv = precise_inv(abs_x);
    smd8 z     = _mm256_blendv_ps(abs_x, x_inv, cmp);

    smd8 z2   = _mm256_mul_ps(z, z);
    smd8 poly = _mm256_fmadd_ps(z2, _mm256_set1_ps( 0.0805374449f), _mm256_set1_ps(-0.138776856f));
    poly      = _mm256_fmadd_ps(poly, z2, _mm256_set1_ps( 0.1997771084f));
    poly      = _mm256_fmadd_ps(poly, z2, _mm256_set1_ps(-0.333023538f));
    poly      = _mm256_fmadd_ps(poly, z2, _mm256_set1_ps(1.0f));
    smd8 res  = _mm256_mul_ps(z, poly);

    smd8 half_pi = _mm256_set1_ps(1.57079632679f);
    res = _mm256_blendv_ps(res, _mm256_sub_ps(half_pi, res), cmp);

    return _mm256_xor_ps(res, sign);
}

inline smd8 simd_atan2(smd8 y, smd8 x) {
    smd8 pi_vec  = _mm256_set1_ps(pi);
    smd8 ratio   = precise_div(y, x);
    smd8 atan_r  = simd_atan(ratio);

    smd8 x_lt_0  = _mm256_cmp_ps(x, _mm256_set1_ps(0.0f), _CMP_LT_OQ);
    smd8 y_ge_0  = _mm256_cmp_ps(y, _mm256_set1_ps(0.0f), _CMP_GE_OQ);

    smd8 offset  = _mm256_blendv_ps(_mm256_set1_ps(-3.14159265359f), pi_vec, y_ge_0);
    offset       = _mm256_and_ps(offset, x_lt_0);

    return _mm256_add_ps(atan_r, offset);
}

inline smd8 simd_asin(smd8 x) {
    smd8 one     = _mm256_set1_ps(1.0f);
    smd8 neg_one = _mm256_set1_ps(-1.0f);

    smd8 safe_x  = _mm256_min_ps(_mm256_max_ps(x, neg_one), one);
    smd8 x2      = _mm256_mul_ps(safe_x, safe_x);
    smd8 denom   = _mm256_sqrt_ps(_mm256_sub_ps(one, x2));

    smd8 ratio   = precise_div(safe_x, denom);
    return simd_atan(ratio);
}

inline smd8 simd_acos(smd8 x) {
    smd8 half_pi = _mm256_set1_ps(1.57079632679f);
    return _mm256_sub_ps(half_pi, simd_asin(x));
}