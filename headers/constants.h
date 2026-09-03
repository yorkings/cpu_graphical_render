#pragma once
#include "general.h"

constexpr float c  = 299792458.0f;   // Speed of light (m/s)
constexpr float h  = 6.62606957e-34f;// Planck's constant (J*s)
constexpr float kb = 1.3806488e-23f; // Boltzmann constant (J/K)

constexpr float C1_SCALED = 2.0f * pi * h * c * c * 1.0e45f; 
constexpr float C2_NM     = (h * c / kb) * 1.0e9f;           

inline smd8 simd_exp(smd8 v) {
    smd8 x = _mm256_max_ps(_mm256_set1_ps(-87.33654474f),_mm256_min_ps(v, _mm256_set1_ps(88.37626266f)));

    smd8 log2e      = _mm256_set1_ps(1.4426950408889634f);
    smd8 magic_bias = _mm256_set1_ps(12582912.0f);
    
    smd8  kd      = _mm256_fmadd_ps(x, log2e, magic_bias);
    smd8i k_int   = _mm256_castps_si256(kd);
    smd8  k_float = _mm256_sub_ps(kd, magic_bias);

    smd8 C1 = _mm256_set1_ps(0.693145751953125f);
    smd8 C2 = _mm256_set1_ps(1.4286068203094172e-6f);
    smd8 r  = _mm256_fnmadd_ps(k_float, C1, x);
    r       = _mm256_fnmadd_ps(k_float, C2, r);

    smd8 r2 = _mm256_mul_ps(r, r);
    
    smd8 c5 = _mm256_set1_ps(0.008333333333333333f);
    smd8 c4 = _mm256_set1_ps(0.041666666666666664f);
    smd8 c3 = _mm256_set1_ps(0.16666666666666666f);
    smd8 c2 = _mm256_set1_ps(0.5f);
    smd8 c1 = _mm256_set1_ps(1.0f);
    smd8 c0 = _mm256_set1_ps(1.0f);

    smd8 p54 = _mm256_fmadd_ps(c5, r, c4);
    smd8 p32 = _mm256_fmadd_ps(c3, r, c2);
    smd8 p10 = _mm256_fmadd_ps(c1, r, c0);

    smd8 p52 = _mm256_fmadd_ps(p54, r2, p32);
    smd8 p   = _mm256_fmadd_ps(p52, r2, p10);

    smd8i pow2k_int = _mm256_slli_epi32(_mm256_add_epi32(k_int, _mm256_set1_epi32(127)), 23);
    smd8  pow2k     = _mm256_castsi256_ps(pow2k_int);

    return _mm256_mul_ps(p, pow2k);
}

inline smd8 simd_log(smd8 v) {
    smd8 zero = _mm256_set1_ps(0.0f);
    smd8 invalid_mask = _mm256_cmp_ps(v, zero, _CMP_LE_OQ);

    smd8 safe_v = _mm256_max_ps(v, _mm256_set1_ps(1.175494351e-38f));
    smd8i xi    = _mm256_castps_si256(safe_v);

    smd8i e_int  = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
    smd8 e_float = _mm256_cvtepi32_ps(e_int);

    smd8i m_int = _mm256_or_si256(_mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF)),_mm256_set1_epi32(0x3F800000));
    smd8 m      = _mm256_castsi256_ps(m_int);

    smd8 mask    = _mm256_cmp_ps(m, _mm256_set1_ps(1.41421356237f), _CMP_GT_OQ);
    m            = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), mask);
    e_float      = _mm256_blendv_ps(e_float, _mm256_add_ps(e_float, _mm256_set1_ps(1.0f)), mask);

    smd8 u  = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
    smd8 u2 = _mm256_mul_ps(u, u);

    smd8 c5 = _mm256_set1_ps( 0.1536257077f);
    smd8 c4 = _mm256_set1_ps(-0.2192938883f);
    smd8 c3 = _mm256_set1_ps( 0.3211822365f);
    smd8 c2 = _mm256_set1_ps(-0.4998741238f);
    smd8 c1 = _mm256_set1_ps( 0.9999964239f);

    smd8 p54 = _mm256_fmadd_ps(c5, u, c4);
    smd8 p32 = _mm256_fmadd_ps(c3, u, c2);
    
    smd8 p52 = _mm256_fmadd_ps(p54, u2, p32);
    smd8 p   = _mm256_fmadd_ps(p52, u2, c1);
    p        = _mm256_mul_ps(p, u);

    smd8 ln2 = _mm256_set1_ps(0.69314718056f);
    smd8 res = _mm256_fmadd_ps(e_float, ln2, p);

    smd8 nan = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
    return _mm256_blendv_ps(res, nan, invalid_mask);
}