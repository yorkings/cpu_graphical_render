#pragma once
#include "general.h"

constexpr float c  = 299792458.0f;   // Speed of light (m/s)
constexpr float h  = 6.62606957e-34f;// Planck's constant (J*s)
constexpr float kb = 1.3806488e-23f; // Boltzmann constant (J/K)

constexpr float C1_scaled      = 3.741771852e20f; // C1 scaled for nm (W*nm^4/m^2) spectral exitance
constexpr float C2_nm          = 1.438776877e7f;  // C2 scaled for nm (nm*K)
constexpr float spec_lamda_min = 360.0f;
constexpr float spec_lamda_max = 830.0f;
constexpr float spec_lamda_range = spec_lamda_max - spec_lamda_min;
constexpr float wien_b_nm      = 2897771.955f;    // Wien's displacement constant (nm*K)

inline static smd8 v_zero = _mm256_setzero_ps();
inline static smd8 v_one  = _mm256_set1_ps(1.0f);

inline static const smd8 v_C1_scaled      = _mm256_set1_ps(C1_scaled);
inline static const smd8 v_C2_nm          = _mm256_set1_ps(C2_nm);
inline static const smd8 v_spec_lamda_min = _mm256_set1_ps(spec_lamda_min);
inline static const smd8 v_spec_lamda_max = _mm256_set1_ps(spec_lamda_max);
inline static const smd8 v_spec_lamda_range = _mm256_set1_ps(spec_lamda_range);


namespace avx2_math {
    [[nodiscard]] inline smd8 log2_ps(smd8 x) noexcept {    
       smd8i i = _mm256_castps_si256(x);
       smd8i exp = _mm256_sub_epi32(_mm256_srli_epi32(i, 23), _mm256_set1_epi32(127));
       smd8 e = _mm256_cvtepi32_ps(exp);

       smd8i mant_i = _mm256_or_si256(_mm256_and_si256(i, _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000));
       smd8 m = _mm256_castsi256_ps(mant_i);

       smd8 m1 = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
       smd8 poly = _mm256_set1_ps(0.2418131f);
        poly = _mm256_fmadd_ps(poly, m1, _mm256_set1_ps(-0.3473055f));
        poly = _mm256_fmadd_ps(poly, m1, _mm256_set1_ps(0.4786848f));
        poly = _mm256_fmadd_ps(poly, m1, _mm256_set1_ps(-0.7211658f));
        poly = _mm256_fmadd_ps(poly, m1, _mm256_set1_ps(1.4426896f));
        poly = _mm256_mul_ps(poly, m1);

        return _mm256_add_ps(e, poly);
    }

    [[nodiscard]] inline smd8 exp2_ps(smd8 x) noexcept {
        x = _mm256_max_ps(_mm256_set1_ps(-126.0f), _mm256_min_ps(_mm256_set1_ps(126.0f), x));

       smd8 xi = _mm256_floor_ps(x);
       smd8 f = _mm256_sub_ps(x, xi);
        // Compute 2^floor(x) using exponent bit manipulation
       smd8i exp_i = _mm256_cvtps_epi32(xi);
       smd8i exp_shifted = _mm256_slli_epi32(_mm256_add_epi32(exp_i, _mm256_set1_epi32(127)), 23);
       smd8 pow2_int = _mm256_castsi256_ps(exp_shifted);
        // Minimax polynomial fit for 2^f where f in [0, 1)
       smd8 poly = _mm256_set1_ps(0.00133336f);
        poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(0.00961592f));
        poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(0.05549068f));
        poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(0.24022651f));
        poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(0.69314718f));
        poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(1.0f));

        return _mm256_mul_ps(pow2_int, poly);
    }
    [[nodiscard]] inline smd8 pow_inv_2_4_ps(smd8 x) noexcept {
    const smd8 inv_gamma = _mm256_set1_ps(1.0f / 2.4f);
    return exp2_ps(_mm256_mul_ps(log2_ps(x), inv_gamma));
}
}


inline smd8 simd_exp(smd8 v) {
    smd8 x = _mm256_max_ps(_mm256_set1_ps(-87.33654474f), _mm256_min_ps(v, _mm256_set1_ps(88.37626266f)));

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

// exp(x) - 1 
inline smd8 simd_expm1(smd8 x) {
    smd8 abs_x    = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    smd8 is_small = _mm256_cmp_ps(abs_x, _mm256_set1_ps(1e-4f), _CMP_LT_OQ);
    // Taylor series: x + 0.5 * x^2 + (1/6) * x^3
    smd8 c1_6  = _mm256_set1_ps(0.16666666666666666f);
    smd8 c1_2  = _mm256_set1_ps(0.5f);
    smd8 x2    = _mm256_mul_ps(x, x);
    smd8 small = _mm256_fmadd_ps(_mm256_fmadd_ps(x, c1_6, c1_2), x2, x);
    // exp(x) - 1.0f
    smd8 std_val = _mm256_sub_ps(simd_exp(x), _mm256_set1_ps(1.0f));
    return _mm256_blendv_ps(std_val, small, is_small);
}

inline smd8 sigmoid(smd8 x) {
    smd8 one = _mm256_set1_ps(1.0f);
    smd8 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    smd8 exp_neg_x = simd_exp(neg_x);
    return precise_div(one, _mm256_add_ps(one, exp_neg_x));
}

inline smd8 smooth_sigmoid(smd8 x) {
    smd8 one = _mm256_set1_ps(1.0f);
    smd8 half = _mm256_set1_ps(0.5f);
    smd8 three_halves = _mm256_set1_ps(1.5f);
    smd8 x2 = _mm256_mul_ps(x, x);
    smd8 y  = _mm256_add_ps(one, x2);
    smd8 rsqrt0 = _mm256_rsqrt_ps(y);
    // rsqrt_refined = rsqrt0 * (1.5 - 0.5 * y * rsqrt0^2)
    smd8 half_y    = _mm256_mul_ps(half, y);
    smd8 rsqrt0_sq = _mm256_mul_ps(rsqrt0, rsqrt0);
    smd8 factor    = _mm256_fnmadd_ps(half_y, rsqrt0_sq, three_halves);
    smd8 rsqrt_y   = _mm256_mul_ps(rsqrt0, factor);

    smd8 half_x = _mm256_mul_ps(half, x);
    return _mm256_fmadd_ps(half_x, rsqrt_y, half);
}

inline smd8 simd_log(smd8 v) {
    smd8 zero = _mm256_set1_ps(0.0f);
    smd8 invalid_mask = _mm256_cmp_ps(v, zero, _CMP_LE_OQ);
    smd8 safe_v = _mm256_max_ps(v, _mm256_set1_ps(1.175494351e-38f));
    smd8i xi    = _mm256_castps_si256(safe_v);

    smd8i e_int  = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
    smd8 e_float = _mm256_cvtepi32_ps(e_int);

    smd8i m_int = _mm256_or_si256(_mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000));
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


inline smd8 simd_blackbody_radiance(smd8 wavelength_nm, smd8 temperature_k) {
    smd8 zero = _mm256_set1_ps(0.0f);
    smd8 denom = _mm256_mul_ps(wavelength_nm, temperature_k);
    smd8 arg   = precise_div(v_C2_nm, denom);
    smd8 exp_m1 = simd_expm1(arg);
    smd8 l2 = _mm256_mul_ps(wavelength_nm, wavelength_nm);
    smd8 l4 = _mm256_mul_ps(l2, l2);
    smd8 l5 = _mm256_mul_ps(l4, wavelength_nm);    
    // C1_radiance = C1_scaled / pi = 3.741771852e20 / pi = 1.191042972e20
    smd8 c1_rad = _mm256_set1_ps(1.191042972396e20f);    
    // L = C1_rad / (lambda^5 * (e^(C2 / (lambda * T)) - 1))
    smd8 planck_denom = _mm256_mul_ps(l5, exp_m1);
    smd8 radiance     = precise_div(c1_rad, planck_denom);
    
    smd8 valid_mask = _mm256_and_ps(_mm256_cmp_ps(temperature_k, zero, _CMP_GT_OQ),_mm256_cmp_ps(wavelength_nm, zero, _CMP_GT_OQ));
    smd8 safe_exponent_mask = _mm256_cmp_ps(arg, _mm256_set1_ps(85.0f), _CMP_LT_OQ);
    valid_mask = _mm256_and_ps(valid_mask, safe_exponent_mask);
    return _mm256_blendv_ps(zero, radiance, valid_mask);
}


inline smd8 simd_blackbody_radiance_normalized(smd8 wavelength_nm, smd8 temperature_k) {
    smd8 zero = _mm256_set1_ps(0.0f);
    smd8 wien_b      = _mm256_set1_ps(wien_b_nm);
    smd8 lambda_max  = precise_div(wien_b, temperature_k);

    smd8 L_lambda = simd_blackbody_radiance(wavelength_nm, temperature_k);
    smd8 L_max    = simd_blackbody_radiance(lambda_max, temperature_k);
    smd8 norm_radiance = precise_div(L_lambda, L_max);

    smd8 valid_mask = _mm256_cmp_ps(L_max, zero, _CMP_GT_OQ);
    return _mm256_blendv_ps(zero, norm_radiance, valid_mask);
}

