#pragma once
#include <cstddef>
#include "constants.h"
#include "vec3_smd8.h"
#include "spectrum.h"

using XYZ8 = vec8;
using RGB8 = vec8;

constexpr float  CIE_MIN_WAVELENGTH = 360.0f;
constexpr float  CIE_MAX_WAVELENGTH = 830.0f;
constexpr size_t CIE_LUT_SIZE       = 480;
inline constexpr float CIE_STEP_NM = 1.0f;
inline constexpr float CIE_INV_STEP = 1.0f / CIE_STEP_NM;


alignas(32) extern const float CIE2012_X_LUT[CIE_LUT_SIZE];
alignas(32) extern const float CIE2012_Y_LUT[CIE_LUT_SIZE];
alignas(32) extern const float CIE2012_Z_LUT[CIE_LUT_SIZE];
alignas(32) extern const float ILLUMINANT_D65_LUT[48];

inline XYZ8 sample_cie2012(smd8 wavelengths_nm) noexcept {
    smd8 wl=_mm256_max_ps(v_spec_lamda_min, _mm256_min_ps(v_spec_lamda_max, wavelengths_nm));
    smd8 offset = _mm256_sub_ps(wl, v_spec_lamda_min);
    offset = _mm256_min_ps(offset, v_spec_lamda_range);

    smd8i idx_i = _mm256_cvttps_epi32(offset);
    smd8  idx_f = _mm256_cvtepi32_ps(idx_i);
    smd8  t     = _mm256_sub_ps(offset, idx_f);

    smd8i idx0 = idx_i;
    smd8i idx1 = _mm256_add_epi32(idx_i, _mm256_set1_epi32(1));
    // Gather LUT values
    vec8 vidx0(_mm256_i32gather_ps(CIE2012_X_LUT, idx0, 4),_mm256_i32gather_ps(CIE2012_Y_LUT, idx0, 4),_mm256_i32gather_ps(CIE2012_Z_LUT, idx0, 4));
    vec8 vidx1(_mm256_i32gather_ps(CIE2012_X_LUT, idx1, 4),_mm256_i32gather_ps(CIE2012_Y_LUT, idx1, 4),_mm256_i32gather_ps(CIE2012_Z_LUT, idx1, 4));

    return lerp(vidx0, vidx1, t);
}

[[nodiscard]] inline RGB8 transform_xyz(XYZ8 xyz,vec8 m0,vec8 m1,vec8 m2) noexcept {
    return RGB8(dot(m0, xyz), dot(m1, xyz), dot(m2, xyz)); 
}

[[nodiscard]] inline RGB8 xyz_to_linear_srgb(XYZ8 xyz) noexcept {
    vec8 m0(3.2404542f, -1.5371385f, -0.4985314f);
    vec8 m1(-0.9692660f, 1.8760108f, 0.0415560f);
    vec8 m2(0.0556434f, -0.2040259f, 1.0569715f);
    return transform_xyz(xyz, m0, m1, m2);
}

[[nodiscard]] inline RGB8 linear_srgb_to_xyz(RGB8 rgb) noexcept {
    vec8 m0(0.4124564f, 0.3575760f, 0.1804375f);
    vec8 m1(0.2126729f, 0.7151522f, 0.0721750f);
    vec8 m2(0.0193339f, 0.1191920f, 0.9503041f);
    return transform_xyz(rgb, m0, m1, m2);
}

[[nodiscard]] inline RGB8 xyz_to_acescg(XYZ8 xyz) noexcept {
    vec8 m0(2.6955079f, -1.2763466f, -0.4731216f);
    vec8 m1(-0.8261224f, 1.7519993f, 0.0741231f);
    vec8 m2(0.1316785f, -0.2580313f, 1.1263527f);
    return transform_xyz(xyz, m0, m1, m2);
}

[[nodiscard]] inline RGB8 acescg_to_xyz(RGB8 rgb) noexcept {
    vec8 m0(0.6624542f, 0.1340049f, 0.1561875f);
    vec8 m1(0.2722287f, 0.6740816f, 0.0536897f);
    vec8 m2(-0.0055745f, 0.0040604f, 1.0103392f);
    return transform_xyz(rgb, m0, m1, m2);
}

[[nodiscard]] inline RGB8 clamp_rgb(RGB8 rgb) noexcept {
    return RGB8(_mm256_max_ps(v_zero, _mm256_min_ps(v_one, rgb.x)),
     _mm256_max_ps(v_zero, _mm256_min_ps(v_one, rgb.y)), 
     _mm256_max_ps(v_zero, _mm256_min_ps(v_one, rgb.z)));
}

[[nodiscard]] inline smd8 linear_to_srgb_channel(smd8 linear)noexcept{
    linear = _mm256_max_ps(v_zero, linear);
    smd8 threshold = _mm256_set1_ps(0.0031308f);
    smd8 mask = _mm256_cmp_ps(linear,threshold, _CMP_LT_OS);
    smd8 branch1 = _mm256_mul_ps(linear, _mm256_set1_ps(12.92f));

    smd8 x_safe = _mm256_max_ps(linear, _mm256_set1_ps(1e-10f));
    smd8 x_pow = avx2_math::pow_inv_2_4_ps(x_safe);
    smd8 gamma_part = _mm256_fmadd_ps(_mm256_set1_ps(1.055f), x_pow, _mm256_set1_ps(-0.055f));
    
    return _mm256_blendv_ps(gamma_part, branch1, mask);
    
}
[[nodiscard]] inline smd8 linear_to_srgb_channel_fast(smd8 x) noexcept {
    x = _mm256_max_ps(x, _mm256_setzero_ps());
    smd8 s = precise_sqrt(precise_sqrt(x));
    // P(s) ≈ 1.055 * s^(5/3) - 0.055
    smd8 poly = avx2_math::evaluate_poly3(s,_mm256_set1_ps(-0.1328f),_mm256_set1_ps( 0.9465f),_mm256_set1_ps( 0.2581f),_mm256_set1_ps(-0.0720f));

    smd8 linear_part = _mm256_mul_ps(x, _mm256_set1_ps(12.92f));
    smd8 mask = _mm256_cmp_ps(x, _mm256_set1_ps(0.0031308f), _CMP_LE_OS);
    return _mm256_blendv_ps(poly, linear_part, mask);
}

[[nodiscard]] inline RGB8 linear_to_srgb_avx2(RGB8 linear_rgb) noexcept {
    return RGB8(linear_to_srgb_channel(linear_rgb.x), linear_to_srgb_channel(linear_rgb.y), linear_to_srgb_channel(linear_rgb.z));
}


[[nodiscard]] inline XYZ8 integrate_spectrum_to_xyz(smd8 L_spec, const SampledWavelengths& sw) noexcept {
    XYZ8 cmf = sample_cie2012(sw.lambdas);

    smd8 weighted_radiance =precise_div(L_spec, sw.pdfs);
    vec8 xyz = vec8(_mm256_mul_ps(weighted_radiance, cmf.x), _mm256_mul_ps(weighted_radiance, cmf.y), _mm256_mul_ps(weighted_radiance, cmf.z));
    smd8 inv_n = _mm256_set1_ps(0.125f); // 1/8 for 8 samples

    return XYZ8(_mm256_mul_ps(xyz.x, inv_n),_mm256_mul_ps(xyz.y, inv_n),_mm256_mul_ps(xyz.z, inv_n));
    
}