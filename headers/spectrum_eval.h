#pragma once
#include "spectrum.h"
#include "constants.h"
#include "color.h"

namespace avx2_spectrum {
    [[nodiscard]] inline smd8 eval_rgb_poly_smd8(const RGBPolynomialPayload& poly, smd8 lambdas_nm) noexcept {
        smd8 c0 = _mm256_set1_ps(poly.c0);
        smd8 c1 = _mm256_set1_ps(poly.c1);
        smd8 c2 = _mm256_set1_ps(poly.c2);
        // Horner's scheme for c0 + c1*λ + c2*λ^2:
        smd8 s = _mm256_fmadd_ps(c2, lambdas_nm, c1);
        s = _mm256_fmadd_ps(s, lambdas_nm, c0);
        
        return smooth_sigmoid(s);
    }

    [[nodiscard]] inline smd8 eval_blackbody_smd8(const BlackbodyPayload& bb, smd8 lambdas_nm) noexcept {
        smd8 temp = _mm256_set1_ps(bb.temperature_k);
        smd8 scale = _mm256_set1_ps(bb.scale);
        smd8 radiance = simd_blackbody_radiance(lambdas_nm, temp);
        return _mm256_mul_ps(radiance, scale);
    }
    
    [[nodiscard]] inline smd8 eval_dense_smd8(const DensePayload& dense, smd8 lambdas_nm) noexcept {
        if (!dense.values) return _mm256_setzero_ps();

        smd8 wl = _mm256_max_ps(v_spec_lamda_min, _mm256_min_ps(v_spec_lamda_max, lambdas_nm));
        smd8 offset = _mm256_sub_ps(wl, v_spec_lamda_min);

        smd8i idx0 = _mm256_cvttps_epi32(offset);
        smd8 idx_f = _mm256_cvtepi32_ps(idx0);
        smd8 t = _mm256_sub_ps(offset, idx_f);

        smd8i idx1 = _mm256_add_epi32(idx0, _mm256_set1_epi32(1));
        smd8 v0 = _mm256_i32gather_ps(dense.values, idx0, 4);
        smd8 v1 = _mm256_i32gather_ps(dense.values, idx1, 4);

        smd8 interpolated = _mm256_fmadd_ps(t, _mm256_sub_ps(v1, v0), v0);
        return _mm256_mul_ps(interpolated, _mm256_set1_ps(dense.scale));
    }

    [[nodiscard]] inline smd8 eval_sampled_smd8(const SampledPayload& sampled, smd8 lambdas_nm) noexcept {
        if (!sampled.values || sampled.num_samples < 2) return _mm256_setzero_ps();

        smd8 l_min = _mm256_set1_ps(sampled.lambda_min);
        smd8 l_max = _mm256_set1_ps(sampled.lambda_max);
        smd8 wl = _mm256_max_ps(l_min, _mm256_min_ps(l_max, lambdas_nm));

        smd8 range = _mm256_sub_ps(l_max, l_min);
        smd8 step_inv = _mm256_set1_ps((float)(sampled.num_samples - 1));
        smd8 norm_offset = precise_div(_mm256_sub_ps(wl, l_min), range);
        smd8 index_f_scaled = _mm256_mul_ps(norm_offset, step_inv);

        smd8i idx0 = _mm256_cvttps_epi32(index_f_scaled);
        smd8 idx_f = _mm256_cvtepi32_ps(idx0);
        smd8 t = _mm256_sub_ps(index_f_scaled, idx_f);

        smd8i max_idx = _mm256_set1_epi32(sampled.num_samples - 1);
        smd8i idx1 = _mm256_min_epi32(_mm256_add_epi32(idx0, _mm256_set1_epi32(1)), max_idx);

        smd8 v0 = _mm256_i32gather_ps(sampled.values, idx0, 4);
        smd8 v1 = _mm256_i32gather_ps(sampled.values, idx1, 4);

        smd8 interpolated = _mm256_fmadd_ps(t, _mm256_sub_ps(v1, v0), v0);
        return _mm256_mul_ps(interpolated, _mm256_set1_ps(sampled.scale));
    }

    [[nodiscard]] inline smd8 eval_piecewise_smd8(const PiecewisePayload& pw, smd8 lambdas_nm) noexcept {
        if (!pw.lambdas || !pw.values || pw.num_points < 2) return _mm256_setzero_ps();

        smd8 v0 = _mm256_set1_ps(pw.values[0]);
        smd8 v1 = _mm256_set1_ps(pw.values[pw.num_points - 1]);

        smd8 result = _mm256_setzero_ps();
        smd8 scale = _mm256_set1_ps(pw.scale);

        smd8 min_mask = _mm256_cmp_ps(lambdas_nm, _mm256_set1_ps(pw.lambdas[0]), _CMP_LE_OQ);
        smd8 max_mask = _mm256_cmp_ps(lambdas_nm, _mm256_set1_ps(pw.lambdas[pw.num_points - 1]), _CMP_GE_OQ);

        result = _mm256_blendv_ps(result, _mm256_mul_ps(v0, scale), min_mask);
        result = _mm256_blendv_ps(result, _mm256_mul_ps(v1, scale), max_mask);

        for (uint32_t i = 0; i < pw.num_points - 1; ++i) {
            smd8 l0 = _mm256_set1_ps(pw.lambdas[i]);
            smd8 l1 = _mm256_set1_ps(pw.lambdas[i + 1]);
            smd8 ge_l0 = _mm256_cmp_ps(lambdas_nm, l0, _CMP_GE_OQ);
            smd8 lt_l1 = _mm256_cmp_ps(lambdas_nm, l1, _CMP_LT_OQ);
            smd8 in_range = _mm256_and_ps(ge_l0, lt_l1);

            if (_mm256_movemask_ps(in_range) != 0) {
                smd8 val0 = _mm256_set1_ps(pw.values[i]);
                smd8 val1 = _mm256_set1_ps(pw.values[i + 1]);
                smd8 t = precise_div(_mm256_sub_ps(lambdas_nm, l0), _mm256_sub_ps(l1, l0));
                smd8 interp = _mm256_fmadd_ps(t, _mm256_sub_ps(val1, val0), val0);
                smd8 scaled_interp = _mm256_mul_ps(interp, scale);
                result = _mm256_blendv_ps(result, scaled_interp, in_range);
            }
        }

        return result;
    }
    [[nodiscard]] inline smd8 eval_illuminant_smd8(const IlluminantPayload& illum, smd8 lambdas_nm) noexcept {
        SampledPayload d65_payload{
            ILLUMINANT_D65_LUT,
            48,
            360.0f,
            830.0f,
            illum.scale,
            {}
        };
        return eval_sampled_smd8(d65_payload, lambdas_nm);
    }
    [[nodiscard]] inline smd8 eval_spectrum_smd8(const Spectrum& spec, const SampledWavelengths& sw) noexcept {
        switch (spec.tag) {
            case SpectrumTag::Constant:
                return _mm256_set1_ps(spec.constant.value);

            case SpectrumTag::Blackbody:
                return eval_blackbody_smd8(spec.blackbody, sw.lambdas);

            case SpectrumTag::RGBPolynomial:
                return eval_rgb_poly_smd8(spec.rgb_poly, sw.lambdas);

            case SpectrumTag::Sampled:
                return eval_sampled_smd8(spec.sampled, sw.lambdas);

            case SpectrumTag::Dense:
                return eval_dense_smd8(spec.dense, sw.lambdas);

            case SpectrumTag::Piecewise:
                return eval_piecewise_smd8(spec.piecewise, sw.lambdas);

            case SpectrumTag::Illuminant:
                return eval_illuminant_smd8(spec.illuminant, sw.lambdas);

            default:
                return _mm256_setzero_ps();
        }
    }


}