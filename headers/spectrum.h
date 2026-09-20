#pragma once
#include "constants.h"




enum class SpectrumTag : uint8_t { Constant = 0, Blackbody, RGBPolynomial, Sampled, Dense, Piecewise, Illuminant, Count};

enum class IlluminantType : uint8_t { D65, D50, A, C, E, F2, F7, F11, Count };


struct ConstantPayload {
    float value;
    float _pad[7];
};

struct BlackbodyPayload {
    float temperature_k;
    float scale;
    float _pad[6];
};

struct RGBPolynomialPayload {
    float c0, c1, c2; 
    float scale;
    float _pad[4];
};

struct SampledPayload {
    const float* values;
    uint32_t num_samples;
    float lambda_min;
    float lambda_max;
    float scale;
    float _pad[3];
};

struct DensePayload {
    const float* values; 
    float scale;
    float _pad[5];
};

struct PiecewisePayload {
    const float* lambdas;
    const float* values;
    uint32_t num_points;
    float scale;
    float _pad[3];
};

struct IlluminantPayload {
    IlluminantType type;
    float scale;
    float _pad[6];
};

struct alignas(32) SampledWavelengths {
    smd8 lambdas;
    smd8 pdfs;

    static inline SampledWavelengths sample_uniform(float u) noexcept {
        SampledWavelengths sw;
        
        float hero_wl = spec_lamda_min + u * spec_lamda_range;
        smd8 v_hero = _mm256_set1_ps(hero_wl);
        static const smd8 v_step_offsets = _mm256_mul_ps(_mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f),
            _mm256_set1_ps(spec_lamda_range * 0.125f));

        smd8 wl = _mm256_add_ps(v_hero, v_step_offsets);

        smd8 over_mask = _mm256_cmp_ps(wl, v_spec_lamda_max, _CMP_GE_OQ);
        smd8 wrapped = _mm256_sub_ps(wl, v_spec_lamda_range);

        sw.lambdas = _mm256_blendv_ps(wl, wrapped, over_mask);
        sw.pdfs    = precise_inv(v_spec_lamda_range);

        return sw;

    }
};

struct alignas(32) Spectrum {
    SpectrumTag tag{SpectrumTag::Constant};
    uint8_t flags{0};
    uint16_t reserved{0};

    union {
        ConstantPayload      constant;
        BlackbodyPayload     blackbody;
        RGBPolynomialPayload rgb_poly;
        SampledPayload       sampled;
        DensePayload         dense;
        PiecewisePayload     piecewise;
        IlluminantPayload    illuminant;
    };
    constexpr Spectrum() noexcept : tag(SpectrumTag::Constant), constant{0.0f, {}} {}
    constexpr Spectrum(float constant_val) noexcept : tag(SpectrumTag::Constant), constant{constant_val, {}} {}
    constexpr Spectrum(float c0, float c1, float c2, float scale = 1.0f) noexcept : tag(SpectrumTag::RGBPolynomial), rgb_poly{c0, c1, c2, scale, {}} {}
    constexpr Spectrum(float temperature_k, float scale, bool is_blackbody) noexcept : tag(SpectrumTag::Blackbody), blackbody{temperature_k, scale, {}} { (void)is_blackbody; }
    constexpr Spectrum(IlluminantType type, float scale = 1.0f) noexcept : tag(SpectrumTag::Illuminant), illuminant{type, scale, {}} {}
   
    constexpr Spectrum(const float* values, uint32_t count, float l_min, float l_max, float scale = 1.0f) noexcept 
        : tag(SpectrumTag::Sampled), sampled{values, count, l_min, l_max, scale, {}} {}

    constexpr Spectrum(const float* values, float scale = 1.0f) noexcept 
        : tag(SpectrumTag::Dense), dense{values, scale, {}} {}

    constexpr Spectrum(const float* lambdas, const float* values, uint32_t count, float scale = 1.0f) noexcept 
        : tag(SpectrumTag::Piecewise), piecewise{lambdas, values, count, scale, {}} {}

    // Static Factory Helpers
    [[nodiscard]] static constexpr Spectrum MakeConstant(float val) noexcept {
        return Spectrum(val);
    }

    [[nodiscard]] static constexpr Spectrum MakeBlackbody(float temp_k, float scale = 1.0f) noexcept {
        return Spectrum(temp_k, scale, true);
    }

    [[nodiscard]] static constexpr Spectrum MakeRGBPoly(float c0, float c1, float c2, float scale = 1.0f) noexcept {
        return Spectrum(c0, c1, c2, scale);
    }

    [[nodiscard]] static constexpr Spectrum MakeIlluminant(IlluminantType type, float scale = 1.0f) noexcept {
        return Spectrum(type, scale);
    }

    [[nodiscard]] static constexpr Spectrum MakeSampled(const float* values, uint32_t count, float l_min, float l_max, float scale = 1.0f) noexcept {
        return Spectrum(values, count, l_min, l_max, scale);
    }

    [[nodiscard]] static constexpr Spectrum MakeDense(const float* values, float scale = 1.0f) noexcept {
        return Spectrum(values, scale);
    }

    [[nodiscard]] static constexpr Spectrum MakePiecewise(const float* lambdas, const float* values, uint32_t count, float scale = 1.0f) noexcept {
        return Spectrum(lambdas, values, count, scale);
    }
};
//verification
static_assert(sizeof(Spectrum) == 32 || sizeof(Spectrum) == 64, "Spectrum container must align to 32/64 byte boundary");
static_assert(std::is_trivially_copyable_v<Spectrum>, "Spectrum must be trivially copyable for SIMD buffer ops");