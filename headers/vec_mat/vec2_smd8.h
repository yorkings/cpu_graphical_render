#pragma once
#include "vec2.h"

struct vec2_x8 {
    smd8 x;
    smd8 y;

    inline vec2_x8() : x(_mm256_setzero_ps()), y(_mm256_setzero_ps()) {}
    inline vec2_x8(float scalar_x, float scalar_y): x(_mm256_set1_ps(scalar_x)), y(_mm256_set1_ps(scalar_y)) {}
    inline vec2_x8(smd8 x, smd8 y) : x(x), y(y) {}

    inline static vec2_x8 load_soa(const float* x_ptr, const float* y_ptr) {
        return vec2_x8(_mm256_loadu_ps(x_ptr), _mm256_loadu_ps(y_ptr));
    }

    inline void store_soa(float* x_ptr, float* y_ptr) const {
        _mm256_storeu_ps(x_ptr, x);
        _mm256_storeu_ps(y_ptr, y);
    }

    inline vec2_x8 operator-() const {
        smd8 sign_bit = _mm256_set1_ps(-0.0f);
        return vec2_x8(_mm256_xor_ps(x, sign_bit), _mm256_xor_ps(y, sign_bit));
    }


    inline vec2_x8& operator+=(const vec2_x8& b) {
        x = _mm256_add_ps(x, b.x);
        y = _mm256_add_ps(y, b.y);
        return *this;
    }

    inline vec2_x8& operator-=(const vec2_x8& b) {
        x = _mm256_sub_ps(x, b.x);
        y = _mm256_sub_ps(y, b.y);
        return *this;
    }

    inline vec2_x8& operator*=(const vec2_x8& b) {
        x = _mm256_mul_ps(x, b.x);
        y = _mm256_mul_ps(y, b.y);
        return *this;
    }

    inline vec2_x8& operator*=(float t) {
        smd8 vt = _mm256_set1_ps(t);
        x = _mm256_mul_ps(x, vt);
        y = _mm256_mul_ps(y, vt);
        return *this;
    }

    inline vec2_x8& operator/=(float t) {
        smd8 inv_t = _mm256_set1_ps(1.0f / t);
        x = _mm256_mul_ps(x, inv_t);
        y = _mm256_mul_ps(y, inv_t);
        return *this;
    }
};


inline vec2_x8 operator+(const vec2_x8& a, const vec2_x8& b) {
    return vec2_x8(_mm256_add_ps(a.x, b.x), _mm256_add_ps(a.y, b.y));
}

inline vec2_x8 operator-(const vec2_x8& a, const vec2_x8& b) {
    return vec2_x8(_mm256_sub_ps(a.x, b.x), _mm256_sub_ps(a.y, b.y));
}

inline vec2_x8 operator*(const vec2_x8& a, const vec2_x8& b) {
    return vec2_x8(_mm256_mul_ps(a.x, b.x), _mm256_mul_ps(a.y, b.y));
}

inline vec2_x8 operator*(const vec2_x8& a, smd8 t) {
    return vec2_x8(_mm256_mul_ps(a.x, t), _mm256_mul_ps(a.y, t));
}

inline vec2_x8 operator*(smd8 t, const vec2_x8& a) {
    return a * t;
}

inline vec2_x8 operator*(const vec2_x8& a, float t) {
    smd8 vt = _mm256_set1_ps(t);
    return vec2_x8(_mm256_mul_ps(a.x, vt), _mm256_mul_ps(a.y, vt));
}

inline vec2_x8 operator*(float t, const vec2_x8& a) {
    return a * t;
}

inline vec2_x8 operator/(const vec2_x8& a, float t) {
    smd8 inv_t = _mm256_set1_ps(1.0f / t);
    return vec2_x8(_mm256_mul_ps(a.x, inv_t), _mm256_mul_ps(a.y, inv_t));
}


inline smd8 dot(const vec2_x8& a, const vec2_x8& b) {
    return _mm256_fmadd_ps(a.x, b.x, _mm256_mul_ps(a.y, b.y));
}

inline smd8 length_squared(const vec2_x8& v) {
    return dot(v, v);
}

inline smd8 length(const vec2_x8& v) {
    return _mm256_sqrt_ps(length_squared(v));
}


inline vec2_x8 lerp(const vec2_x8& a, const vec2_x8& b, smd8 t) {
    return vec2_x8(lerp(a.x,b.x,t),lerp(a.y,b.y,t));
}

inline smd8 cross(const vec2_x8& a, const vec2_x8& b) {
    return _mm256_fmsub_ps(a.x, b.y, _mm256_mul_ps(a.y, b.x));
}

inline vec2_x8 blendv(const vec2_x8& a, const vec2_x8& b, smd8 mask) {
    return vec2_x8(_mm256_blendv_ps(a.x, b.x, mask),_mm256_blendv_ps(a.y, b.y, mask));
}

inline vec2_x8 select(const vec2_x8& condition_false, const vec2_x8& condition_true, smd8 mask) {
    return blendv(condition_false, condition_true, mask);
}

inline vec2_x8 load_aos_vec2x8(const float* ptr) {
    smd8 r0 = _mm256_loadu_ps(ptr + 0); 
    smd8 r1 = _mm256_loadu_ps(ptr + 8);

    smd8 u_lanes = _mm256_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0)); 
    smd8 v_lanes = _mm256_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1)); 

    smd8 u = _mm256_permute2f128_ps(u_lanes, u_lanes, 0x20);    

    smd8 u_final = _mm256_blend_ps(u_lanes, _mm256_permute2f128_ps(u_lanes, u_lanes, 0x01), 0xCC);
    smd8 v_final = _mm256_blend_ps(v_lanes, _mm256_permute2f128_ps(v_lanes, v_lanes, 0x01), 0xCC);

    return vec2_x8(u_final, v_final);
}

inline void store_aos_vec2x8(float* ptr, const vec2_x8& uv) {
    smd8 r0 = _mm256_unpacklo_ps(uv.x, uv.y);
    smd8 r1 = _mm256_unpackhi_ps(uv.x, uv.y);
    smd8 out0 = _mm256_permute2f128_ps(r0, r1, 0x20); 
    smd8 out1 = _mm256_permute2f128_ps(r0, r1, 0x31);
    _mm256_storeu_ps(ptr + 0, out0);
    _mm256_storeu_ps(ptr + 8, out1);
}