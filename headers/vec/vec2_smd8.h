#pragma once
#include "general.h"

struct vec2_smd8 {
    smd8 x;
    smd8 y;

    inline vec2_smd8() : x(_mm256_setzero_ps()), y(_mm256_setzero_ps()) {}
    inline vec2_smd8(float scalar_x, float scalar_y): x(_mm256_set1_ps(scalar_x)), y(_mm256_set1_ps(scalar_y)) {}
    inline vec2_smd8(smd8 x, smd8 y) : x(x), y(y) {}

    inline static vec2_smd8 load_soa(const float* x_ptr, const float* y_ptr) {
        return vec2_smd8(_mm256_loadu_ps(x_ptr), _mm256_loadu_ps(y_ptr));
    }

    inline void store_soa(float* x_ptr, float* y_ptr) const {
        _mm256_storeu_ps(x_ptr, x);
        _mm256_storeu_ps(y_ptr, y);
    }

    inline smd8 u() const { return x; }
    inline smd8 v() const { return y; }

    inline vec2_smd8 operator-() const {
        smd8 sign_bit = _mm256_set1_ps(-0.0f);
        return vec2_smd8(_mm256_xor_ps(x, sign_bit), _mm256_xor_ps(y, sign_bit));
    }


    inline vec2_smd8& operator+=(const vec2_smd8& b) {
        x = _mm256_add_ps(x, b.x);
        y = _mm256_add_ps(y, b.y);
        return *this;
    }

    inline vec2_smd8& operator-=(const vec2_smd8& b) {
        x = _mm256_sub_ps(x, b.x);
        y = _mm256_sub_ps(y, b.y);
        return *this;
    }

    inline vec2_smd8& operator*=(const vec2_smd8& b) {
        x = _mm256_mul_ps(x, b.x);
        y = _mm256_mul_ps(y, b.y);
        return *this;
    }

    inline vec2_smd8& operator*=(float t) {
        smd8 vt = _mm256_set1_ps(t);
        x = _mm256_mul_ps(x, vt);
        y = _mm256_mul_ps(y, vt);
        return *this;
    }

    inline vec2_smd8& operator/=(float t) {
        smd8 inv_t = _mm256_set1_ps(1.0f / t);
        x = _mm256_mul_ps(x, inv_t);
        y = _mm256_mul_ps(y, inv_t);
        return *this;
    }
};


inline vec2_smd8 operator+(const vec2_smd8& a, const vec2_smd8& b) {
    return vec2_smd8(_mm256_add_ps(a.x, b.x), _mm256_add_ps(a.y, b.y));
}

inline vec2_smd8 operator-(const vec2_smd8& a, const vec2_smd8& b) {
    return vec2_smd8(_mm256_sub_ps(a.x, b.x), _mm256_sub_ps(a.y, b.y));
}

inline vec2_smd8 operator*(const vec2_smd8& a, const vec2_smd8& b) {
    return vec2_smd8(_mm256_mul_ps(a.x, b.x), _mm256_mul_ps(a.y, b.y));
}

inline vec2_smd8 operator*(const vec2_smd8& a, smd8 t) {
    return vec2_smd8(_mm256_mul_ps(a.x, t), _mm256_mul_ps(a.y, t));
}

inline vec2_smd8 operator*(smd8 t, const vec2_smd8& a) {
    return a * t;
}

inline vec2_smd8 operator*(const vec2_smd8& a, float t) {
    smd8 vt = _mm256_set1_ps(t);
    return vec2_smd8(_mm256_mul_ps(a.x, vt), _mm256_mul_ps(a.y, vt));
}

inline vec2_smd8 operator*(float t, const vec2_smd8& a) {
    return a * t;
}

inline vec2_smd8 operator/(const vec2_smd8& a, float t) {
    smd8 inv_t = _mm256_set1_ps(1.0f / t);
    return vec2_smd8(_mm256_mul_ps(a.x, inv_t), _mm256_mul_ps(a.y, inv_t));
}


inline smd8 dot(const vec2_smd8& a, const vec2_smd8& b) {
    return _mm256_fmadd_ps(a.x, b.x, _mm256_mul_ps(a.y, b.y));
}

inline smd8 length_squared(const vec2_smd8& v) {
    return dot(v, v);
}

inline smd8 length(const vec2_smd8& v) {
    return _mm256_sqrt_ps(length_squared(v));
}

// Linear Interpolation: lerp(a, b, t) = a + t * (b - a) 
inline vec2_smd8 lerp(const vec2_smd8& a, const vec2_smd8& b, smd8 t) {
    return vec2_smd8(_mm256_fmadd_ps(t, _mm256_sub_ps(b.x, a.x), a.x),_mm256_fmadd_ps(t, _mm256_sub_ps(b.y, a.y), a.y));
}

inline smd8 cross(const vec2_smd8& a, const vec2_smd8& b) {
    return _mm256_fmsub_ps(a.x, b.y, _mm256_mul_ps(a.y, b.x));
}