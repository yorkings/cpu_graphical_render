#pragma once
#include "general.h"

class vec3 {
    public:
        smd4 v;

        inline vec3() : v(_mm_setzero_ps()) {}
        inline vec3(float x, float y, float z) : v(_mm_set_ps(0.0f, z, y, x)) {}
        inline vec3(smd4 v) : v(v) {}

        inline float x() const { return _mm_cvtss_f32(v); }
        inline float y() const { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1))); }
        inline float z() const { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2))); }

        inline float operator[](int i) const { 
            switch (i) {
                case 0: return _mm_cvtss_f32(v);
                case 1: return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)));
                case 2: return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)));
                default: return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)));
            }
        }

        inline vec3 operator-() const { return vec3(_mm_xor_ps(v, _mm_set1_ps(-0.0f)));}

        inline vec3& operator+=(const vec3& b) { 
            v = _mm_add_ps(v, b.v); 
            return *this; 
        }

        inline vec3& operator*=(const float d) {
            v = _mm_mul_ps(v, _mm_set1_ps(d));
            return *this;
        }

        inline vec3& operator/=(const float d) { 
            return *this *= (1.0f / d); 
        }

        inline float length_squared() const {
            smd4 dot_res = _mm_dp_ps(v, v, 0x71); 
            return _mm_cvtss_f32(dot_res);
        }

        inline float length() const {
            return std::sqrt(length_squared());
        }
};

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(_mm_add_ps(u.v, v.v));
}

inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(_mm_sub_ps(u.v, v.v));
}

inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(_mm_mul_ps(u.v, v.v)); 
}

inline vec3 operator*(float t, const vec3& v) {
    return vec3(_mm_mul_ps(_mm_set1_ps(t), v.v));
}

inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

inline vec3 operator/(const vec3& v, float t) {
    return (1.0f / t) * v;
}

inline float dot(const vec3& u, const vec3& v) {
    smd4 dot_res = _mm_dp_ps(u.v, v.v, 0x71);
    return _mm_cvtss_f32(dot_res);
}


inline vec3 cross(const vec3& a, const vec3& b) {
    smd4 a_yzx = _mm_shuffle_ps(a.v, a.v, _MM_SHUFFLE(3, 0, 2, 1));
    smd4 b_zxy = _mm_shuffle_ps(b.v, b.v, _MM_SHUFFLE(3, 1, 0, 2));
    smd4 a_zxy = _mm_shuffle_ps(a.v, a.v, _MM_SHUFFLE(3, 1, 0, 2));
    smd4 b_yzx = _mm_shuffle_ps(b.v, b.v, _MM_SHUFFLE(3, 0, 2, 1));
    //(a_yzx * b_zxy) - (a_zxy * b_yzx) 
    return vec3(_mm_fmsub_ps(a_yzx, b_zxy, _mm_mul_ps(a_zxy, b_yzx)));
}

inline vec3 unit_vector(const vec3& v) {
    smd4 len_sq = _mm_dp_ps(v.v, v.v, 0x77);    
    smd4 nr = _mm_rsqrt_ps(len_sq);    
    // 1-Step Newton-Raphson Refinement: nr = nr * 0.5 * (3.0 - len_sq * nr * nr)
    smd4 half       = _mm_set1_ps(0.5f);
    smd4 three      = _mm_set1_ps(3.0f);
    smd4 muls       = _mm_mul_ps(_mm_mul_ps(len_sq, nr), nr);
    smd4 nr_refined = _mm_mul_ps(_mm_mul_ps(half, nr), _mm_sub_ps(three, muls));

    return vec3(_mm_mul_ps(v.v, nr_refined));
}

using point3 = vec3;
using color  = vec3;
