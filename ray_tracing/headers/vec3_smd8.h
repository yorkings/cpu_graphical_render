#pragma once
#include "vec3.h"
class vec8{
    public:
        smd8 x,y,z;

        inline vec8():x(_mm256_setzero_ps()), y(_mm256_setzero_ps()), z(_mm256_setzero_ps()) {}
        inline vec8(smd8 x, smd8 y, smd8 z):x(x), y(y), z(z) {}
        inline vec8(float vx, float vy, float vz):x(_mm256_set1_ps(vx)), y(_mm256_set1_ps(vy)), z(_mm256_set1_ps(vz)) {}
        inline vec8(const vec3& v): x(_mm256_set1_ps(v.x())), y(_mm256_set1_ps(v.y())), z(_mm256_set1_ps(v.z())) {}

        static inline vec8 load_aos(const vec3 v[8]) {
            alignas(32) float fx[8], fy[8], fz[8];
            for (int i = 0; i < 8; ++i) {
                fx[i] = v[i].x();
                fy[i] = v[i].y();
                fz[i] = v[i].z();
            }
            return vec8(_mm256_load_ps(fx), _mm256_load_ps(fy), _mm256_load_ps(fz));
        }
        
        inline vec3 get(int i) const {
            alignas(32) float fx[8], fy[8], fz[8];
            _mm256_store_ps(fx, x);
            _mm256_store_ps(fy, y);
            _mm256_store_ps(fz, z);
            return vec3(fx[i], fy[i], fz[i]);
        }

        inline vec8 operator-() const {
            smd8 sign_bit = _mm256_set1_ps(-0.0f);
            return vec8(_mm256_xor_ps(x, sign_bit),_mm256_xor_ps(y, sign_bit),_mm256_xor_ps(z, sign_bit));
        }

        inline vec8& operator+=(const vec8& b) {
            x = _mm256_add_ps(x, b.x);
            y = _mm256_add_ps(y, b.y);
            z = _mm256_add_ps(z, b.z);
            return *this;
        }

        inline vec8& operator-=(const vec8& b) {
            x = _mm256_sub_ps(x, b.x);
            y = _mm256_sub_ps(y, b.y);
            z = _mm256_sub_ps(z, b.z);
            return *this;
        }

        inline vec8& operator*=(smd8 scalar) {
            x = _mm256_mul_ps(x, scalar);
            y = _mm256_mul_ps(y, scalar);
            z = _mm256_mul_ps(z, scalar);
            return *this;
        }

        inline vec8& operator*=(float scalar) {
            return *this *= _mm256_set1_ps(scalar);
        }

        inline smd8 length_squared() const {
            smd8 xx = _mm256_mul_ps(x, x);
            smd8 yy = _mm256_fmadd_ps(y, y, xx);
            return _mm256_fmadd_ps(z, z, yy);
        }
};

inline vec8 operator+(const vec8& u, const vec8& v) {
    return vec8(_mm256_add_ps(u.x, v.x), _mm256_add_ps(u.y, v.y), _mm256_add_ps(u.z, v.z));
}

inline vec8 operator-(const vec8& u, const vec8& v) {
    return vec8(_mm256_sub_ps(u.x, v.x), _mm256_sub_ps(u.y, v.y), _mm256_sub_ps(u.z, v.z));
}

inline vec8 operator*(const vec8& u, const vec8& v) {
    return vec8(_mm256_mul_ps(u.x, v.x), _mm256_mul_ps(u.y, v.y), _mm256_mul_ps(u.z, v.z));
}

inline vec8 operator*(smd8 s, const vec8& v) {
    return vec8(_mm256_mul_ps(s, v.x), _mm256_mul_ps(s, v.y), _mm256_mul_ps(s, v.z));
}

inline vec8 operator*(const vec8& v, smd8 s) { return s * v; }
inline vec8 operator*(float s, const vec8& v) { return _mm256_set1_ps(s) * v; }
inline vec8 operator*(const vec8& v, float s) { return s * v; }

inline smd8 dot(const vec8& u, const vec8& v) {
    smd8 xx = _mm256_mul_ps(u.x, v.x);
    smd8 yy = _mm256_fmadd_ps(u.y, v.y, xx);
    return _mm256_fmadd_ps(u.z, v.z, yy);
}
inline vec8 cross(const vec8& u, const vec8& v) {
    smd8 cx = _mm256_fmsub_ps(u.y, v.z, _mm256_mul_ps(u.z, v.y));
    smd8 cy = _mm256_fmsub_ps(u.z, v.x, _mm256_mul_ps(u.x, v.z));
    smd8 cz = _mm256_fmsub_ps(u.x, v.y, _mm256_mul_ps(u.y, v.x));
    return vec8(cx, cy, cz);
}

inline vec8 unit_vector(const vec8& v) {
    smd8 len_sq = v.length_squared();
    smd8 nr = _mm256_rsqrt_ps(len_sq);    
    // Newton-Raphson Iteration for 23-bit precision
    smd8 half       = _mm256_set1_ps(0.5f);
    smd8 three      = _mm256_set1_ps(3.0f);
    smd8 muls       = _mm256_mul_ps(_mm256_mul_ps(len_sq, nr), nr);
    smd8 nr_refined = _mm256_mul_ps(_mm256_mul_ps(half, nr), _mm256_sub_ps(three, muls));
    return v * nr_refined;
}