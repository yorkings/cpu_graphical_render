#pragma once
#include "vec3.h"

inline smd8 simd_sign(smd8 v) {
    smd8 sign_mask = _mm256_set1_ps(-0.0f);
    smd8 one       = _mm256_set1_ps(1.0f);
    return _mm256_or_ps(one, _mm256_and_ps(v, sign_mask));
}

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

        inline vec8 permutevar(const __m256i& perm_idx) const {
            return vec8( _mm256_permutevar8x32_ps(x, perm_idx), _mm256_permutevar8x32_ps(y, perm_idx), _mm256_permutevar8x32_ps(z, perm_idx));
        }

        template<int Imm>
        inline vec8 permute2f128() const {
            return vec8(_mm256_permute2f128_ps(x, x, Imm),_mm256_permute2f128_ps(y, y, Imm),_mm256_permute2f128_ps(z, z, Imm));
        }
        // Swaps horizontal neighbors in 2x(2x2) quad layout: [Lane 0<->2, 1<->3, 4<->6, 5<->7]
        inline vec8 permute_quad_dx() const {
            smd8 rx_x = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
            smd8 rx_y = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(1, 0, 3, 2));
            smd8 rx_z = _mm256_shuffle_ps(z, z, _MM_SHUFFLE(1, 0, 3, 2));
            return vec8(rx_x, rx_y, rx_z);
        }

        // Swaps vertical neighbors in 2x(2x2) quad layout: [Lane 0<->1, 2<->3, 4<->5, 6<->7]
        inline vec8 permute_quad_dy() const {
            smd8 ry_x = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
            smd8 ry_y = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(2, 3, 0, 1));
            smd8 ry_z = _mm256_shuffle_ps(z, z, _MM_SHUFFLE(2, 3, 0, 1));
            return vec8(ry_x, ry_y, ry_z);
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
    return vec8(
        _mm256_fmsub_ps(u.y, v.z, _mm256_mul_ps(u.z, v.y)),
        _mm256_fmsub_ps(u.z, v.x, _mm256_mul_ps(u.x, v.z)),
        _mm256_fmsub_ps(u.x, v.y, _mm256_mul_ps(u.y, v.x))
    );
}

inline vec8 unit_vector(const vec8& v) {
    smd8 len_sq = v.length_squared();
    return v * precise_inv(precise_sqrt(len_sq));
}


inline vec8 precise_inv(const vec8& v) {
    return vec8(precise_inv(v.x), precise_inv(v.y), precise_inv(v.z));
}


inline vec2_x8 oct_encode(const vec8& dir) {
    smd8 sign_mask = _mm256_set1_ps(-0.0f);
    smd8 abs_x     = _mm256_andnot_ps(sign_mask, dir.x);
    smd8 abs_y     = _mm256_andnot_ps(sign_mask, dir.y);
    smd8 abs_z     = _mm256_andnot_ps(sign_mask, dir.z);

    smd8 l1_norm   = _mm256_add_ps(_mm256_add_ps(abs_x, abs_y), abs_z);

    vec2_x8 uv(precise_div(dir.x, l1_norm), precise_div(dir.y, l1_norm));

    smd8 one       = _mm256_set1_ps(1.0f);
    smd8 abs_u     = _mm256_andnot_ps(sign_mask, uv.x);
    smd8 abs_v     = _mm256_andnot_ps(sign_mask, uv.y);

    vec2_x8 refl(_mm256_mul_ps(_mm256_sub_ps(one, abs_v), simd_sign(uv.x)),_mm256_mul_ps(_mm256_sub_ps(one, abs_u), simd_sign(uv.y)));
    smd8 z_lt_0    = _mm256_cmp_ps(dir.z, _mm256_set1_ps(0.0f), _CMP_LT_OQ);
    
    return blendv(uv, refl, z_lt_0);
}

inline vec8 oct_decode(const vec2_x8& uv) {
    smd8 sign_mask = _mm256_set1_ps(-0.0f);
    smd8 one       = _mm256_set1_ps(1.0f);

    smd8 abs_u     = _mm256_andnot_ps(sign_mask, uv.x);
    smd8 abs_v     = _mm256_andnot_ps(sign_mask, uv.y);

    smd8 z         = _mm256_sub_ps(one, _mm256_add_ps(abs_u, abs_v));
    smd8 z_lt_0    = _mm256_cmp_ps(z, _mm256_set1_ps(0.0f), _CMP_LT_OQ);

    smd8 refl_x = _mm256_fnmadd_ps(abs_v, simd_sign(uv.x), simd_sign(uv.x));
    smd8 refl_y = _mm256_fnmadd_ps(abs_u, simd_sign(uv.y), simd_sign(uv.y));

    smd8 x         = _mm256_blendv_ps(uv.x, refl_x, z_lt_0);
    smd8 y         = _mm256_blendv_ps(uv.y, refl_y, z_lt_0);

    return unit_vector(vec8(x, y, z));
}

inline vec8 precise_div(const vec8& u, const vec8& v) {
    return vec8(precise_div(u.x, v.x), precise_div(u.y, v.y), precise_div(u.z, v.z));
}

inline vec8 lerp(const vec8& a, const vec8& b, smd8 t) {
    return vec8(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t));
}