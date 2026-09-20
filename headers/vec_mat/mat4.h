#pragma once
#include "vec3_smd8.h"

struct alignas(32) mat4 {
    alignas(32) float m[4][4];
    constexpr mat4() : m{
                     {1.0f, 0.0f, 0.0f, 0.0f},
                     {0.0f, 1.0f, 0.0f, 0.0f},
                     {0.0f, 0.0f, 1.0f, 0.0f},
                     {0.0f, 0.0f, 0.0f, 1.0f}
                    } {}
    constexpr mat4(float m00, float m01, float m02, float m03,float m10, float m11, float m12, float m13,float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33)noexcept : m(
            {{m00, m10, m20, m30},{m01, m11, m21, m31},{m02, m12, m22, m32},{m03, m13, m23, m33}//column-major order
        }){}
    
    [[nodiscard]] static constexpr mat4 identity() noexcept {
        return mat4();
    }
    //P_out = M * P_in
    [[nodiscard]] inline vec8 transform_point(const vec8& p) const noexcept {
        smd8 c0_x = _mm256_set1_ps(m[0][0]), c0_y = _mm256_set1_ps(m[0][1]), c0_z = _mm256_set1_ps(m[0][2]), c0_w = _mm256_set1_ps(m[0][3]);
        smd8 c1_x = _mm256_set1_ps(m[1][0]), c1_y = _mm256_set1_ps(m[1][1]), c1_z = _mm256_set1_ps(m[1][2]), c1_w = _mm256_set1_ps(m[1][3]);
        smd8 c2_x = _mm256_set1_ps(m[2][0]), c2_y = _mm256_set1_ps(m[2][1]), c2_z = _mm256_set1_ps(m[2][2]), c2_w = _mm256_set1_ps(m[2][3]);    
        smd8 c3_x = _mm256_set1_ps(m[3][0]), c3_y = _mm256_set1_ps(m[3][1]), c3_z = _mm256_set1_ps(m[3][2]), c3_w = _mm256_set1_ps(m[3][3]);

        smd8 rx = _mm256_fmadd_ps(c0_x, p.x, _mm256_fmadd_ps(c1_x, p.y, _mm256_fmadd_ps(c2_x, p.z, c3_x)));
        smd8 ry = _mm256_fmadd_ps(c0_y, p.x, _mm256_fmadd_ps(c1_y, p.y, _mm256_fmadd_ps(c2_y, p.z, c3_y)));
        smd8 rz = _mm256_fmadd_ps(c0_z, p.x, _mm256_fmadd_ps(c1_z, p.y, _mm256_fmadd_ps(c2_z, p.z, c3_z)));
        smd8 rw = _mm256_fmadd_ps(c0_w, p.x, _mm256_fmadd_ps(c1_w, p.y, _mm256_fmadd_ps(c2_w, p.z, c3_w)));

        smd8 inv_w = precise_inv(rw);
        return vec8(rx * inv_w, ry * inv_w, rz * inv_w);
    }
    [[nodiscard]] inline vec8 transform_vector(const vec8& v) const noexcept {
        smd8 c0_x = _mm256_set1_ps(m[0][0]), c0_y = _mm256_set1_ps(m[0][1]), c0_z = _mm256_set1_ps(m[0][2]);
        smd8 c1_x = _mm256_set1_ps(m[1][0]), c1_y = _mm256_set1_ps(m[1][1]), c1_z = _mm256_set1_ps(m[1][2]);
        smd8 c2_x = _mm256_set1_ps(m[2][0]), c2_y = _mm256_set1_ps(m[2][1]), c2_z = _mm256_set1_ps(m[2][2]);

        smd8 rx = _mm256_fmadd_ps(c0_x, v.x, _mm256_fmadd_ps(c1_x, v.y, _mm256_mul_ps(c2_x, v.z)));
        smd8 ry = _mm256_fmadd_ps(c0_y, v.x, _mm256_fmadd_ps(c1_y, v.y, _mm256_mul_ps(c2_y, v.z)));
        smd8 rz = _mm256_fmadd_ps(c0_z, v.x, _mm256_fmadd_ps(c1_z, v.y, _mm256_mul_ps(c2_z, v.z)));

        return vec8(rx, ry, rz);
    }

    [[nodiscard]] mat4 operator*(const mat4& rhs) const noexcept {
        mat4 result;
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                result.m[c][r] = m[0][r]*rhs.m[c][0] + m[1][r]*rhs.m[c][1]+ m[2][r]*rhs.m[c][2] + m[3][r]*rhs.m[c][3];
            }
        }
        return result;
    }

    [[nodiscard]] mat4 transpose() const noexcept {
        mat4 result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result.m[i][j] = m[j][i];
            }
        }
        return result;
    }

    [[nodiscard]] constexpr mat4 inverse_affine() const noexcept{
        mat4 inv;
        float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        if (det == 0.0f) {
            return mat4();
        }

        float inv_det = 1.0f / det;

        inv.m[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
        inv.m[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det;
        inv.m[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
        inv.m[0][3] = 0.0f;

        inv.m[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det;
        inv.m[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
        inv.m[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det;
        inv.m[1][3] = 0.0f;

        inv.m[2][0] =  (m[1][0] * m[2][1] - m[1][1] *m[2][0]) * inv_det;
        inv.m[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det;
        inv.m[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
        inv.m[2][3] = 0.0f; 

        float tx = m[3][0], ty = m[3][1], tz = m[3][2];
        inv.m[3][0] = -(inv.m[0][0] * tx + inv.m[1][0] * ty + inv.m[2][0] * tz);
        inv.m[3][1] = -(inv.m[0][1] * tx + inv.m[1][1] * ty + inv.m[2][1] * tz);
        inv.m[3][2] = -(inv.m[0][2] * tx + inv.m[1][2] * ty + inv.m[2][2] * tz);
        inv.m[3][3] = 1.0f;

        return inv;
    }

};