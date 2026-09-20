#pragma once
#include "mat4.h"
#include "ray.h"

struct alignas(32) Transform{
    mat4 m;
    mat4 m_inv;
    constexpr Transform() noexcept : m(), m_inv() {}
    constexpr Transform(const mat4& mat, const mat4& inv_mat) noexcept : m(mat), m_inv(inv_mat) {}

    [[nodiscard]] inline Transform inverse() const noexcept {
        return Transform(m_inv, m);
    }
    
    [[nodiscard]] inline mat4 normal_matrix() const noexcept {
        return m_inv.transpose();
    }


    [[nodiscard]] inline vec8 transform_point(const vec8& p) const noexcept {
        return m.transform_point(p);
    }

    [[nodiscard]] inline vec8 transform_vector(const vec8& v) const noexcept {
        return m.transform_vector(v);
    }

    [[nodiscard]] inline Transform operator*(const Transform& rhs) const noexcept {
        return Transform(m * rhs.m, rhs.m_inv * m_inv);
    }
    
    [[nodiscard]] static constexpr Transform translate(const vec3& translation) noexcept {
        mat4 m = mat4::identity();
        m.m[3][0] = translation.x();
        m.m[3][1] = translation.y();
        m.m[3][2] = translation.z();
        return Transform(m, m.inverse_affine());
    }

    [[nodiscard]] static constexpr Transform scale(const vec3& scale) noexcept {
        mat4 m = mat4::identity();
        m.m[0][0] = scale.x();
        m.m[1][1] = scale.y();
        m.m[2][2] = scale.z();
        return Transform(m, m.inverse_affine());
    }

    [[nodiscard]] inline vec8 transform_normal(const vec8& n) const noexcept {
        return m_inv.transpose().transform_vector(n);
    }

    [[nodiscard]] inline ray8 transform_ray8(const ray8& r) const noexcept {
        vec8 o = transform_point(r.orig);
        vec8 d = transform_vector(r.dir);
        return ray8(o, d, r.time, r.medium_id);
    }

};


inline Transform look_at(const vec3& pos, const vec3& target, const vec3& up) noexcept {
    vec3 dir = unit_vector(target - pos);
    vec3 right = unit_vector(cross(dir, up));
    vec3 new_up = cross(right, dir);

    mat4 world_to_cam(right.x(), right.y(), right.z(), -dot(right,  pos),new_up.x(),new_up.y(),new_up.z(),-dot(new_up, pos),
        -dir.x(),-dir.y(),-dir.z(),dot(dir,   pos), 0.0f,0.0f,0.0f,1.0f);    

    return Transform(world_to_cam, world_to_cam.inverse_affine());
}

