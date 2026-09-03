#pragma once
#include "vec3_smd8.h"
#include "smd8_transpose.h"

struct ray{
    point3 orig;
    vec3 dir;
    float time;
    uint32_t medium_id;

    inline ray() : orig(), dir(), time(0.0f), medium_id(0) {}
    inline ray(const point3& origin, const vec3& direction,float tm = 0.0f, uint32_t med = 0):orig(origin), dir(direction),time(tm),medium_id(med){}
    inline point3 at(float t) const {
        return point3(_mm_fmadd_ps(_mm_set1_ps(t), dir.v, orig.v));
    }
};

struct alignas(32) ray8{
    vec8 orig;
    vec8 dir;
    vec8 inv_dir;    
    // Sign masks for zero-branch AABB box bounds swapping
    smd8 sign_x, sign_y, sign_z;
    smd8 time;
    smd8i medium_id;

    inline ray8()=default;
    inline ray8(const vec8& origin, const vec8& direction,const smd8 &ray_time,const smd8i &med_id): orig(origin), dir(direction),time(ray_time),medium_id(med_id),inv_dir(precise_inv(direction)) {
        smd8 sign_mask = _mm256_set1_ps(-0.0f);
        sign_x = _mm256_and_ps(dir.x, sign_mask);
        sign_y = _mm256_and_ps(dir.y, sign_mask);
        sign_z = _mm256_and_ps(dir.z, sign_mask);
    }

    inline vec8 at(smd8 t) const {
        return vec8(_mm256_fmadd_ps(t, dir.x, orig.x),_mm256_fmadd_ps(t, dir.y, orig.y),_mm256_fmadd_ps(t, dir.z, orig.z));
    }
    
    static inline ray8 load_aos(const ray rays[8]) {
        const float* ptr = reinterpret_cast<const float*>(rays);        
        vec8 o, d;
        smd8 t_vec, med_vec;
        load_aos_transpose8x8(ptr, o.x, o.y, o.z, d.x, d.y, d.z, t_vec, med_vec);
        smd8i med_id_i = _mm256_castps_si256(med_vec);
        return ray8(o, d, t_vec,med_id_i);
    }
    // Computes ray origin differentials (dpdx, dpdy) for 2x(2x2) dual-quad layout
    inline void get_pixel_differentials(vec8& dpdx, vec8& dpdy) const {
        // Sign correction masks: flip backward differences on right (lanes 2,3,6,7) & bottom (lanes 1,3,5,7)
        smd8 x_sign_flip = _mm256_set_ps(-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f);
        smd8 y_sign_flip = _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
    
        dpdx = (orig.permute_quad_dx() - orig) * x_sign_flip;
        dpdy = (orig.permute_quad_dy() - orig) * y_sign_flip;
    }
    
    // Computes ray direction differentials (dddx, dddy) for 2x(2x2) dual-quad layout
    inline void get_direction_differentials(vec8& dddx, vec8& dddy) const {
        smd8 x_sign_flip = _mm256_set_ps(-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f);
        smd8 y_sign_flip = _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
    
        dddx = (dir.permute_quad_dx() - dir) * x_sign_flip;
        dddy = (dir.permute_quad_dy() - dir) * y_sign_flip;
    }
    
    inline ray get(int i) const {
        return ray(orig.get(i), dir.get(i),time[i],static_cast<uint32_t>(medium_id[i]));
    }
};


