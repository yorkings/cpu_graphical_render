#pragma once
#include "vec3_smd8.h"
#include "smd8_transpose.h"

inline vec8 precise_inv(const vec8& v) {
    return vec8(precise_inv(v.x), precise_inv(v.y), precise_inv(v.z));
}

struct ray{
    point3 orig;
    vec3 dir;
    float time;
    float pad;

    inline ray() : orig(), dir(), time(0.0f), pad(0.0f) {}
    inline ray(const point3& origin, const vec3& direction):ray(orig,dir,0.0f){}
    inline ray(const point3& origin, const vec3& direction,float tm):orig(origin), dir(direction),time(tm),pad(0.0f){}

    inline const point3& origin() const { return orig; }
    inline const vec3& direction() const { return dir; }
    inline const float &time()const{return time;}

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

    inline ray8()=default;
    inline ray8(const vec8& origin, const vec8& direction,const smd8 &ray_time): orig(origin), dir(direction),time(ray_time),inv_dir(precise_inv(direction)) {
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
        smd8 t_vec, pad_vec;
        load_aos_transpose8x8(ptr, o.x, o.y, o.z, d.x, d.y, d.z, t_vec, pad_vec);

        return ray8(o, d, t_vec);
    }
    
    inline ray get(int i) const {
        return ray(orig.get(i), dir.get(i),time[i]);
    }
};
