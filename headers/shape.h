#pragma once
#include "ray.h"
#include "vec3_smd8.h"
#include "math_utils.h"

enum class ShapeType:uint8_t{
    none=0,sphere,cylinder,disk,triangle,curve,quad,bilinear_patches
};

struct alignas(32) Sphere {
    public:
        Sphere(const vec3& c, float r, float zmin = -1.0f, float zmax = 1.0f, float pm = 360.0f):center(c),radius(r),
            z_range(std::clamp(std::min(zmin, zmax), -r, r),std::clamp(std::max(zmin, zmax), -r, r)),phiMax(degrees_to_radians(std::clamp(pm, 0.0f, 360.0f))){}

        inline vec3 get_center(){return center;}
        inline interval get_z_range(){ return z_range;}
        inline float get_max_phi(){return phiMax;}
    private:
        vec3 center;
        interval z_range;
        float radius;        
        // float thetaMin, thetaMax;
        float phiMax;
};

struct cylinder{
    float zMin,zMax;
    float radius; 
};

struct disk{};
struct triangle{};
struct curve{};
struct quad{};
struct bilinear_patches{};

inline smd8 intersect_sphere(const ray8 &rays,const Sphere &sphere){
    
}