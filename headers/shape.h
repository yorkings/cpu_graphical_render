#pragma once
#include "ray.h"
#include "vec3_smd8.h"
#include "math_utils.h"
#include "records.h"

enum class ShapeType:uint8_t{
    none=0,sphere,cylinder,disk,triangle,curve,quad,bilinear_patches
};

struct alignas(32) Sphere {
    Sphere(const vec3& c, float r, float zmin = -1.0f, float zmax = 1.0f, float pmin = 0.0f, float pmax = 360.0f): center(c), radius(r),
        z_range(std::clamp(std::min(zmin, zmax), -r, r), std::clamp(std::max(zmin, zmax), -r, r)),phiMin(degrees_to_radians(std::clamp(pmin, 0.0f, 360.0f))),
        phiMax(degrees_to_radians(std::clamp(pmax, 0.0f, 360.0f))){
        float inv_r = 1.0f / radius;
        thetaMin = std::acos(std::clamp(z_range.max * inv_r, -1.0f, 1.0f));
        thetaMax = std::acos(std::clamp(z_range.min * inv_r, -1.0f, 1.0f));
        dTheta   = thetaMax - thetaMin;
        dPhi     = phiMax - phiMin;
        invDTheta = (std::abs(dTheta) > 1e-6f) ? (1.0f / dTheta) : 0.0f;
        invDPhi   = (std::abs(dPhi) > 1e-6f)   ? (1.0f / dPhi)   : 0.0f;
        
        is_full_sphere = (std::abs(dPhi - two_pi) < 1e-4f) && (std::abs(z_range.min + radius) < 1e-4f) && (std::abs(z_range.max - radius) < 1e-4f);
    }
    vec3 center;
    interval z_range;
    float radius;
    float phiMin, phiMax;
    float thetaMin, thetaMax;
    float dTheta, dPhi;
    float invDTheta, invDPhi;
    bool is_full_sphere;
};

struct cylinder{
    public:
     cylinder(float zmin,float zmax,float r):z_range(zmin, zmax),radius(r){}
    private:
        interval z_range;
        float radius; 
};

struct disk{
    public:
        disk(float zmin,float zmax,float r):z_range(zmin, zmax),radius(r){}
    private:
        interval z_range;
        float radius;
};
struct triangle{
    public:
        triangle(const vec3& v0,const vec3& v1,const vec3& v2):v0(v0),v1(v1),v2(v2){}
    private:
        vec3 v0,v1,v2;
};
struct curve{
    public:
        curve(const vec3& p0,const vec3& p1,const vec3& p2):p0(p0),p1(p1),p2(p2){}
    private:
        vec3 p0,p1,p2;
};
struct quad{
    public:
        quad(const vec3& v0,const vec3& v1,const vec3& v2,const vec3& v3):v0(v0),v1(v1),v2(v2),v3(v3){}
    private:
        vec3 v0,v1,v2,v3;
};
struct bilinear_patches{
    public:
        bilinear_patches(const vec3& v0,const vec3& v1,const vec3& v2,const vec3& v3):v0(v0),v1(v1),v2(v2),v3(v3){}
    private:
        vec3 v0,v1,v2,v3;
};

struct SphereShadingGeometry8 {
    vec2x8 uv;
    vec8 dpdu, dpdv;
    vec8 dndu, dndv;
};

inline SphereShadingGeometry8 compute_sphere_shading_geometry(const vec8 &p_local, const Sphere &sphere,smd8 inv_r,smd8 phi) {
    SphereShadingGeometry8 geom;

    smd8 phimin     = _mm256_set1_ps(sphere.phiMin);
    smd8 inv_dphi   = _mm256_set1_ps(sphere.invDPhi);
    smd8 dphi       = _mm256_set1_ps(sphere.dPhi);
    smd8 theta_min  = _mm256_set1_ps(sphere.thetaMin);
    smd8 inv_dtheta = _mm256_set1_ps(sphere.invDTheta);
    smd8 dtheta     = _mm256_set1_ps(sphere.dTheta);
    smd8 cos_phi,sin_phi;

    geom.dpdu = vec8(_mm256_mul_ps(_mm256_sub_ps(v_zero, dphi), p_local.y),_mm256_mul_ps(dphi, p_local.x),v_zero);

    smd8 r2_xy     = _mm256_fmadd_ps(p_local.x, p_local.x, _mm256_mul_ps(p_local.y, p_local.y));
    smd8 r_xy      = precise_sqrt(r2_xy);
    smd8 safe_r_xy = _mm256_max_ps(r_xy, _mm256_set1_ps(1e-5f));
    smd8 inv_r_xy  = precise_inv(safe_r_xy);
    
    sincos_simd8(phi,sin_phi,cos_phi);    
    geom.dpdv = vec8(_mm256_mul_ps(dtheta, _mm256_mul_ps(p_local.z, cos_phi)),_mm256_mul_ps(dtheta, _mm256_mul_ps(p_local.z, sin_phi)),_mm256_mul_ps(_mm256_sub_ps(v_zero, dtheta), r_xy));

    smd8 cos_theta = clamp(_mm256_mul_ps(p_local.z, inv_r), v_neg_one, v_one);
    smd8 theta     = simd_acos(cos_theta);

    geom.uv.x = _mm256_mul_ps(_mm256_sub_ps(phi, phimin), inv_dphi);
    geom.uv.y = _mm256_mul_ps(_mm256_sub_ps(theta, theta_min), inv_dtheta);

    geom.dndu = geom.dpdu * inv_r;
    geom.dndv = geom.dpdv * inv_r;
    return geom;
}


inline smd8 intersect_sphere(const ray8 &rays, const Sphere &sphere, interval8 ray_t, SurfaceInteraction8 &recs) {
    smd8 r      = _mm256_set1_ps(sphere.radius);
    vec8 center = vec8(sphere.center.x(), sphere.center.y(), sphere.center.z());
    smd8 inv_r  = precise_inv(r);

    vec8 oc = rays.orig - center;

    smd8 a      = dot(rays.dir, rays.dir);
    smd8 half_b = dot(oc, rays.dir);
    smd8 c      = _mm256_fnmadd_ps(r, r, dot(oc, oc));

    smd8 discriminant = _mm256_fnmadd_ps(a, c, _mm256_mul_ps(half_b, half_b));
    smd8 has_hit      = _mm256_cmp_ps(discriminant, v_zero, _CMP_GE_OQ);

    if (_mm256_movemask_ps(has_hit) == 0) return v_zero;

    smd8 sqrtd      = precise_sqrt(discriminant);
    smd8 inv_a      = precise_inv(a);
    smd8 neg_half_b = _mm256_sub_ps(v_zero, half_b);

    smd8 root1 = _mm256_mul_ps(_mm256_sub_ps(neg_half_b, sqrtd), inv_a); // Front root
    smd8 root2 = _mm256_mul_ps(_mm256_add_ps(neg_half_b, sqrtd), inv_a); // Back root

    interval8 active_t(ray_t.min, recs.t);
    
    smd8 valid_t1 = active_t.surrounds(root1);
    smd8 valid_t2 = active_t.surrounds(root2);

    vec8 p_hit1   = rays.at(root1);
    vec8 p_local1 = p_hit1 - center;

    vec8 p_hit2   = rays.at(root2);
    vec8 p_local2 = p_hit2 - center;

    smd8 valid1 = _mm256_and_ps(has_hit, valid_t1);
    smd8 valid2 = _mm256_and_ps(has_hit, valid_t2);
    // Fast-path / Lazy Phi evaluation
    if (!sphere.is_full_sphere) {
        smd8 z_min = _mm256_set1_ps(sphere.z_range.min);
        smd8 z_max = _mm256_set1_ps(sphere.z_range.max);
        smd8 phimin = _mm256_set1_ps(sphere.phiMin);
        smd8 phimax = _mm256_set1_ps(sphere.phiMax);

        smd8 valid_z1 = _mm256_and_ps(_mm256_cmp_ps(p_local1.z, z_min, _CMP_GE_OQ), _mm256_cmp_ps(p_local1.z, z_max, _CMP_LE_OQ));
        smd8 valid_z2 = _mm256_and_ps(_mm256_cmp_ps(p_local2.z, z_min, _CMP_GE_OQ), _mm256_cmp_ps(p_local2.z, z_max, _CMP_LE_OQ));

        valid1 = _mm256_and_ps(valid1, valid_z1);
        valid2 = _mm256_and_ps(valid2, valid_z2);
    
        if (_mm256_movemask_ps(_mm256_or_ps(valid1, valid2)) != 0) {
            smd8 phi1 = simd_atan2(p_local1.y, p_local1.x);
            phi1 = blendv(phi1, _mm256_add_ps(phi1, two_pi_simd8), _mm256_cmp_ps(phi1, v_zero, _CMP_LT_OQ));
            smd8 valid_phi1 = _mm256_and_ps(_mm256_cmp_ps(phi1, phimin, _CMP_GE_OQ), _mm256_cmp_ps(phi1, phimax, _CMP_LE_OQ));
            valid1 = _mm256_and_ps(valid1, valid_phi1);
            smd8 phi2 = simd_atan2(p_local2.y, p_local2.x);
            phi2 = blendv(phi2, _mm256_add_ps(phi2, two_pi_simd8), _mm256_cmp_ps(phi2, v_zero, _CMP_LT_OQ));
            smd8 valid_phi2 = _mm256_and_ps(_mm256_cmp_ps(phi2, phimin, _CMP_GE_OQ), _mm256_cmp_ps(phi2, phimax, _CMP_LE_OQ));
            valid2 = _mm256_and_ps(valid2, valid_phi2);
        }
    }

    smd8 hit_mask = _mm256_or_ps(valid1, valid2);    
    if (_mm256_movemask_ps(hit_mask) == 0) return v_zero;

    smd8 selected_root = blendv(root2, root1, valid1);
    vec8 p_hit          = blendv(p_hit2, p_hit1, valid1);
    vec8 p_local        = blendv(p_local2, p_local1, valid1);

    smd8 phi = simd_atan2(p_local.y, p_local.x);
    phi = blendv(phi, _mm256_add_ps(phi, two_pi_simd8), _mm256_cmp_ps(phi, v_zero, _CMP_LT_OQ));
    
    SphereShadingGeometry8 geom = compute_sphere_shading_geometry(p_local, sphere, inv_r, phi);
    vec8 outward_normal = p_local * inv_r;

    recs.t         = blendv(recs.t, selected_root, hit_mask);
    recs.p         = blendv(recs.p, p_hit, hit_mask);
    recs.n         = blendv(recs.n, outward_normal, hit_mask);
    recs.shading_n = blendv(recs.shading_n, outward_normal, hit_mask);
    recs.u         = blendv(recs.u, geom.uv.x, hit_mask);
    recs.v         = blendv(recs.v, geom.uv.y, hit_mask);
    recs.dpdu      = blendv(recs.dpdu, geom.dpdu, hit_mask);
    recs.dpdv      = blendv(recs.dpdv, geom.dpdv, hit_mask);
    recs.dndu      = blendv(recs.dndu, geom.dndu, hit_mask);
    recs.dndv      = blendv(recs.dndv, geom.dndv, hit_mask);
    recs.hitmask   = _mm256_or_ps(recs.hitmask, hit_mask);

    return hit_mask;
}
