#pragma once
#include "ray.h"

struct alignas(32) SurfaceInteraction8 {
    vec8 p; 
    vec8 n;    
    smd8 t;  
    smd8 u, v;

    vec8 dpdu, dpdv;  // Derivatives of position w.r.t (u, v)
    vec8 dndu, dndv;  // Derivatives of normal w.r.t (u, v) [for curvature/shading]

    vec8 shading_n;   // Perturbed shading normal

    smd8i material_id;  // Which material to evaluate
    smd8i primitive_id; // Specific instance/primitive
    smd8i area_light_id;// Index if primitive is an area light (Le)

    smd8 hitmask;       

    inline void get_shading_frame(vec8& tangent, vec8& bitangent) const {
        tangent = unit_vector(dpdu);
        bitangent = cross(n, tangent);
    }
};