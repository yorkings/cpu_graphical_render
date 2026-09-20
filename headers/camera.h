#pragma once
#include "ray.h"
#include "spectrum.h"
#include "transform.hpp"
#include<mdspan> 

enum class CameraType : uint8_t { Perspective, Orthographic, Fisheye, Spherical, Environment };

struct CameraSample{
 vec2_x8 pfilm;
 vec2_x8 lens;
 smd8 time;
 smd8 weight;
 smd8i medium_id;
};

struct CameraRay {
    ray8 rays;
    Spectrum weight = Spectrum::MakeConstant(1.0f);
    smd8 has_differentials = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
};



struct Camera{
    CameraType type;
    //AnimatedTransform cam_to_world;      
    float lens_radius    = 0.0f;
    float focal_distance = 10.0f;
    float fov            = 90.0f;    
    // Raster-to-Camera space conversion constants
    float dx_scale       = 1.0f;
    float dy_scale       = 1.0f;
    
};

