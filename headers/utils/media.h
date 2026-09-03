#pragma once
#include "vec3.h"

enum class MediumType : uint32_t {
    Vacuum      = 0,
    Air         = 1,
    ClearWater  = 2,
    HomogeneousFog = 3,
    GlassInterior  = 4
};
struct MediumProperties {
    vec3 sigma_ab; // Absorption coefficient
    vec3 sigma_sc; // Scattering coefficient
    float g;      // Phase function anisotropy (Henyey-Greenstein)
};

// Global or scene-wide media table
struct SceneMedia {
    std::vector<MediumProperties> media;
    inline const MediumProperties& get(uint32_t id) const {
        return media[id];
    }
};