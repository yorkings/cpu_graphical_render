#pragma once
#include "ray.h"

enum class ShapeType:uint8_t{
    none=0,sphere,cylinder,disk,triangle,curve,quad,bilinear_patches
};