#pragma once
#include "vec2_smd8.h"
#include "vec3_smd8.h"
#include<mdspan> 

struct CameraSample{
 vec2_x8 pfilm;
 vec2_x8 lens;
 smd8 time;
};

class camera{
    public:
    protected:
};