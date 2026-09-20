#pragma once
#include "camera.h"

enum class FilmFilterType:uint8_t{
    BoxFilter,TriangleFilter,GausianFilter,Mitchell_NetravaliFilter,LancozFilter
};
enum class FilmType:uint8_t{
    GbufferFilm,spectralFilm,RGBFilm
};

struct Mitchelfilter{
    smd8 c3, c2, c0;
    smd8 d3, d2, d1, d0;
    
};
