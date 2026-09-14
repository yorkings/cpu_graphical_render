#pragma once
#include "general.h"


struct interval{
    smd8 min;
    smd8 max;

    interval():min(_mm256_set1_ps(+infinity)), max(_mm256_set1_ps(-infinity)) {}
    interval(smd8 min_val, smd8 max_val):min(min_val), max(max_val) {}
    interval(float min_val, float max_val):min(_mm256_set1_ps(min_val)), max(_mm256_set1_ps(max_val)) {}

    inline smd8 length() const { return _mm256_sub_ps(max, min); }
    inline smd8 center() const { return _mm256_mul_ps(_mm256_add_ps(min, max), _mm256_set1_ps(0.5f)); }
    inline bool is_valid() const { return _mm256_movemask_ps(_mm256_cmp_ps(min, max, _CMP_LE_OQ)) == 0xFF; }
    inline smd8 clamp(smd8 x) const {return _mm256_max_ps(min, _mm256_min_ps(max, x));}
};


