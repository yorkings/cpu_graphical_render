#pragma once
#include "general.h"

inline smd8 power_heustiric(smd8 pdf_a,smd8 pdf_b){
    smd8 a=_mm256_mul_ps(pdf_a,pdf_a);
    smd8 sum=_mm256_fmadd_ps(pdf_b,pdf_b,a);
    return precise_div(a,sum);   
}
