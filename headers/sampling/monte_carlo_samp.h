#pragma once
#include "general.h"

inline smd8 power_heuristic(smd8 pdf_a, smd8 pdf_b) {
    smd8 a = _mm256_mul_ps(pdf_a, pdf_a);
    smd8 sum = _mm256_fmadd_ps(pdf_b, pdf_b, a);
    return precise_div(a, sum);
}

inline smd8 power_heuristic(int nf, smd8 pdf_a, int ng, smd8 pdf_b) {
    smd8 f_nf = _mm256_set1_ps(static_cast<float>(nf));
    smd8 f_ng = _mm256_set1_ps(static_cast<float>(ng));

    smd8 scaled_a = _mm256_mul_ps(f_nf, pdf_a);
    smd8 scaled_b = _mm256_mul_ps(f_ng, pdf_b);

    return power_heuristic(scaled_a, scaled_b);
}
