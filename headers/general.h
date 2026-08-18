#pragma once
#include<mdspan>
#include <iostream>

#include <iomanip>
#include <immintrin.h>

#include <limits>
#include <memory>

#include <string>
#include <fstream>

#include <ctime>
#include <omp.h>
#include <chrono>

#include<algorithm>
#include <random> 
#include<cmath>
#include<cstdlib>

#include <atomic>
#include<vector>
#include <sstream>

using std::make_shared;
using std::shared_ptr;

using smd4=__m128;
using smd4i=__m128i;
using smd8=__m256;
using smd8i=__m256i;

constexpr float infinity =std::numeric_limits<float>::infinity();
constexpr float pi =3.14159265358979323846f;

inline std::string get_time_current();

inline smd8 precise_div(smd8 num, smd8 den) {
    smd8 eps        = _mm256_set1_ps(1e-9f);
    smd8 sign_bit   = _mm256_set1_ps(-0.0f);
    // Absolute value check to guard zero/near-zero denominators
    smd8 abs_den    = _mm256_andnot_ps(sign_bit, den);
    smd8 is_zero    = _mm256_cmp_ps(abs_den, eps, _CMP_LT_OQ);
    smd8 val_sign   = _mm256_and_ps(den, sign_bit);
    smd8 signed_eps = _mm256_or_ps(eps, val_sign);
    smd8 safe_den   = _mm256_blendv_ps(den, signed_eps, is_zero);
    // Initial 11-bit reciprocal estimate
    smd8 x0  = _mm256_rcp_ps(safe_den); 
    smd8 one = _mm256_set1_ps(1.0f);
    // FMA3 Newton-Raphson Refinement: x1 = x0 + x0 * (1.0 - safe_den * x0)
    smd8 e   = _mm256_fnmadd_ps(safe_den, x0, one); // e = 1.0 - safe_den * x0   
    smd8 x1  = _mm256_fmadd_ps(x0, e, x0);// Refined reciprocal (23-bit accuracy)  
    return _mm256_mul_ps(num, x1);
}

// Precise 23-bit inverse vector 
inline smd8 precise_inv(smd8 dir) {
    return precise_div(_mm256_set1_ps(1.0f), dir);
}