#pragma once
#include "general.h"
struct vec2 {
    float x, y;

    inline vec2() = default; 
    inline constexpr vec2(float x, float y) : x(x), y(y) {}

    inline float u() const { return x; }
    inline float v() const { return y; }
    inline float& u() { return x; }
    inline float& v() { return y; }

    inline float operator[](int i) const { return (&x)[i]; }
    inline float& operator[](int i)       { return (&x)[i]; }

    inline vec2 operator-() const { return {-x, -y}; }
    inline vec2& operator+=(vec2 b) { x += b.x; y += b.y; return *this; }
    inline vec2& operator-=(vec2 b) { x -= b.x; y -= b.y; return *this; }
    inline vec2& operator*=(vec2 b) { x *= b.x; y *= b.y; return *this; }
    inline vec2& operator*=(float t) { x *= t; y *= t; return *this; }
    inline vec2& operator/=(float t) { float inv = 1.0f / t; x *= inv; y *= inv; return *this; }
};


inline vec2 operator+(vec2 a, vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline vec2 operator-(vec2 a, vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline vec2 operator*(vec2 a, vec2 b) { return {a.x * b.x, a.y * b.y}; }
inline vec2 operator*(float t, vec2 a) { return {a.x * t, a.y * t}; }
inline vec2 operator*(vec2 a, float t) { return {a.x * t, a.y * t}; }
inline vec2 operator/(vec2 a, float t) { float inv = 1.0f / t; return {a.x * inv, a.y * inv}; }


inline float dot(vec2 a, vec2 b) { return a.x * b.x + a.y * b.y; }
inline float length_squared(vec2 v) { return dot(v, v); }