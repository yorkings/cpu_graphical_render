# SIMD RTIOW + PBR  combination Ray Tracing Engine

A step-by-step C++ ray tracing engine built from first principles, with SIMD acceleration and physically based rendering techniques.

- i have kept the image in their raw Portable **Pixel Map** (**.ppm**)

## Goals

- Build a minimal ray tracer incrementally, from camera rays to a complete renderer.
- Use SIMD-friendly data layouts and vector math for parallel ray processing.
- Add physically based materials, lighting, textures, and importance sampling.
- Keep each stage testable, measurable, and easy to compare against a scalar reference.

## Roadmap

1. **Foundations** — vectors, points, rays, intervals, colors, and image output.
2. **Geometry** — spheres, planes, triangles, transforms, and surface intersections.
3. **Acceleration** — bounding volumes and a BVH for efficient scene traversal.
4. **SIMD core** — portable SIMD abstractions, packet rays, masks, and vectorized intersection tests.
5. **Materials** — diffuse, metallic, dielectric, emissive, and PBR parameter workflows.
6. **Lighting** — direct lighting, multiple importance sampling, shadows, and environment lighting.
7. **Production features** — textures, depth of field, motion blur, denoising, and progressive rendering.
8. **Profiling** — scalar/SIMD benchmarks, cache analysis, image-difference tests, and regression checks.

## Design Principles

- Preserve a clear scalar implementation as a correctness reference.
- Prefer structure-of-arrays layouts for coherent SIMD traversal.
- Separate rendering policy from geometry and platform-specific SIMD code.
- Validate every milestone with deterministic tests and rendered reference images.
- Optimize only after profiling; correctness and reproducibility come first.

## Status

This project is being developed incrementally. Each milestone will document its implementation, tests, example renders, and performance results.


### Requirements
---
Before we proceed ensure you already have cmake installed in your pc


1. install cmake  then create build directory 
 ```
 mkdir build
 ```
2. run this command
```
 cd build  &&
```

3. run this command 
```bash
 cmake --build build/.
 ```

4. search the executable **GraphicsRenderer** and run it.
``` bash
./GraphicsRenderer
```
## sampling and optimization
Russian Roulette and the Power Heuristic cannot be compared as "better" or "worse" because they solve two completely different problems in Monte Carlo path tracing. 

Russian Roulette is a path-length control technique designed to cut compute cycles, whereas the Power Heuristic is a weighting formula used in Multiple Importance Sampling (MIS) designed to reduce image noise