#pragma once
#include <cuda_runtime.h>

constexpr int BR_DEFAULT = 32;
constexpr int BC_DEFAULT = 32;

void flash_attn_forward(
    const float* Q, const float* K, const float* V,
    float* O, float* LSE,
    int B, int H, int N, int d,
    float sm_scale, cudaStream_t stream);

void flash_attn_backward(
    const float* dO, const float* Q, const float* K, const float* V,
    const float* O, const float* LSE,
    float* dQ, float* dK, float* dV,
    int B, int H, int N, int d,
    float sm_scale, cudaStream_t stream);

void flash_attn_forward_v2(
    const float* Q, const float* K, const float* V,
    float* O, float* LSE,
    int B, int H, int N, int d,
    float sm_scale, cudaStream_t stream);

void flash_attn_backward_v2(
    const float* dO, const float* Q, const float* K, const float* V,
    const float* O, const float* LSE,
    float* dQ, float* dK, float* dV,
    int B, int H, int N, int d,
    float sm_scale, cudaStream_t stream);
