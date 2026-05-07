#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "flash_attn.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")

// ---------------------------------------------------------------------------
// Forward: returns (O, LSE)
//   O:   (B, H, N, d)
//   LSE: (B, H, N)  -- log-sum-exp, needed for backward recomputation
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> flash_attn_fwd_cpp(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float sm_scale
) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V);
    CHECK_FLOAT(Q); CHECK_FLOAT(K); CHECK_FLOAT(V);

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    auto O   = torch::zeros_like(Q);
    auto LSE = torch::empty({B, H, N}, Q.options());

    auto stream = at::cuda::getCurrentCUDAStream();

    flash_attn_forward(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        LSE.data_ptr<float>(),
        B, H, N, d,
        sm_scale,
        stream
    );

    return {O, LSE};
}

// ---------------------------------------------------------------------------
// Backward: returns (dQ, dK, dV)
//   Internally computes D = rowsum(dO * O) and launches bwd kernel
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> flash_attn_bwd_cpp(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor LSE,
    float sm_scale
) {
    CHECK_CUDA(dO); CHECK_CUDA(Q); CHECK_CUDA(K);
    CHECK_CUDA(V); CHECK_CUDA(O); CHECK_CUDA(LSE);
    CHECK_CONTIGUOUS(dO); CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K);
    CHECK_CONTIGUOUS(V); CHECK_CONTIGUOUS(O); CHECK_CONTIGUOUS(LSE);
    CHECK_FLOAT(dO); CHECK_FLOAT(Q); CHECK_FLOAT(K);
    CHECK_FLOAT(V); CHECK_FLOAT(O); CHECK_FLOAT(LSE);

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    auto stream = at::cuda::getCurrentCUDAStream();

    flash_attn_backward(
        dO.data_ptr<float>(),
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        LSE.data_ptr<float>(),
        dQ.data_ptr<float>(),
        dK.data_ptr<float>(),
        dV.data_ptr<float>(),
        B, H, N, d,
        sm_scale,
        stream
    );

    return {dQ, dK, dV};
}

std::vector<torch::Tensor> flash_attn_fwd_v2_cpp(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float sm_scale
) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V);
    CHECK_FLOAT(Q); CHECK_FLOAT(K); CHECK_FLOAT(V);

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    auto O   = torch::zeros_like(Q);
    auto LSE = torch::empty({B, H, N}, Q.options());

    auto stream = at::cuda::getCurrentCUDAStream();

    flash_attn_forward_v2(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        LSE.data_ptr<float>(),
        B, H, N, d,
        sm_scale,
        stream
    );

    return {O, LSE};
}

std::vector<torch::Tensor> flash_attn_bwd_v2_cpp(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor LSE,
    float sm_scale
) {
    CHECK_CUDA(dO); CHECK_CUDA(Q); CHECK_CUDA(K);
    CHECK_CUDA(V); CHECK_CUDA(O); CHECK_CUDA(LSE);
    CHECK_CONTIGUOUS(dO); CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K);
    CHECK_CONTIGUOUS(V); CHECK_CONTIGUOUS(O); CHECK_CONTIGUOUS(LSE);
    CHECK_FLOAT(dO); CHECK_FLOAT(Q); CHECK_FLOAT(K);
    CHECK_FLOAT(V); CHECK_FLOAT(O); CHECK_FLOAT(LSE);

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    auto stream = at::cuda::getCurrentCUDAStream();

    flash_attn_backward_v2(
        dO.data_ptr<float>(),
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        LSE.data_ptr<float>(),
        dQ.data_ptr<float>(),
        dK.data_ptr<float>(),
        dV.data_ptr<float>(),
        B, H, N, d,
        sm_scale,
        stream
    );

    return {dQ, dK, dV};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_fwd_cpp,
          "FlashAttention-1 forward -> (O, LSE)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("sm_scale") = -1.0f);
    m.def("backward", &flash_attn_bwd_cpp,
          "FlashAttention-1 backward -> (dQ, dK, dV)",
          py::arg("dO"), py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("O"), py::arg("LSE"), py::arg("sm_scale") = -1.0f);
    m.def("forward_v2", &flash_attn_fwd_v2_cpp,
          "FlashAttention-2 forward -> (O, LSE)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("sm_scale") = -1.0f);
    m.def("backward_v2", &flash_attn_bwd_v2_cpp,
          "FlashAttention-2 backward -> (dQ, dK, dV)",
          py::arg("dO"), py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("O"), py::arg("LSE"), py::arg("sm_scale") = -1.0f);
}
