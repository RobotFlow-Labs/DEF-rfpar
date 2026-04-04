/**
 * DEF-rfpar: RL-based Pixel Adversarial Attack CUDA Kernels
 *
 * 1. parallel_pixel_sample — Sample and evaluate pixel perturbation candidates
 * 2. batch_reward_compute — Compute per-pixel rewards from detection confidence
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void parallel_pixel_sample_kernel(
    const float* __restrict__ image,
    float* __restrict__ candidates,
    float* __restrict__ perturbations,
    unsigned long long seed,
    int B, int C, int H, int W,
    int n_candidates, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * n_candidates) return;

    int b = idx / n_candidates;
    int cand = idx % n_candidates;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    // Random pixel position
    int h = (int)(curand_uniform(&state) * H) % H;
    int w = (int)(curand_uniform(&state) * W) % W;

    int out_base = (b * n_candidates + cand) * (C + 2);
    candidates[out_base + 0] = (float)h;
    candidates[out_base + 1] = (float)w;

    // Generate perturbation per channel
    for (int c = 0; c < C; c++) {
        float orig = image[b * C * H * W + c * H * W + h * W + w];
        float pert = (curand_uniform(&state) * 2.0f - 1.0f) * eps;
        float new_val = fminf(fmaxf(orig + pert, 0.0f), 1.0f);
        perturbations[(b * n_candidates + cand) * C + c] = new_val;
    }
}

__global__ void batch_reward_kernel(
    const float* __restrict__ conf_before,
    const float* __restrict__ conf_after,
    float* __restrict__ rewards,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    rewards[idx] = conf_before[idx] - conf_after[idx];
}

torch::Tensor parallel_pixel_sample(
    torch::Tensor image, int n_candidates, float eps, int64_t seed
) {
    TORCH_CHECK(image.is_cuda(), "image must be CUDA");
    int B = image.size(0), C = image.size(1), H = image.size(2), W = image.size(3);

    auto candidates = torch::empty({B, n_candidates, C + 2}, image.options());
    auto perturbations = torch::empty({B, n_candidates, C}, image.options());

    int total = B * n_candidates;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    parallel_pixel_sample_kernel<<<blocks, threads>>>(
        image.data_ptr<float>(),
        candidates.data_ptr<float>(),
        perturbations.data_ptr<float>(),
        (unsigned long long)seed,
        B, C, H, W, n_candidates, eps
    );
    return perturbations;
}

torch::Tensor batch_reward(torch::Tensor conf_before, torch::Tensor conf_after) {
    TORCH_CHECK(conf_before.is_cuda(), "must be CUDA");
    auto rewards = torch::empty_like(conf_before);
    int N = conf_before.numel();
    batch_reward_kernel<<<(N+255)/256, 256>>>(
        conf_before.data_ptr<float>(), conf_after.data_ptr<float>(),
        rewards.data_ptr<float>(), N
    );
    return rewards;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("parallel_pixel_sample", &parallel_pixel_sample, "Parallel pixel perturbation sampling (CUDA)");
    m.def("batch_reward", &batch_reward, "Batch reward computation (CUDA)");
}
