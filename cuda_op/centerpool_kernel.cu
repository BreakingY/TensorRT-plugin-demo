#include <cuda.h>
#include <cuda_runtime.h>

__global__ void centerpool_kernel(const float *input, float *output, int C, int H, int W){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float sum = 0;
    for (int h = 0; h < H; h++)
    {
        for (int w = 0; w < W; w++)
        {
            int index = c * H * W + h * W + w;
            float value = input[index];
            float weight = (float)(h + w + 1) / (float)(H + W);
            sum += value * weight;
        }
    }

    output[c] = sum;
}
// demo演示，没有增加对batch的处理
void launch_centerpool(float *input, float *output, int C, int H, int W)
{

    int threads = 256;

    int blocks = (C + threads - 1) / threads;

    centerpool_kernel<<<blocks, threads>>>(input, output, C, H, W);
}
