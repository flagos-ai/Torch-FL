// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*

mxcc -o test_cuda_api test_cuda_api.c \
  -I/opt/maca-3.3.0/tools/cu-bridge/include \
  -L/opt/maca-3.3.0/lib \
  -lruntime_cu

export LD_LIBRARY_PATH=/opt/maca-3.3.0/lib:$LD_LIBRARY_PATH

./test_cuda_api

*/


#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // 1. Allocate MetaX NPU memory (cudaMalloc)
    float* d_data;
    cudaError_t err = cudaMalloc(&d_data, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 2. Create MetaX NPU stream (cudaStreamCreate)
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return -1;
    }

    // 3. Host -> MetaX NPU data copy (cudaMemcpyAsync)
    float h_data[1024] = {1.0f};
    err = cudaMemcpyAsync(d_data, h_data, 1024 * sizeof(float), 
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        printf("cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return -1;
    }

    // 4. Synchronize stream + release resources
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    printf("All operations on MetaX NPU success!\n");
    return 0;
}
