#include <iostream>
#include <cuda_runtime.h>

// Matrix multiplication kernel
__global__ void matmul(float* A, float* B, float* C, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Print a message from the GPU kernel (for demonstration)
    if (row == 0 && col == 0) 
    {
        printf("Hello from GPU kernel\n");
    }

    float sum = 0.0f;
    if (row < N && col < N) 
    {
        for (int k = 0; k < N; k++) 
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Main function
int main() 
{
    int N = 1024;

    // Bytes of memory to allocate for each matrix
    int size = N * N * sizeof(float);

    // float h_A[16] = {
    //     1, 2, 3, 4,
    //     5, 6, 7, 8,
    //     9, 10, 11, 12,
    //     13, 14, 15, 16
    // };

    // float h_B[16] = {
    //     1, 0, 0, 0,
    //     0, 1, 0, 0,
    //     0, 0, 1, 0,
    //     0, 0, 0, 1
    // };

    // to store the result of A * B for CPU
    // float h_C[16] = {0};

    // Allocate host memory (CPU)
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    // Initialize matrices A and B with some values
    for (int i = 0; i < N * N; i++) 
    {
        h_A[i] = i % 10;   // values: 0,1,2,...9 repeating
        h_B[i] = (i + 1) % 10;
    }
    

    // Allocate memory on the GPU for these matrices
    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from CPU memory to GPU memory
    // cudaMemcpy(destination, source, size, direction)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and thread sizes
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    matmul<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // 🔹 Error check (launch)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        std::cout << "Launch Error: " << cudaGetErrorString(err) << std::endl;
    }

    // Wait for GPU to finish execution before accessing the result
    cudaDeviceSynchronize();


    // Error check (execution)
    err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        std::cout << "Execution Error: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy result from GPU memory to CPU memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print a small part (DON’T print full matrix)
    std::cout << "Top-left 5x5 of result:\n";
    for (int i = 0; i < 5; i++) 
    {
        for (int j = 0; j < 5; j++) 
        {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}