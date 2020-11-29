#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <curand_kernel.h>
#include <time.h>
#include <pthread.h>
#include "gpu_spec.h"

#define N 3000
#define BLOCK_SIZE 16
#define JOB 1
#define PART 4
#define TERM (N + BLOCK_SIZE * 11 - 1) / (BLOCK_SIZE - 11)

#define CPU 0

__global__ void gpu_matrix_mult(int *a,int *b, int *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    while(row < N){
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        while(col < N){ 
            int sum = 0;
            if( col < N && row < N) 
            {
                for(int i = 0; i < N; i++) 
                {
                    sum += a[row * N + i] * b[i * N + col];
                }
                c[row * N + col] = sum;
            }
            col += 11 * BLOCK_SIZE;
        }
        row += 11 * BLOCK_SIZE;
    }
} 

__device__ int gpuRand(curandState *s, int A, int B){
	float rand_int = curand_uniform(s);
	rand_int = rand_int * (B-A) + A;

	return rand_int;
}

__global__ void gpu_make_rand_matrix(int * mat){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int seed = id;
	curandState s;
	curand_init(seed, 0, 0, &s); 

	for(int i = 0; i < N*N; i++){
		mat[i] = gpuRand(&s, 0, 1024); 
	}
}

void cpu_make_rand_matrix(int * mat){
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            mat[i * N + j] = rand() % 100;
        }
    }
}

void cpu_matrix_mult(int *host_mat1, int *host_mat2, int *h_result) {
    for (int i = 0; i < N; ++i) 
    {
        for (int j = 0; j < N; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < N; ++h) 
            {
                tmp += host_mat1[i * N + h] * host_mat2[h * N + j];
            }
            h_result[i * N + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{

    srand(time(NULL));

    // Host matrices memory allocation
    int *host_mat1, *host_mat2, *host_result, *device_result_in_host;
    cudaMallocHost((void **) &host_mat1, sizeof(int)*N*N);
    cudaMallocHost((void **) &host_mat2, sizeof(int)*N*N);
    cudaMallocHost((void **) &host_result, sizeof(int)*N*N);
    cudaMallocHost((void **) &device_result_in_host, sizeof(int)*N*N);

    // Fill host matrices
    cpu_make_rand_matrix(host_mat1);
    cpu_make_rand_matrix(host_mat2);

    // Cuda Event create
    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Device matrices memory allocation
    int *device_mat1, *device_mat2, *device_result;
    cudaMalloc((void **) &device_mat1, sizeof(int)*N*N);
    cudaMalloc((void **) &device_mat2, sizeof(int)*N*N);
    cudaMalloc((void **) &device_result, sizeof(int)*N*N);

    // Fill device matrices with host matrices
    cudaMemcpy(device_mat1, host_mat1, sizeof(int)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mat2, host_mat2, sizeof(int)*N*N, cudaMemcpyHostToDevice);

    // Set the grid
    unsigned int grid_rows = 11;
    unsigned int gridevice_resultols = 11;

    dim3 dimGrid(gridevice_resultols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	printf("THREAD PER JOB : %d, THREAD BLOCK PER JOB : %d\n", THREAD_PER_JOB, THREAD_BLOCK_PER_JOB);

	//--------------------------------------GPU Computing Start--------------------------------------------//

    cudaEventRecord(start, 0);
   
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(device_mat1, device_mat2, device_result);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaMemcpy(host_result, device_result, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", N, N, N, N, gpu_elapsed_time_ms);

	//--------------------------------------GPU Computing End--------------------------------------------//

#if CPU

	//--------------------------------------CPU Computing Start--------------------------------------------//
    cudaEventRecord(start, 0);

    cpu_matrix_mult(host_mat1, host_mat2, device_result_in_host);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", N, N, N, N, cpu_elapsed_time_ms);

	//--------------------------------------CPU Computing End--------------------------------------------//

    int all_ok = 0;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if(device_result_in_host[i*N + j] != host_result[i*N + j])
            {
                all_ok++;
            }
        }
    }

    if(!all_ok)
    {
        printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("%d Incorrect results\n", all_ok);
    }
#endif

    // free memory
    cudaFree(device_mat1);
    cudaFree(device_mat2);
    cudaFree(device_result);
    cudaFreeHost(host_mat1);
    cudaFreeHost(host_mat2);
    cudaFreeHost(host_result);
    cudaFreeHost(device_result_in_host);
    return 0;
}
