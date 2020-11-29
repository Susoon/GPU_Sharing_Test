#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <curand_kernel.h>
#include <time.h>
#include <pthread.h>
#include "gpu_spec.h"

#define N 10000
#define BLOCK_SIZE 16
#define PART 4
#define TERM (N + BLOCK_SIZE * 11 - 1) / (BLOCK_SIZE - 11)

#define JOB PART

bool start_mult = false;
bool done[JOB] = {0};

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

__global__ void gpu_make_rand_matrix(int * mat, unsigned int seed){
	curandState s;
	curand_init(seed, 0, 0, &s); 

	for(int i = 0; i < N*N; i++){
		mat[i] = gpuRand(&s, 0, 1024); 
	}
}

void *thread_main(void * arg){
    unsigned int seed = *((unsigned int*)arg);

    clock_t start, end;

    // Cuda Event create
    float gpu_elapsed_time_ms;

    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

//    printf("Before Excution : %s\n", cudaGetErrorName(cudaGetLastError()));

    // Device matrices memory allocation
    int *device_mat1, *device_mat2, *device_result;
    cudaMalloc((void **) &device_mat1, sizeof(int)*N*N);
    cudaMalloc((void **) &device_mat2, sizeof(int)*N*N);
    cudaMalloc((void **) &device_result, sizeof(int)*N*N);

    // Fill device matrices 
    gpu_make_rand_matrix<<< 1, 1 >>>(device_mat1, seed);
    gpu_make_rand_matrix<<< 1, 1 >>>(device_mat2, seed);

    // Set the grid
    unsigned int grid_rows = 11;
    unsigned int gridevice_resultols = 11;

    dim3 dimGrid(gridevice_resultols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    while(!start_mult);

	//--------------------------------------GPU Computing Start--------------------------------------------//

    start = clock();   

    gpu_matrix_mult<<<dimGrid, dimBlock, 0, cuda_stream >>>(device_mat1, device_mat2, device_result);
    cudaDeviceSynchronize();

    end = clock();

    gpu_elapsed_time_ms = (float)(end - start) / (CLOCKS_PER_SEC / 1000);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", N, N, N, N, gpu_elapsed_time_ms);

	//--------------------------------------GPU Computing End--------------------------------------------//

//    printf("After Excution : %s\n", cudaGetErrorName(cudaGetLastError()));

    // free memory
    cudaFree(device_mat1);
    cudaFree(device_mat2);
    cudaFree(device_result);

    return 0;
}

int main(int argc, char const *argv[])
{
    unsigned int seed = 0;

    int status;
    int err;

    pthread_t threads[JOB];
    
    printf("%d Parts\n", PART);
	printf("THREAD PER JOB : %d, THREAD BLOCK PER JOB : %d\n", THREAD_PER_JOB, THREAD_BLOCK_PER_JOB);

    for(int i = 0; i < JOB; i++){
        done[i] = false;
        pthread_create(&threads[i], NULL, &thread_main, (void *)&seed);
        printf("%d / %d thread starts\n", i + 1, JOB);
    }

    start_mult = true;
    
    for(int i = 0; i < JOB; i++){
        err = pthread_join(threads[i], (void **)&status);
        if(err == 0){
            printf("Completed join with thread %d status : %d\n", i, status);
        }
        else{
            printf("ERROR: return code from pthread_join() is %d, thread %d\n", err, i);
            return -1;
        }
    }

    return 0;
}
