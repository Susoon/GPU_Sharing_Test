#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <curand_kernel.h>
#include <time.h>
#include <pthread.h>
#include "gpu_spec.h"
#include "gpu_info.h"

#define N (1024 * 16)
#define BLOCK_SIZE 16
#define PART 4
#define GRID_SIZE 11

#define JOB PART

bool start_mult = false;
bool *d_sm;
bool *h_sm;

__global__ void clear_sm(bool * sm){
    for(int i = 0; i < MAX_SM; i++){
        sm[i] = false;
    }
}

__global__ void gpu_matrix_mult(int *a,int *b, int *c, bool *sm)
{
    if(threadIdx.x == 0 && threadIdx.y == 0)
        sm[__mysmid()] = true;

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

    //printf("Before Excution : %s\n", cudaGetErrorName(cudaGetLastError()));

    // Device matrices memory allocation
    int *device_mat1, *device_mat2, *device_result;
    cudaMalloc((void **) &device_mat1, sizeof(int)*N*N);
    cudaMalloc((void **) &device_mat2, sizeof(int)*N*N);
    cudaMalloc((void **) &device_result, sizeof(int)*N*N);

    // Fill device matrices 
    gpu_make_rand_matrix<<< 1, 1 >>>(device_mat1, seed);
    gpu_make_rand_matrix<<< 1, 1 >>>(device_mat2, seed);

    dim3 dimGrid(GRID_SIZE, GRID_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    while(!start_mult);

	//--------------------------------------GPU Computing Start--------------------------------------------//

    start = clock();   

    gpu_matrix_mult<<< dimGrid, dimBlock, 0, cuda_stream >>>(device_mat1, device_mat2, device_result, d_sm);
    cudaDeviceSynchronize();

    end = clock();

    gpu_elapsed_time_ms = (float)(end - start) / (CLOCKS_PER_SEC / 1000);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", N, N, N, N, gpu_elapsed_time_ms);

	//--------------------------------------GPU Computing End--------------------------------------------//

    printf("After Excution : %s\n", cudaGetErrorName(cudaGetLastError()));

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

    h_sm = (bool *)calloc(MAX_SM, sizeof(bool));
    cudaMalloc((void **) &d_sm, sizeof(bool)*MAX_SM);

    clear_sm<<< 1, 1 >>>(d_sm);

    for(int i = 0; i < JOB; i++){
        pthread_create(&threads[i], NULL, &thread_main, (void *)&seed);
        printf("%d / %d thread starts\n", i + 1, JOB);
    }

    start_mult = true;
    
    for(int i = 0; i < JOB; i++){
        err = pthread_join(threads[i], (void **)&status);
        if(err == 0){
            printf("Completed join with thread %d status : %d\n", i + 1, status);
        }
        else{
            printf("ERROR: return code from pthread_join() is %d, thread %d\n", err, i + 1);
            return -1;
        }
    }

    cudaMemcpy(h_sm, d_sm, sizeof(bool)*MAX_SM, cudaMemcpyDeviceToHost);

    int count = 0;
    for(int i = 0; i < MAX_SM; i++){
        if(h_sm[i] == false){
            //printf("%dth sm is not used!!!\n", i + 1);
            count++;
        }
    }

    printf("Unused SM : %d\n", count);

    return 0;
}
