all: build

build: cudaMM

cudaMM: matrix_cuda.cu
	@nvcc $^ -o $@ -lpthread 

clean:
	@rm -rf cudaMM
