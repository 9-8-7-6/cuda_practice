#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorMul(float *M, float *N, float *P, int width) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row < width && col < width) {
		float sum = 0.0;
		for (int k=0;k<width;k++) {
			sum += M[row * width + k] * N[k * width + col];
		}
		P[row * width + col] = sum;
	}
}

int main(void){
	cudaError_t err = cudaSuccess;			
	int numElements = 500000;
	size_t size = numElements * sizeof(float);

	printf("Vector addition of %d elements\n", numElements);

	float *h_a = (float *)malloc(size);
	float *h_b = (float *)malloc(size);
	float *h_c = (float *)malloc(size);

	if(h_a == NULL || h_b == NULL || h_c == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	for(int i=0;i<numElements;i++) {
		h_a[i] = rand() / (float) RAND_MAX;
		h_b[i] = rand() / (float) RAND_MAX;
	}

	float *d_a = NULL;
	err = cudaMalloc((void**)&d_a, size);

	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector a (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_b = NULL;
	err = cudaMalloc((void **)&d_b, size);

	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector b (eror code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_c = NULL;
	err = cudaMalloc((void **)&d_c, size);

	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector c(error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector a from host to device, error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector b from host to device, error code %s)!\n", cudaGetErrorString(err));
	}

	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	vectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);

	err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	// verification??

	err = cudaFree(d_a);

	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector a(error code is %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_b);

	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector b(error code is %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_c);

	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector c(error code is %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	free(h_a);
	free(h_b);
	free(h_c);

	printf("Done\n");
	return 0;
}
