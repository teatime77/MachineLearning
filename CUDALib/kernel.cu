
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

struct ArrayN {
	double* dt;
	double* DevDt;
	int Length;

	void Alloc(){
		cudaError_t  cudaStatus = cudaMalloc((void**)&DevDt, Length * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw -1;
		}
	}

	void Free(){
		cudaFree(DevDt);
	}

	void ToDev(){
		if (DevDt == NULL){
			Alloc();
		}

		cudaError_t cudaStatus = cudaMemcpy(DevDt, dt, Length * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw - 1;
		}
	}

	void ToHost(){
		cudaError_t cudaStatus = cudaMemcpy(dt, DevDt, Length * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			throw - 1;
		}
	}
};

struct Array1 : ArrayN {
};

struct Array2 : ArrayN {
	int nRow;
	int nCol;
};

__device__ __forceinline__ double At2(Array2 a, int i, int j){
	return a.DevDt[ i * a.nCol + j ];
}

__device__ __forceinline__ void Set2(Array2 a, int i, int j, double d){
	a.DevDt[i * a.nCol + j] = d;
}

__global__ void addKernel1(const Array1 a, const Array1 b, Array1 c){
    int i = threadIdx.x;
	c.DevDt[i] = a.DevDt[i] + b.DevDt[i];
}

__global__ void addKernel2(Array2 a, const Array2 b, const Array2 c){
	int i = threadIdx.y * a.nCol + threadIdx.x;
	c.DevDt[i] = a.DevDt[i] + b.DevDt[i];
}

__device__ double dotKernel(Array2 a, const Array2 b, int i, int j){
	double sum = 0;

	for (int k = 0; k < a.nCol; k++){
		sum += At2(a, i, k) * At2(b, k, j);
	}

	return sum;
}


__device__ __forceinline__ double Sigmoid(double z){
	return 1.0 / (1.0 + exp(-z));
}

__global__ void dotSigmoidKernel(Array2 a, Array2 b, Array2 c, Array2 d){
	int i = threadIdx.y;
	int j = threadIdx.x;

	double sum = dotKernel(a, b, i, j);

	Set2(c, i, j, sum);
	Set2(d, i, j, Sigmoid(sum));
}

extern "C" __declspec(dllexport) int CUDASetDevice(int device){
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return -1;
	}

	return (int)cudaSuccess;
}

extern "C" __declspec(dllexport) int CUDADeviceReset(){
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return -1;
	}

	return (int)cudaSuccess;
}

// Helper function for using CUDA to add vectors in parallel.
extern "C" __declspec(dllexport) int AddCuda1(Array1 a, Array1 b, Array1 c)
{
    cudaError_t cudaStatus;

	a.ToDev();
	b.ToDev();
	c.Alloc();
	cudaDeviceSynchronize();

    // Launch a kernel on the GPU with one thread for each element.
    addKernel1<<<1, c.Length>>>(a, b, c);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
	c.ToHost();
	cudaDeviceSynchronize();

Error:
	c.Free();
	a.Free();
	b.Free();
    
    return (int)cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
extern "C" __declspec(dllexport) int AddCuda2(Array2 a, Array2 b, Array2 c)
{
    cudaError_t cudaStatus;

	a.ToDev();
	b.ToDev();
	c.Alloc();
	cudaDeviceSynchronize();

    // Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(a.nCol, a.nRow);
    addKernel2<<<1, threadsPerBlock>>>(a, b, c);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
	c.ToHost();
	cudaDeviceSynchronize();

Error:
	c.Free();
	a.Free();
	b.Free();
    
    return (int)cudaStatus;
}

extern "C" __declspec(dllexport) int CudaDotSigmoid(Array2 a, Array2 b, Array2 c, Array2 d)
{
	cudaError_t cudaStatus;

	a.ToDev();
	b.ToDev();
	c.Alloc();
	d.Alloc();
	cudaDeviceSynchronize();

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(c.nCol, c.nRow);
	dotSigmoidKernel<<<1, threadsPerBlock >>>(a, b, c, d);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	c.ToHost();
	d.ToHost();
	cudaDeviceSynchronize();

Error:
	a.Free();
	b.Free();
	c.Free();
	d.Free();

	return (int)cudaStatus;
}
