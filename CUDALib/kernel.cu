
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

__global__ void addKernel(Array1 c, const Array1 a, const Array1 b)
{
    int i = threadIdx.x;
	c.DevDt[i] = a.DevDt[i] + b.DevDt[i];
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
extern "C" __declspec(dllexport) int addWithCuda(Array1 a, Array1 b, Array1 c)
{
    cudaError_t cudaStatus;

	a.ToDev();
	b.ToDev();
	c.Alloc();
	cudaDeviceSynchronize();

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, c.Length>>>(c, a, b);

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
