
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

struct ArrayN {

};

struct Array1 : ArrayN {
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

__global__ void addKernel(Array1 c, const Array1 a, const Array1 b)
{
    int i = threadIdx.x;
	c.DevDt[i] = a.DevDt[i] + b.DevDt[i];
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(Array1 c, Array1 a, Array1 b, unsigned int size)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	// Allocate GPU buffers for three vectors (two input, one output)    .
	c.Alloc();
	a.Alloc();
	b.Alloc();

	// Copy input vectors from host memory to GPU buffers.
	a.ToDev();
	b.ToDev();
	

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(c, a, b);

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

Error:
	c.Free();
	a.Free();
	b.Free();
    
    return cudaStatus;
}

extern "C" __declspec(dllexport) int CUDAmain(Array1 a, Array1 b, Array1 c)
{
	const int arraySize = 5;
	//const double a[arraySize] = { 1, 2, 3, 4, 5 };
	//const double b[arraySize] = { 10, 20, 30, 40, 50 };
	//double c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",
	//    c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
