#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define chk(x)	{ cudaError_t  sts = x; if(sts != cudaSuccess){ LogA("%s : Err %d", #x, sts); } }


void Log(wchar_t *szFormat, ...);
void LogA(char *szFormat, ...);

struct ArrayN {
	double* dt;
	double* DevDt;
	int Length;
};

struct Array1 : ArrayN {
};

struct Array2 : ArrayN {
	int nRow;
	int nCol;

	__device__ __forceinline__ double At(int i, int j){
		return DevDt[i * nCol + j];
	}

	__device__ __forceinline__ void Set(int i, int j, double d){
		DevDt[i * nCol + j] = d;
	}
};

struct Array3 : ArrayN {
	int nDepth;
	int nRow;
	int nCol;
	int nRowCol;

	__device__ __forceinline__ double At(int i, int j, int k){
		return DevDt[i * nRowCol + j * nCol + k];
	}

	__device__ __forceinline__ void Set(int i, int j, int k, double d){
		DevDt[i * nRowCol + j * nCol + k] = d;
	}
};

struct Array4 : ArrayN {
	int Dims[4];
	int Sizes[3];

	__device__ __forceinline__ double At(int i, int j, int k, int l){
		return DevDt[i * Sizes[0] + j * Sizes[1] + k * Sizes[2] + l];
	}

	__device__ __forceinline__ void Set(int i, int j, int k, int l, double d){
		DevDt[i * Sizes[0] + j * Sizes[1] + k * Sizes[2] + l] = d;
	}
};

__global__ void addKernel1(Array1 a, Array1 b, Array1 c){
    int i = threadIdx.x;
	c.DevDt[i] = a.DevDt[i] + b.DevDt[i];
}

__global__ void addKernel2(Array2 a, Array2 b, Array2 c){
	int i = threadIdx.y;
	int j = threadIdx.x;

	c.Set(i, j , a.At(i, j) + b.At(i, j));
}

__global__ void addKernel3(Array3 a, Array3 b, Array3 c){
	int i = threadIdx.z;
	int j = threadIdx.y;
	int k = threadIdx.x;

	c.Set(i, j, k, a.At(i, j, k) + b.At(i, j, k));
}

__global__ void addKernel4(Array4 a, Array4 b, Array4 c){
	int i = blockIdx.x;
	int j = threadIdx.z;
	int k = threadIdx.y;
	int l = threadIdx.x;

	c.Set(i, j, k, l, a.At(i, j, k, l) + b.At(i, j, k, l));
}

__device__ double dotKernel(Array2 a, Array2 b, int i, int j){
	double sum = 0;

	for (int k = 0; k < a.nCol; k++){
		sum += a.At(i, k) * b.At(k, j);
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

	c.Set(i, j, sum);
	d.Set(i, j, Sigmoid(sum));
}

extern "C" __declspec(dllexport) int CudaSetDevice(int device){
	// Choose which GPU to run on, change this on a multi-GPU system.
	chk( cudaSetDevice(device) );

	return (int)cudaSuccess;
}

extern "C" __declspec(dllexport) int CudaDeviceReset(){
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	chk( cudaDeviceReset() );

	return (int)cudaSuccess;
}

extern "C" __declspec(dllexport) double* CudaAlloc(int len){
	double* dev;
	chk(cudaMalloc((void**)&dev, len * sizeof(double)));

	return dev;
}

extern "C" __declspec(dllexport) int CudaFree(double* dev){
	chk( cudaFree(dev) );
	return (int)cudaSuccess;
}

extern "C" __declspec(dllexport) int CudaToDev(double* dev, double* dt, int len){
	chk(cudaMemcpy(dev, dt, len * sizeof(double), cudaMemcpyHostToDevice));

	return (int)cudaSuccess;
}

extern "C" __declspec(dllexport) int CudaToHost(double* dev, double* dt, int len){
	chk( cudaMemcpy(dt, dev, len * sizeof(double), cudaMemcpyDeviceToHost) );
	return (int)cudaSuccess;
}

extern "C" __declspec(dllexport) int CudaSync(){
	chk(cudaDeviceSynchronize());
	return (int)cudaSuccess;
}

// Helper function for using CUDA to add vectors in parallel.
extern "C" __declspec(dllexport) int CudaAdd1(Array1 a, Array1 b, Array1 c)
{
    cudaError_t sts;

	chk( cudaDeviceSynchronize() );

    // Launch a kernel on the GPU with one thread for each element.
	addKernel1<<<1, c.Length>>>(a, b, c);

    // Check for any errors launching the kernel
    sts = cudaGetLastError();
    if (sts != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(sts));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    chk( cudaDeviceSynchronize() );

Error:
    
    return (int)sts;
}

extern "C" __declspec(dllexport) int CudaAdd2(Array2 a, Array2 b, Array2 c){
	chk( cudaDeviceSynchronize() );

	dim3 threadsPerBlock(a.nCol, a.nRow);
    addKernel2<<<1, threadsPerBlock>>>(a, b, c);

    chk( cudaGetLastError() );
    chk( cudaDeviceSynchronize() );
    
	return (int)cudaSuccess;
}

extern "C" __declspec(dllexport) int CudaAdd3(Array3 a, Array3 b, Array3 c){
	chk(cudaDeviceSynchronize());

	dim3 threadsPerBlock(a.nCol, a.nRow, a.nDepth);
	addKernel3<<<1, threadsPerBlock>>>(a, b, c);

	chk(cudaGetLastError());
	chk(cudaDeviceSynchronize());

	return (int)cudaSuccess;
}

extern "C" __declspec(dllexport) int CudaAdd4(Array4 a, Array4 b, Array4 c){
	chk(cudaDeviceSynchronize());

	dim3 threadsPerBlock(a.Dims[3], a.Dims[2], a.Dims[1]);
	addKernel4<<<a.Dims[0], threadsPerBlock>>>(a, b, c);

	chk(cudaGetLastError());
	chk(cudaDeviceSynchronize());

	return (int)cudaSuccess;
}

extern "C" __declspec(dllexport) int CudaDotSigmoid(Array2 a, Array2 b, Array2 c, Array2 d)
{
	cudaError_t sts;

	chk( cudaDeviceSynchronize() );

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(c.nCol, c.nRow);
	dotSigmoidKernel<<<1, threadsPerBlock >>>(a, b, c, d);

	// Check for any errors launching the kernel
	sts = cudaGetLastError();
	if (sts != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(sts));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	chk( cudaDeviceSynchronize() );

Error:

	return (int)sts;
}
