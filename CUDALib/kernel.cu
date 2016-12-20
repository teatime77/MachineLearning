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
	int	Dims[3];
};

struct Array4 : ArrayN {
	int	Dims[4];
};

__device__ __forceinline__ double At2(Array2 a, int i, int j){
	return a.DevDt[ i * a.nCol + j ];
}

__device__ __forceinline__ void Set2(Array2 a, int i, int j, double d){
	a.DevDt[i * a.nCol + j] = d;
}

__global__ void addKernel1(Array1 a, Array1 b, Array1 c){
    int i = threadIdx.x;
	c.DevDt[i] = a.DevDt[i] + b.DevDt[i];
}

__global__ void addKernel2(Array2 a, Array2 b, Array2 c){
	int i = threadIdx.y * a.nCol + threadIdx.x;
	c.DevDt[i] = a.DevDt[i] + b.DevDt[i];
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

// Helper function for using CUDA to add vectors in parallel.
extern "C" __declspec(dllexport) int CudaAdd2(Array2 a, Array2 b, Array2 c)
{
    cudaError_t sts;

	chk( cudaDeviceSynchronize() );

    // Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(a.nCol, a.nRow);
    addKernel2<<<1, threadsPerBlock>>>(a, b, c);

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
