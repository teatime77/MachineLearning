#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define chk(x)	{ cudaError_t  sts = x; if(sts != cudaSuccess){ LogA("%s : Err %d", #x, sts); } }


void Log(wchar_t *szFormat, ...);
void LogA(char *szFormat, ...);
void Assert(bool ok, wchar_t* msg);

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

	return 0;
}

extern "C" __declspec(dllexport) int CudaDeviceReset(){
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	chk( cudaDeviceReset() );

	return 0;
}

extern "C" __declspec(dllexport) double* CudaAlloc(int len){
	double* dev;
	chk(cudaMalloc((void**)&dev, len * sizeof(double)));

	return dev;
}

extern "C" __declspec(dllexport) int CudaFree(double* dev){
	chk( cudaFree(dev) );
	return 0;
}

extern "C" __declspec(dllexport) int CudaToDev(double* dev, double* dt, int len){
	chk(cudaMemcpy(dev, dt, len * sizeof(double), cudaMemcpyHostToDevice));

	return 0;
}

extern "C" __declspec(dllexport) int CudaToHost(double* dev, double* dt, int len){
	chk( cudaMemcpy(dt, dev, len * sizeof(double), cudaMemcpyDeviceToHost) );
	return 0;
}

extern "C" __declspec(dllexport) int CudaSync(){
	chk(cudaDeviceSynchronize());
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
extern "C" __declspec(dllexport) int CudaAdd1(Array1 a, Array1 b, Array1 c){
	chk( cudaDeviceSynchronize() );

	addKernel1<<<1, c.Length>>>(a, b, c);

	chk(cudaGetLastError());
    chk( cudaDeviceSynchronize() );
    
    return 0;
}

extern "C" __declspec(dllexport) int CudaAdd2(Array2 a, Array2 b, Array2 c){
	chk( cudaDeviceSynchronize() );

	dim3 threadsPerBlock(a.nCol, a.nRow);
    addKernel2<<<1, threadsPerBlock>>>(a, b, c);

    chk( cudaGetLastError() );
    chk( cudaDeviceSynchronize() );
    
	return 0;
}

extern "C" __declspec(dllexport) int CudaAdd3(Array3 a, Array3 b, Array3 c){
	chk(cudaDeviceSynchronize());

	dim3 threadsPerBlock(a.nCol, a.nRow, a.nDepth);
	addKernel3<<<1, threadsPerBlock>>>(a, b, c);

	chk(cudaGetLastError());
	chk(cudaDeviceSynchronize());

	return 0;
}

extern "C" __declspec(dllexport) int CudaAdd4(Array4 a, Array4 b, Array4 c){
	chk(cudaDeviceSynchronize());

	dim3 threadsPerBlock(a.Dims[3], a.Dims[2], a.Dims[1]);
	addKernel4<<<a.Dims[0], threadsPerBlock>>>(a, b, c);

	chk(cudaGetLastError());
	chk(cudaDeviceSynchronize());

	return 0;
}

extern "C" __declspec(dllexport) int CudaDotSigmoid(Array2 a, Array2 b, Array2 c, Array2 d){
	chk( cudaDeviceSynchronize() );

	dim3 threadsPerBlock(c.nCol, c.nRow);
	dotSigmoidKernel<<<1, threadsPerBlock>>>(a, b, c, d);

	chk(cudaGetLastError());
	chk( cudaDeviceSynchronize() );

	return 0;
}

//-------------------------------------------------- 全結合レイヤー

/*
	Z2 = prev_A2.Dot(Weight) + Bias;
	Activation2 = Z2.Map(Sys.Sigmoid);
*/
__global__ void FullForwardKernel(Array2 prev_A2, Array2 Weight, Array1 Bias, Array2 z2, Array2 a2){
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (z2.nRow <= r || z2.nCol <= c){
		return;
	}

	double sum = dotKernel(prev_A2, Weight, r, c) + Bias.DevDt[c];

	z2.Set(r, c, sum);
	a2.Set(r, c, Sigmoid(sum));
}

extern "C" __declspec(dllexport) int FullForward(Array2 prev_A2, Array2 Weight, Array1 Bias, Array2 z2, Array2 a2){
	chk( cudaDeviceSynchronize() );

	dim3 threadsPerBlock;
	dim3 blocksPerGrid;

	if (z2.nCol * z2.nRow <= 1024){

		threadsPerBlock = dim3(z2.nCol, z2.nRow);
		blocksPerGrid = dim3(1, 1);
	}
	else{
		int col1, col2, row1, row2;

		if (1024 < z2.nCol){

			int col1 = 1024;
			int col2 = z2.nCol / col1;
			col2 += (col1 * col2 < z2.nCol ? 1 : 0);
		}
		else{

			col1 = z2.nCol;
			col2 = 1;
		}
		row1 = min(z2.nRow, 1024 / col1);
		row2 = z2.nRow / row1;
		row2 += (row1 * row2 < z2.nRow ? 1 : 0);

		threadsPerBlock = dim3(col1, row1);
		blocksPerGrid = dim3(col2, row2);

		Assert(col1 * row1 < 1024, L"Full-Forward");
		Assert(z2.nCol <= col1 * col2 && z2.nRow <= row1 * row2, L"Full-Forward");
	}

	FullForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(prev_A2, Weight, Bias, z2, a2);

	chk(cudaGetLastError());
	chk( cudaDeviceSynchronize() );

	return 0;
}

//-------------------------------------------------- 畳み込みレイヤー

/*
Z2 = prev_A2.Dot(Weight) + Bias;
Activation2 = Z2.Map(Sys.Sigmoid);
*/
__global__ void ConvolutionForwardKernel(Array3 prev_A3, Array3 weight3, Array1 bias, Array4 z4, Array4 a4){
	int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
	int r1		  = blockIdx.y * blockDim.y + threadIdx.y;
	int c1		  = blockIdx.x * blockDim.x + threadIdx.x;
	if (z4.Dims[0] <= batch_idx || z4.Dims[1] <= r1 || z4.Dims[2] <= c1){
		return;
	}

	int filter_count = weight3.nDepth;
	int filter_size  = weight3.nRow;

	double A[5][5];
	__shared__ double W[20][5][5];
	__shared__ double B[20];

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){

		for (int filter_idx = 0; filter_idx < filter_count; filter_idx++) {

			B[filter_idx] = bias.DevDt[filter_idx];

			// フィルターの行に対し
			for (int r2 = 0; r2 < filter_size; r2++) {

				// フィルターの列に対し
				for (int c2 = 0; c2 < filter_size; c2++) {

					W[filter_idx][r2][c2] = weight3.At(filter_idx, r2, c2);
				}
			}
		}
	}

	__syncthreads();

	// フィルターの行に対し
	for (int r2 = 0; r2 < filter_size; r2++) {

		// フィルターの列に対し
		for (int c2 = 0; c2 < filter_size; c2++) {

			A[r2][c2] = prev_A3.At(batch_idx, r1 + r2, c1 + c2);
		}
	}


	// すべてのフィルターに対し
	for (int filter_idx = 0; filter_idx < filter_count; filter_idx++) {

		double sum = 0.0;

		// フィルターの行に対し
		for (int r2 = 0; r2 < filter_size; r2++) {

			// フィルターの列に対し
			for (int c2 = 0; c2 < filter_size; c2++) {
				//sum += prev_A3.At(batch_idx, r1 + r2, c1 + c2) * weight3.At(filter_idx, r2, c2);
				sum += A[r2][c2] * W[filter_idx][r2][c2];
			}
		}

		// 出力
		//double z_val = sum + bias.DevDt[filter_idx];
		double z_val = sum + B[filter_idx];

		z4.Set(batch_idx, r1, c1, filter_idx, z_val);
		a4.Set(batch_idx, r1, c1, filter_idx, Sigmoid(z_val));
	}
}

extern "C" __declspec(dllexport) int ConvolutionForward(Array3 prev_A3, Array3 weight3, Array1 bias, Array4 z4, Array4 a4){
	chk(cudaDeviceSynchronize());

	dim3 threadsPerBlock;
	dim3 blocksPerGrid;

	int mini_batch_size = z4.Dims[0];
	int img_rows = z4.Dims[1];
	int img_cols = z4.Dims[2];
	int filter_count = weight3.nDepth;
	int filter_size = weight3.nRow;

	Assert(weight3.nRow == weight3.nCol, L"");
	Assert(filter_count * filter_size * filter_size < 1024, L"");

	int col1, col2, row1, row2, depth1, depth2;

	if (1024 < img_cols){

		int col1 = 1024;
		int col2 = img_cols / col1;
		col2 += (col1 * col2 < img_cols ? 1 : 0);
	}
	else{

		col1 = img_cols;
		col2 = 1;
	}

	row1 = min(img_rows, 1024 / col1);
	row2 = img_rows / row1;
	row2 += (row1 * row2 < img_rows ? 1 : 0);

	depth1 = min(mini_batch_size, 1024 / (row1 * col1));
	depth2 = mini_batch_size / depth1;
	depth2 += (depth1 * depth2 < mini_batch_size ? 1 : 0);

	Assert(depth1 * row1 * col1 < 1024, L"Conv-Forward");
	Assert(mini_batch_size <= depth1 * depth2 && img_rows <= row1 * row2 && img_cols <= col1 * col2, L"Conv-Forward");

	threadsPerBlock = dim3(col1, row1, depth1);
	blocksPerGrid = dim3(col2, row2, depth2);

	ConvolutionForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(prev_A3, weight3, bias, z4, a4);

	chk(cudaGetLastError());
	chk(cudaDeviceSynchronize());

	return 0;
}
