/*
 * LightSpMVCore.h
 *
 *  Created on: May 29, 2015
 *      Author: Yongchao Liu
 *      Affiliation: School of Computational Science & Engineering, Georgia Institute of Technology
 *      Email: yliu@cc.gatech.edu
 *      
 */

#ifndef LIGHTSPMVCORE_H_
#define LIGHTSPMVCORE_H_


#pragma once


#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <vector>
#include <iostream>
using namespace std;

/*program version*/
#define VERSION "v1.0.1"

/*macros for cuda array*/
#if !defined(SPMV_CUDA_ARRAY_WIDTH_SHIFT) || SPMV_CUDA_ARRAY_WIDTH_SHIFT < 10 || SPMV_CUDA_ARRAY_WIDTH_SHIFT > 16
#define SPMV_CUDA_ARRAY_WIDTH_SHIFT		15
#endif
#define SPMV_CUDA_ARRAY_WIDTH_MASK		((1 << SPMV_CUDA_ARRAY_WIDTH_SHIFT) - 1)
#define SPMV_CUDA_ARRAY_WIDTH 			(1 << SPMV_CUDA_ARRAY_WIDTH_SHIFT)

/*maximum number of threads per block*/
#define MAX_NUM_THREADS_PER_BLOCK			1024

/*error check*/
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError(const char* file, const int32_t line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		std::cerr << "cudaCheckError() failed at " << file << ":" << line << " : "
				<< cudaGetErrorString(err) << endl;
		exit(-1);
	}
}

namespace lightspmv {

/*device functions*/
template<typename T>
__device__   inline T shfl_down_64bits(T var, int32_t srcLane, int32_t width) {

	int2 a = *reinterpret_cast<int2*>(&var);

	/*exchange the data*/
	a.x = __shfl_down(a.x, srcLane, width);
	a.y = __shfl_down(a.y, srcLane, width);

	return *reinterpret_cast<T*>(&a);
}

/*macro to get the X value*/
__device__ inline float FLOAT_VECTOR_GET(const cudaTextureObject_t vectorX,
		uint32_t index) {
	return tex1Dfetch<float>(vectorX, index);
}
__device__ inline float FLOAT_VECTOR_GET(const float* __restrict vectorX,
		uint32_t index) {
	return vectorX[index];
}

__device__ inline double DOUBLE_VECTOR_GET(const cudaTextureObject_t vectorX,
		uint32_t index) {
	/*load the data*/
	int2 v = tex1Dfetch < int2 > (vectorX, index);

	/*convert to double*/
	return __hiloint2double(v.y, v.x);
}
__device__ inline double DOUBLE_VECTOR_GET(const double* __restrict vectorX,
		uint32_t index) {
	return vectorX[index];
}

/*32-bit*/
template<typename T, uint32_t THREADS_PER_VECTOR,
		uint32_t MAX_NUM_VECTORS_PER_BLOCK>
__global__ void csr32DynamicWarp(uint32_t* __restrict cudaRowCounter,
		const uint32_t cudaNumRows, const uint32_t cudaNumCols,
		const uint32_t* __restrict rowOffsets,
		const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const cudaTextureObject_t vectorX,
		T* vectorY) {
	uint32_t i;
	T sum;
	uint32_t row;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const uint32_t warpLaneId = threadIdx.x & 31; /*lane index in the warp*/
	const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR; /*vector index in the warp*/

	__shared__   volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp and compute the row index of each vector*/
	row = __shfl(row, 0) + warpVectorId;

	/*check the row range*/
	while (row < cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}

			/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}
		/*intra-vector reduction*/
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += __shfl_down(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum;
		}

		/*get a new row index*/
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the row index to the other threads in the same warp and compute the row index of each vetor*/
		row = __shfl(row, 0) + warpVectorId;

	}/*while*/
}
//with traditional arrays
template<typename T, uint32_t THREADS_PER_VECTOR,
		uint32_t MAX_NUM_VECTORS_PER_BLOCK>
__global__ void csr32DynamicWarp(uint32_t* __restrict cudaRowCounter,
		const uint32_t cudaNumRows, const uint32_t cudaNumCols,
		const uint32_t* __restrict rowOffsets,
		const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const T* __restrict vectorX,
		T* vectorY) {
	uint32_t i;
	T sum;
	uint32_t row;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const uint32_t warpLaneId = threadIdx.x & 31; /*lane index in the warp*/
	const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR; /*vector index in the warp*/

	__shared__   volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp and compute the row index of each vector*/
	row = __shfl(row, 0) + warpVectorId;

	/*check the row range*/
	while (row < cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}

			/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}
		/*intra-vector reduction*/
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += __shfl_down(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum;
		}

		/*get a new row index*/
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the row index to the other threads in the same warp and compute the row index of each vetor*/
		row = __shfl(row, 0) + warpVectorId;

	}/*while*/
}

/*32-bit*/
template<typename T, uint32_t THREADS_PER_VECTOR,
		uint32_t MAX_NUM_VECTORS_PER_BLOCK>
__global__ void csr32DynamicWarpBLAS(uint32_t* __restrict cudaRowCounter,
		const uint32_t cudaNumRows, const uint32_t cudaNumCols,
		const uint32_t* __restrict rowOffsets,
		const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const cudaTextureObject_t vectorX,
		T* vectorY, const T alpha, const T beta) {
	uint32_t i;
	T sum;
	uint32_t row;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const uint32_t warpLaneId = threadIdx.x & 31; /*lane index in the warp*/
	const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR; /*vector index in the warp*/

	__shared__   volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp and compute the row index of each vector*/
	row = __shfl(row, 0) + warpVectorId;

	/*check the row range*/
	while (row < cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}

			/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}
		/*intra-vector reduction*/
		sum *= alpha;
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += __shfl_down(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum + beta * vectorY[row];
		}

		/*get a new row index*/
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the row index to the other threads in the same warp and compute the row index of each vetor*/
		row = __shfl(row, 0) + warpVectorId;

	}/*while*/
}
//with traditional arrays
template<typename T, uint32_t THREADS_PER_VECTOR,
		uint32_t MAX_NUM_VECTORS_PER_BLOCK>
__global__ void csr32DynamicWarpBLAS(uint32_t* __restrict cudaRowCounter,
		const uint32_t cudaNumRows, const uint32_t cudaNumCols,
		const uint32_t* __restrict rowOffsets,
		const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const T* __restrict vectorX,
		T* vectorY, const T alpha, const T beta) {
	uint32_t i;
	T sum;
	uint32_t row;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const uint32_t warpLaneId = threadIdx.x & 31; /*lane index in the warp*/
	const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR; /*vector index in the warp*/

	__shared__   volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp and compute the row index of each vector*/
	row = __shfl(row, 0) + warpVectorId;

	/*check the row range*/
	while (row < cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}

			/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}
		/*intra-vector reduction*/
		sum *= alpha;
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += __shfl_down(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum + beta * vectorY[row];
		}

		/*get a new row index*/
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the row index to the other threads in the same warp and compute the row index of each vetor*/
		row = __shfl(row, 0) + warpVectorId;

	}/*while*/
}


template<typename T, uint32_t THREADS_PER_VECTOR,
		uint32_t MAX_NUM_VECTORS_PER_BLOCK>
__global__ void csr64DynamicWarp(uint32_t* __restrict cudaRowCounter,
		const uint32_t cudaNumRows, const uint32_t cudaNumCols,
		const uint32_t* __restrict rowOffsets,
		const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const cudaTextureObject_t vectorX,
		T* vectorY)
		{
	uint32_t i;
	T sum;
	uint32_t row;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const uint32_t warpLaneId = threadIdx.x & 31; /*lane index in the warp*/
	const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR; /*vector index in the warp*/

	__shared__   volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp*/
	row = __shfl(row, 0) + warpVectorId;

	/*check the row range*/
	while (row < cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}

			/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}

		/*intra-vector reduction*/
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += shfl_down_64bits<T>(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum;
		}

		/*get a new row index*/
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the value to other threads in the same warp*/
		row = __shfl(row, 0) + warpVectorId;

	}/*while*/
}

template<typename T, uint32_t THREADS_PER_VECTOR,
		uint32_t MAX_NUM_VECTORS_PER_BLOCK>
__global__ void csr64DynamicWarp(uint32_t* __restrict cudaRowCounter,
		const uint32_t cudaNumRows, const uint32_t cudaNumCols,
		const uint32_t* __restrict rowOffsets,
		const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const T* __restrict vectorX,
		T* vectorY)
		{
	uint32_t i;
	T sum;
	uint32_t row;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const uint32_t warpLaneId = threadIdx.x & 31; /*lane index in the warp*/
	const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR; /*vector index in the warp*/

	__shared__   volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp*/
	row = __shfl(row, 0) + warpVectorId;

	/*check the row range*/
	while (row < cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}

			/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}

		/*intra-vector reduction*/
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += shfl_down_64bits<T>(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum;
		}

		/*get a new row index*/
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the value to other threads in the same warp*/
		row = __shfl(row, 0) + warpVectorId;

	}/*while*/
}


template<typename T, uint32_t THREADS_PER_VECTOR,
		uint32_t MAX_NUM_VECTORS_PER_BLOCK>
__global__ void csr64DynamicWarpBLAS(uint32_t* __restrict cudaRowCounter,
		const uint32_t cudaNumRows, const uint32_t cudaNumCols,
		const uint32_t* __restrict rowOffsets,
		const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const cudaTextureObject_t vectorX,
		const T* __restrict inVectorY, T* vectorY, const T alpha,
		const T beta)
		{
	uint32_t i;
	T sum;
	uint32_t row;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const uint32_t warpLaneId = threadIdx.x & 31; /*lane index in the warp*/
	const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR; /*vector index in the warp*/

	__shared__   volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp*/
	row = __shfl(row, 0) + warpVectorId;

	/*check the row range*/
	while (row < cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}

			/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}

		/*intra-vector reduction*/
		sum *= alpha;
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += shfl_down_64bits<T>(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum + beta * DOUBLE_VECTOR_GET(inVectorY, row);
		}

		/*get a new row index*/
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the value to other threads in the same warp*/
		row = __shfl(row, 0) + warpVectorId;

	}/*while*/
}

template<typename T, uint32_t THREADS_PER_VECTOR,
		uint32_t MAX_NUM_VECTORS_PER_BLOCK>
__global__ void csr64DynamicWarpBLAS(uint32_t* __restrict cudaRowCounter,
		const uint32_t cudaNumRows, const uint32_t cudaNumCols,
		const uint32_t* __restrict rowOffsets,
		const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const T* __restrict vectorX,
		const T* __restrict inVectorY, T* vectorY, const T alpha,
		const T beta)
		{
	uint32_t i;
	T sum;
	uint32_t row;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const uint32_t warpLaneId = threadIdx.x & 31; /*lane index in the warp*/
	const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR; /*vector index in the warp*/

	__shared__   volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp*/
	row = __shfl(row, 0) + warpVectorId;

	/*check the row range*/
	while (row < cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}

			/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i]
						* DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}

		/*intra-vector reduction*/
		sum *= alpha;
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += shfl_down_64bits<T>(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum + beta * DOUBLE_VECTOR_GET(inVectorY, row);
		}

		/*get a new row index*/
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the value to other threads in the same warp*/
		row = __shfl(row, 0) + warpVectorId;

	}/*while*/
}


}/*namespace*/



#endif /* LIGHTSPMVCORE_H_ */
