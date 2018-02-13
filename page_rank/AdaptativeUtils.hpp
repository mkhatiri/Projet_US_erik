#ifndef ADAPTATIVE_UTILS_HPP
#define ADAPTATIVE_UTILS_HPP

#include <iostream>
#include <unistd.h>
#include <string>
#include <list>
#include <stdio.h>
#include <string.h>
#include "tbb/concurrent_queue.h"
#include <stdlib.h>
//#include "streamUtils.hpp"
#include <cstdlib>
#include <iterator>
#include <assert.h> 
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "helper_cuda.h"

#define ROW_BITS  32
#define WG_BITS  24 

using namespace std;

template <typename VertexType, typename EdgeType, typename Scalar>
struct stream_container {
	
	int id;
	Scalar* alpha;
	Scalar* beta;
	Scalar* d_val;
	EdgeType* d_xadj;
	VertexType* d_adj ;
	Scalar* d_prin;
	Scalar* d_prout;
	
	unsigned long *d_rowBlocks;
	EdgeType rowBlocksPtr;
	int rowBlockSize;

	unsigned int* d_blkSize;
	unsigned int* d_rows_for_vector;
	unsigned int* d_blkMultiplier;
	
	tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*> * streams;
	cudaStream_t stream;
	int device;

};

struct Task {  
	int id;
	int rowBlockSize;
	int rowBlocksPtr;

};


__global__ void csr_adaptativeT(int* a);

__global__ void csr_adaptative(float* vals, int* cols, int* rowPtrs, float* vec, float* out,
                unsigned long* rowBlocks, float* d_alpha, float* d_beta, unsigned int* d_blkSize,
                unsigned int* d_blkMultiple, unsigned int* d_rowForVector, int rowBlockSize, int* method, float * rowErr);
 

unsigned int flp2(unsigned int x);

int cutRowBlocks(unsigned long* rowBlocks, int rowBlockSize);

int split_input_to_tasks(unsigned long* rowBlocks, int rowBlockSize, int subsize, list<Task>& tasks); 

Task get_task(list<Task>* tasks,int index); 

template <typename VertexType, typename EdgeType, typename Scalar>       
void creat_stream(unsigned long *d_rowBlocks, Scalar* alpha, Scalar* beta, Scalar* d_val, EdgeType* d_xadj, VertexType *d_adj, Scalar* d_prin, Scalar* d_prout, unsigned int* d_blkSize, unsigned int* d_rows_for_vector, unsigned int* d_blkMultiplier, tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, int stream_number );

template <typename VertexType, typename EdgeType, typename Scalar>
void creat_stream_2gpus(unsigned long *d_rowBlocks0, Scalar* alpha0, Scalar* beta0, Scalar* d_val0, EdgeType* d_xadj0, VertexType *d_adj0, Scalar* d_prin0, Scalar* d_prout0, unsigned int* d_blkSize0, unsigned int* d_rows_for_vector0, unsigned int* d_blkMultiplier0, unsigned long *d_rowBlocks1, Scalar* alpha1, Scalar* beta1, Scalar* d_val1, EdgeType* d_xadj1, VertexType *d_adj1, Scalar* d_prin1, Scalar* d_prout1, unsigned int* d_blkSize1, unsigned int* d_rows_for_vector1, unsigned int* d_blkMultiplier1, tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, int stream_number );

template <typename VertexType, typename EdgeType, typename Scalar>
void put_work_on_stream(stream_container<VertexType, EdgeType, Scalar>* current_stream, Task current_task);


template <typename VertexType, typename EdgeType, typename Scalar>       
void add_new_idle_stream(tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, stream_container <VertexType, EdgeType, Scalar>* stream);

template <typename VertexType, typename EdgeType>
unsigned long numThreadsForReduction(unsigned long num_rows, int WGSIZE);

template <typename VertexType, typename EdgeType>
void ComputeRowBlocks( unsigned long* rowBlocks, EdgeType& rowBlockSize, const EdgeType* xadj,
		const EdgeType nRows, const int blkSize, const int blkMultiplier, 
		const int rows_for_vector, int WGSIZE, const bool allocate_row_blocks = true);

template <typename VertexType, typename EdgeType>
size_t ComputeRowBlocksSize( const EdgeType* rowDelimiters, const EdgeType nRows, const unsigned int blkSize,
		const unsigned int blkMultiplier, const unsigned int rows_for_vector, int WGSIZE);

void CUDART_CB call_back(cudaStream_t Stream, cudaError_t err, void* data);

void cudaPrintError(std::string m);
  

#endif

