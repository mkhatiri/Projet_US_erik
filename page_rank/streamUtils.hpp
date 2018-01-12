#ifndef STREAM_UTILS_HPP
#define STREAM_UTILS_HPP

#include <stdio.h>
#include "tbb/concurrent_queue.h"
#include <cuda_runtime_api.h>
#include <list>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "helper_cuda.h"

using namespace std;

template <typename VertexType, typename EdgeType, typename Scalar>
struct stream_container {
	int id;
	VertexType m;
	VertexType n;
	EdgeType nnz;
	Scalar* alpha;
	Scalar* beta;
	Scalar* d_val;
	EdgeType* d_xadj;
	VertexType* d_adj ;
	Scalar* d_prin;
	Scalar* d_prout;
	VertexType RowPtr;
	cusparseHandle_t* cusparseHandle;
	cusparseMatDescr_t* descr;
	tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*> * streams;
	cudaStream_t stream;
	int device;
};

template <typename VertexType, typename EdgeType, typename Scalar>
struct Task {  
	int id;
	VertexType nVtx;
	EdgeType nnz;
	VertexType RowPtr;
};

//template <typename VertexType, typename EdgeType, typename Scalar>
void CUDART_CB call_back(cudaStream_t Stream, cudaError_t err, void* data);

template <typename VertexType, typename EdgeType, typename Scalar>
void add_new_idle_stream(tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, stream_container<VertexType, EdgeType, Scalar>* stream) ;


template <typename VertexType, typename EdgeType, typename Scalar>
int split_input_to_tasks(EdgeType* xadj, VertexType nVtx, VertexType subsize, list<Task<VertexType, EdgeType, Scalar> >& tasks);

template <typename VertexType, typename EdgeType, typename Scalar>
int smart_split_input_to_tasks(EdgeType* xadj, VertexType nVtx, EdgeType subsize, list<Task<VertexType, EdgeType, Scalar> >& tasks);

template <typename VertexType, typename EdgeType, typename Scalar>
Task<VertexType, EdgeType, Scalar> get_task(list<Task<VertexType, EdgeType, Scalar> >* tasks, int index);

template <typename VertexType, typename EdgeType, typename Scalar>
void creat_stream(VertexType nVtx, Scalar* alpha, Scalar* beta, Scalar* d_val, EdgeType* d_xadj, VertexType *d_adj, Scalar* d_prin, Scalar* d_prout, tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>*  streams, int stream_number ); 


template <typename VertexType, typename EdgeType, typename Scalar>
void creat_stream_2gpus(VertexType nVtx, Scalar* alpha, Scalar* beta, Scalar* d_val0, EdgeType* d_xadj0, VertexType *d_adj0, Scalar* d_prin0, Scalar* d_prout0, Scalar* d_val1, EdgeType* d_xadj1, VertexType *d_adj1, Scalar* d_prin1, Scalar* d_prout1, cusparseHandle_t* cusparseHandle0, cusparseHandle_t* cusparseHandle1, cusparseMatDescr_t* descr0, cusparseMatDescr_t* descr1, tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>*  streams, int stream_number ); 

template <typename VertexType, typename EdgeType, typename Scalar>
void put_work_on_stream(stream_container<VertexType, EdgeType, Scalar>* current_stream, Task<VertexType, EdgeType, Scalar> current_task);

#endif
