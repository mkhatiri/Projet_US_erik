#include <iostream>
#include <unistd.h>
#include <string>
#include <list>
#include <stdio.h>
#include <string.h>
#include "tbb/concurrent_queue.h"
#include <stdlib.h>
#include "streamUtils.hpp"
#include <cstdlib>
#include <iterator>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "helper_cuda.h"

using namespace std;


//template <typename VertexType, typename EdgeType, typename Scalar>
void CUDART_CB call_back(cudaStream_t Stream, cudaError_t err, void* data){

	stream_container<int,int,float> *stream = (stream_container<int,int,float> *) data;
//	std::cout << "stream " << stream->id << " terminer  on " << stream->device << std::endl;
	
	//cudaPrintError("in call_back_befor update stream list");
//	stream->streams->push(stream->stream);	
	add_new_idle_stream<int,int,float>(stream->streams, stream);
	//cudaPrintError("in call_back after update stream list");

}


template <typename VertexType, typename EdgeType, typename Scalar>
void add_new_idle_stream(tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, stream_container <VertexType, EdgeType, Scalar>* stream) {
//	std::cout << "add stream " << stream->id << " terminer on GPU " << stream->device << std::endl;
	streams->push(stream);
}

template <typename VertexType, typename EdgeType, typename Scalar>
int smart_split_input_to_tasks(EdgeType* xadj, VertexType nVtx, EdgeType subsize, list<Task<VertexType, EdgeType, Scalar> >& tasks) { 
	
	EdgeType lnnz = 0;
	VertexType i = 0;
	int current_subsize = subsize;
	EdgeType lastRowPtr = 0;
	int lastnnz = 0;
	int xadj_current = 0;
	int id = 0;
	for(i = 1 ; i!= nVtx; i++){
		xadj_current = xadj[i];
		if(xadj_current >= current_subsize) {
	                Task<VertexType, EdgeType, Scalar> t;
			t.id = id ++;
			t.RowPtr = lastRowPtr;
			t.nnz = xadj_current - lastnnz;
			lastnnz = xadj_current; 
			t.nVtx = i - lastRowPtr; 
			lastnnz = xadj[i];
			lastRowPtr = i;
			current_subsize += subsize;
			//std::cout << id<< " xadj_current " << xadj_current << " T.RowPtr " << t.RowPtr << " t.nnz "  << t.nnz << " t.nVtx "<< t.nVtx << std::endl;
			xadj_current = 0;
			
			tasks.push_back(t);
		}
	}
			Task<VertexType, EdgeType, Scalar> t;
                        t.id = id;
			t.RowPtr = lastRowPtr;
                        t.nnz = xadj[nVtx] - lastnnz;
                        t.nVtx = i - lastRowPtr;
			//std::cout << "--- > " << id << " xadj_current " << xadj_current << " T.RowPtr " << lastRowPtr  << " t.nnz "  << t.nnz  << " t.nVtx "<< i - lastRowPtr << std::endl;
			tasks.push_back(t);
	return id;
}

template <typename VertexType, typename EdgeType, typename Scalar>
int split_input_to_tasks(EdgeType* xadj, VertexType nVtx, VertexType subsize, list<Task<VertexType, EdgeType, Scalar> >& tasks) { 

	int id = 0;
	EdgeType lnnz = 0;
	for(VertexType i = 0; i < nVtx; i += subsize){
		Task<VertexType, EdgeType, Scalar> t;        
		t.id = id++;	
		t.RowPtr = i  ;
		if(i < nVtx - subsize){
			if(i == 0){
			  t.nnz = xadj[subsize];
			  lnnz = xadj[subsize];
			}else{
			  t.nnz = xadj[i  + subsize] - lnnz ;
			  lnnz = xadj[i + subsize];
			}
			t.nVtx = subsize  ;
		}else{
			t.nnz = xadj[nVtx] - lnnz;
			lnnz = xadj[nVtx];
			t.nVtx = nVtx - i  ;
		}        
		//std::cout << id << " nVtx " << t.nVtx << " Nnz "<< t.nnz << std::endl;
		//std::cout << "RowPtr : " << t.RowPtr << " , nnz : " << t.nnz << " , m : "<< t.nVtx << endl;
//		std::cout << "RowPtr : " << t.RowPtr << " , nnz : " << t.nnz << " , m : "<< t.nVtx << endl; 		
		tasks.push_back(t);
	}

	return id;
	//	cout << "lnnx total" << lnnz << endl;
}

template <typename VertexType, typename EdgeType, typename Scalar>
Task<VertexType, EdgeType, Scalar> get_task(list<Task<VertexType, EdgeType, Scalar> >* tasks,int index) {

	//Task<VertexType, EdgeType, Scalar> t = tasks.front();
	//tasks.pop_front();
	//template <typename VertexType, typename EdgeType, typename Scalar>
	typename list<Task<VertexType, EdgeType, Scalar> >::iterator it = tasks->begin();
	advance(it, index);

	return *it;
}


template <typename VertexType, typename EdgeType, typename Scalar>
void creat_stream(VertexType nVtx, Scalar* alpha, Scalar* beta, Scalar* d_val, EdgeType* d_xadj, VertexType *d_adj, Scalar* d_prin, Scalar* d_prout, tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, int stream_number ) {

	for(int i=0 ; i < stream_number ; i++ ) {

		stream_container<VertexType, EdgeType, Scalar> * stream;
		stream = (stream_container<VertexType, EdgeType, Scalar> * ) malloc(1*sizeof(stream_container<VertexType, EdgeType, Scalar>));

		stream->id = i;
		stream->n = nVtx;
		stream->alpha = alpha;
		stream->beta = beta;
		stream->d_val = d_val;
		stream->d_xadj = d_xadj;
		stream->d_adj = d_adj;
		stream->d_prin = d_prin;
		stream->d_prout = d_prout;
		stream->streams = streams;
		cudaStream_t new_stream;
		cudaStreamCreate( &new_stream );
		stream->stream = new_stream;
		//add_new_idle_stream(stream);
		//streams->push(stream);
		//stream->lspmv = new lightSpMVCSRKernel();
		add_new_idle_stream<VertexType, EdgeType, Scalar>(streams, stream);
//		std::cout << " creat stream : " << i << std::endl;
	}
}

template <typename VertexType, typename EdgeType, typename Scalar>
void put_work_on_stream(stream_container<VertexType, EdgeType, Scalar>* current_stream, Task<VertexType, EdgeType, Scalar> current_task){

	current_stream->RowPtr = current_task.RowPtr;
	current_stream->m = current_task.nVtx;
	current_stream->nnz = current_task.nnz;
//	current_stream->d_xadj = current_task.xadj;
//	current_stream->d_prin = current_task.prin;
//	current_stream->d_prout = current_task.prout;
}


/* ------------------------- 2 GPU ------------------------ */


template <typename VertexType, typename EdgeType, typename Scalar>
void creat_stream_2gpus(VertexType nVtx, Scalar* alpha, Scalar* beta, Scalar* d_val0, EdgeType* d_xadj0, VertexType *d_adj0, Scalar* d_prin0, Scalar* d_prout0, Scalar* d_val1, EdgeType* d_xadj1, VertexType *d_adj1, Scalar* d_prin1, Scalar* d_prout1,cusparseHandle_t* cusparseHandle0,cusparseHandle_t* cusparseHandle1, cusparseMatDescr_t* descr0, cusparseMatDescr_t* descr1, tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, int stream_number ) {
	for(int i=0 ; i < stream_number ; i++ ) {

		cudaSetDevice(0);
		stream_container<VertexType, EdgeType, Scalar> * stream0;
		stream0 = (stream_container<VertexType, EdgeType, Scalar> * ) malloc(1*sizeof(stream_container<VertexType, EdgeType, Scalar>));

		stream0->id = i;
		stream0->n = nVtx;
		stream0->alpha = alpha;
		stream0->beta = beta;
		stream0->d_val = d_val0;
		stream0->d_xadj = d_xadj0;
		stream0->d_adj = d_adj0;
		stream0->d_prin = d_prin0;
		stream0->d_prout = d_prout0;
		stream0->streams = streams;
		stream0->device = 0;
		cudaStream_t new_stream0;
		cudaStreamCreate( &new_stream0 );
		stream0->stream = new_stream0;
		stream0->cusparseHandle = cusparseHandle0;
		stream0->descr = descr0;
		//add_new_idle_stream(stream);
		//streams->push(stream);
		add_new_idle_stream<VertexType, EdgeType, Scalar>(streams, stream0);
		//		std::cout << " creat stream : " << i << std::endl;

		cudaSetDevice(1);
		stream_container<VertexType, EdgeType, Scalar> * stream1;
		stream1 = (stream_container<VertexType, EdgeType, Scalar> * ) malloc(1*sizeof(stream_container<VertexType, EdgeType, Scalar>));

		stream1->id = i;
		stream1->n = nVtx;
		stream1->alpha = alpha;
		stream1->beta = beta;
		stream1->d_val = d_val1;
		stream1->d_xadj = d_xadj1;
		stream1->d_adj = d_adj1;
		stream1->d_prin = d_prin1;
		stream1->d_prout = d_prout1;
		stream1->streams = streams;
		stream1->device = 1;
		cudaStream_t new_stream1;
		cudaStreamCreate( &new_stream1 );
		stream1->stream = new_stream1;
		stream1->cusparseHandle = cusparseHandle1;
		stream1->descr = descr1;
		//add_new_idle_stream(stream);
		//streams->push(stream);
		add_new_idle_stream<VertexType, EdgeType, Scalar>(streams, stream1);
		//		std::cout << " creat stream : " << i << std::endl;

	}
}

template 
void add_new_idle_stream(tbb::concurrent_bounded_queue<stream_container<int,int,float>*>* streams, stream_container <int,int,float>* stream);

template 
int split_input_to_tasks<int,int,float>(int* xadj, int nVtx, int subsize, list<Task<int, int, float> >& tasks);


template 
int smart_split_input_to_tasks<int,int,float>(int* xadj, int nVtx, int subsize, list<Task<int, int, float> >& tasks);

template 
Task<int,int,float> get_task(list<Task<int,int,float> >* tasks, int index);

template
void creat_stream(int nVtx, float* alpha, float* beta, float* d_val, int* d_xadj, int *d_adj, float* d_prin, float* d_prout, tbb::concurrent_bounded_queue<stream_container<int, int, float>*>* streams, int stream_number );

template
void creat_stream_2gpus(int nVtx, float* alpha, float* beta, float* d_val0, int* d_xadj0, int *d_adj0, float* d_prin0, float* d_prout0, float* d_val1, int* d_xadj1, int *d_adj1, float* d_prin1, float* d_prout1, cusparseHandle_t* cusparseHandle0,cusparseHandle_t* cusparseHandle1, cusparseMatDescr_t* descr0, cusparseMatDescr_t* descr1, tbb::concurrent_bounded_queue<stream_container<int, int, float>*>* streams, int stream_number );


template 
void put_work_on_stream(stream_container<int, int, float>* current_stream, Task<int, int, float> current_task);


void CUDART_CB call_back(cudaStream_t Stream, cudaError_t err, void* data);

