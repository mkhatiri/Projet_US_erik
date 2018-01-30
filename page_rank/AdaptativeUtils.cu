#include <iostream>
#include <unistd.h>
#include <string>
#include <list>
#include <stdio.h>
#include <string.h>
#include "tbb/concurrent_queue.h"
#include <stdlib.h>
#include "AdaptativeUtils.hpp"
#include <cstdlib>
#include <iterator>
#include <assert.h> 
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "helper_cuda.h"



void cudaPrintError(std::string m) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr<<m<<" : "<<cudaGetErrorString(err)<<std::endl;
	}
}


void CUDART_CB call_back(cudaStream_t Stream, cudaError_t err, void* data){

	stream_container<int,int,float> *stream = (stream_container<int,int,float> *) data;
	//std::cout << "stream " << stream->id << " terminer  on " << stream->device << std::endl;
	cudaPrintError("in call_back_befor update stream list");
	//stream->streams->push(stream);  
	add_new_idle_stream<int,int,float>(stream->streams, stream);
	//cout << " call back : size  "<< stream->streams->size() <<endl;

	cudaPrintError("in call_back after update stream list");

}


unsigned int flp2(unsigned int x)
{
	x |= (x >> 1);
	x |= (x >> 2);
	x |= (x >> 4);
	x |= (x >> 8);
	x |= (x >> 16);
	return x - (x >> 1);
}


unsigned long numThreadsForReduction(unsigned long num_rows,int WGSIZE)
{
	return (unsigned long) flp2(WGSIZE/num_rows);
}


	template <typename VertexType, typename EdgeType>
void ComputeRowBlocks( unsigned long* rowBlocks, EdgeType& rowBlockSize, const EdgeType* xadj,
		const EdgeType nRows, const int blkSize, const int blkMultiplier, 
		const int rows_for_vector, int WGSIZE, const bool allocate_row_blocks = true)
{
	unsigned long* rowBlocksBase;
	EdgeType total_row_blocks = 1; // Start at one because of rowBlock[0]

	if (allocate_row_blocks)
	{
		rowBlocksBase = rowBlocks;
		*rowBlocks = 0;
		rowBlocks++;
	}
	unsigned long sum = 0;
	unsigned long i, last_i = 0;
	// Check to ensure nRows can fit in 32 bits
	if( (EdgeType)nRows > (EdgeType)pow( 2, ROW_BITS ) )
	{
		printf( "Number of Rows in the Sparse Matrix is greater than what is supported at present (%d bits) !", ROW_BITS );
		return;
	}

	EdgeType consecutive_long_rows = 0;
	for( i = 1; i <= nRows; i++ )
	{
		EdgeType row_length = ( xadj[ i ] - xadj[ i - 1 ] );
	if (allocate_row_blocks)
                                {
		//std::cout << i << " row_length " << row_length << " - " << endl;
				}
		sum += row_length;
		//		std::cout << "i" << i <<std::endl;

		// The following section of code calculates whether you're moving between
		// a series of "short" rows and a series of "long" rows.
		// This is because the reduction in CSR-Adaptive likes things to be
		// roughly the same length. Long rows can be reduced horizontally.
		// Short rows can be reduced one-thread-per-row. Try not to mix them.

		if ( row_length > blkSize){
		//**	std::cout << "row_lenght1 > blkSize " << row_length << " at iter : " << i <<std::endl;
			consecutive_long_rows++;
		} else if ( consecutive_long_rows > 0 )
		{
		//**	std::cout << "else cons > 0  at iter : " << i <<std::endl;
			// If it turns out we WERE in a long-row region, cut if off now.
			if (row_length < 32 ) // Now we're in a short-row region
			{//**	std::cout << "row_length2<32  at iter : " << i <<std::endl;
				consecutive_long_rows = -1;
			}else{
				consecutive_long_rows++;
			//**	std::cout << "else row_length3 < 256  at iter : " << i <<std::endl;
			}
		}
		if (allocate_row_blocks)
                                {
			//**		std::cout << i << " row_length=" << row_length << " blkSize=" << blkSize << " consecutive_long_rows" <<consecutive_long_rows<< endl;
				}

		// If you just entered into a "long" row from a series of short rows,
		// then we need to make sure we cut off those short rows. Put them in
		// their own workgroup.
		if ( consecutive_long_rows == 1 )
		{
			// Assuming there *was* a previous workgroup. If not, nothing to do here.
			if( i - last_i > 1 )
			{
				if (allocate_row_blocks)
				{
					*rowBlocks = ( (i-1) << 32) ;
			//**		std::cout << "Long :  R_rowBlocks " << (i-1) << " rowBlocks " << *rowBlocks << " sum " << sum <<std::endl;
					// If this row fits into CSR-Stream, calculate how many rows
					// can be used to do a parallel reduction.
					// Fill in the low-order bits with the numThreadsForRed
					if (((i-1) - last_i) > rows_for_vector)
					{  
						*(rowBlocks-1) |= numThreadsForReduction((i - 1) - last_i,WGSIZE);
					}
					rowBlocks++;
				}
				total_row_blocks++;
				last_i = i-1;
				sum = row_length;
			}
		}
		else if (consecutive_long_rows == -1)
		{
			// We see the first short row after some long ones that
			// didn't previously fill up a rssertow block.
			if (allocate_row_blocks)
			{
				*rowBlocks = ( (i - 1) << (64 - ROW_BITS) );
			//**	std::cout << "short : R_rowBlocks " << (i-1) << " rowBlocks " << *rowBlocks << " sum " << sum <<std::endl;
				if (((i-1) - last_i) > rows_for_vector){
					*(rowBlocks-1) |= numThreadsForReduction((i - 1) - last_i,WGSIZE);
				}
				rowBlocks++;
			}
			total_row_blocks++;
			last_i = i-1;
			sum = row_length;
			consecutive_long_rows = 0;
		}

		// Now, what's up with this row? What did it do?

		// exactly one row results in non-zero elements to be greater than blockSize
		// This is csr-vector case; bottom WG_BITS == workgroup ID
		if( ( i - last_i == 1 ) && sum > blkSize )
		{
			int numWGReq = static_cast< int >( ceil( (double)row_length / (blkMultiplier*blkSize) ) );

			// Check to ensure #workgroups can fit in WG_BITS bits, if not
			// then the last workgroup will do all the remaining work
			numWGReq = ( numWGReq < (int)pow( 2, WG_BITS ) ) ? numWGReq : (int)pow( 2, WG_BITS );

			if (allocate_row_blocks)
			{
				for( unsigned long w = 1; w < numWGReq; w++ )
				{
					*rowBlocks = ( (i - 1) << (64 - ROW_BITS) );
				//**		std::cout << "!!! mediun :  R_rowBlocks " << (i-1) << " rowBlocks " << *rowBlocks << " sum " << sum << std::endl;
					*rowBlocks |= static_cast< unsigned long >( w );
					rowBlocks++;
				}
				*rowBlocks = ( i << (64 - ROW_BITS) );
			//**	std::cout << "mediun :  R_rowBlocks " << i << " rowBlocks " << *rowBlocks << " sum " << sum << std::endl;
				rowBlocks++;
			}
			total_row_blocks += numWGReq;
			last_i = i;
			sum = 0;
			consecutive_long_rows = 0;
		}
		// more than one row results in non-zero elements to be greater than blockSize
		// This is csr-stream case; bottom WG_BITS = number of parallel reduction threads
		else if( ( i - last_i > 1 ) && sum > blkSize )
		{
			i--; // This row won't fit, so back off one.
			if (allocate_row_blocks)
			{
				*rowBlocks = ( i << (64 - ROW_BITS) );
			//**	std::cout << "more : R_rowBlocks " << (i-1) << " rowBlocks " << *rowBlocks << " sum "<< sum << std::endl;
				if ((i - last_i) > rows_for_vector)
					*(rowBlocks-1) |= numThreadsForReduction(i - last_i, WGSIZE);
				rowBlocks++;
			}
			total_row_blocks++;
			last_i = i;
			sum = 0;
			consecutive_long_rows = 0;
		}
		// This is csr-stream case; bottom WG_BITS = number of parallel reduction threads
		else if( sum == blkSize )
		{
			if (allocate_row_blocks)
			{
				*rowBlocks = ( i << (64 - ROW_BITS) );
			//**	std::cout << "exacte : R_rowBlocks " << (i-1) << " rowBlocks " << *rowBlocks << " sum " << sum << std::endl;
				if ((i - last_i) > rows_for_vector)
					*(rowBlocks-1) |= numThreadsForReduction(i - last_i, WGSIZE);
				rowBlocks++;
			}
			total_row_blocks++;
			last_i = i;
			sum = 0;
			consecutive_long_rows = 0;
		}
	}

	// If we didn't fill a row block with the last row, make sure we don't lose it.
	if ( allocate_row_blocks && (*(rowBlocks-1) >> (64 - ROW_BITS)) != static_cast< unsigned long >(nRows) )
	{
		*rowBlocks = (static_cast< unsigned long >( nRows ) << 32 ) ;
	//**	std::cout << "Last : iter : " << i << " R_rowBlocks " << (i-1) << " rowBlocks " << *(rowBlocks) << std::endl;
		if ((nRows - last_i) > rows_for_vector)
			*(rowBlocks-1) |= numThreadsForReduction(i - last_i, WGSIZE);
		rowBlocks++;
	}
	total_row_blocks++;

	if (allocate_row_blocks)
	{
		size_t dist = std::distance( rowBlocksBase, rowBlocks );
		assert( (2 * dist) <= rowBlockSize );
		// Update the size of rowBlocks to reflect the actual amount of memory used
		// We're multiplying the size by two because the extended precision form of
		// CSR-Adaptive requires more space for the final global reduction.
		rowBlockSize =   dist;
	}
	else
		rowBlockSize  =  total_row_blocks;
}


int cutRowBlocks(unsigned long * rowBlocks, int rowBlockSize){
	int medium = (rowBlockSize % 2 ? rowBlockSize/2 : (rowBlockSize-1)/2);

	int mediumRows = ((rowBlocks[medium] >> (64-32)) & ((1UL << 32) - 1UL));
	int mediumRows1 = ((rowBlocks[medium + 1] >> (64-32)) & ((1UL << 32) - 1UL));

	while( mediumRows == mediumRows1 )
	{	medium++;
		mediumRows = ((rowBlocks[medium] >> (64-32)) & ((1UL << 32) - 1UL));
		mediumRows1 = ((rowBlocks[medium + 1] >> (64-32)) & ((1UL << 32) - 1UL));
	}
	return medium;

}



	template <typename VertexType, typename EdgeType>
size_t ComputeRowBlocksSize( const EdgeType* rowDelimiters, const EdgeType nRows, const unsigned int blkSize,
		const unsigned int blkMultiplier, const unsigned int rows_for_vector, int WGSIZE )
{
	EdgeType rowBlockSize;
	ComputeRowBlocks<VertexType,EdgeType>( NULL, rowBlockSize, rowDelimiters, nRows, blkSize, blkMultiplier, rows_for_vector, WGSIZE, false);
	return rowBlockSize;
}


int split_input_to_tasks(unsigned long *rowBlocks, int rowBlockSize, int subsize, list<Task>& tasks) { 

	int id = 0;
	int subRowBlockSize = 0;
	int ptr = 0;
	bool verifi = false;
	//cerr << "sub=" << subsize<<endl;
	while(ptr < rowBlockSize){
		//cerr << "ptr=" << ptr << " rowBlockSize=" << rowBlockSize << endl;
		Task t ;
		t.id = id++;
		t.rowBlocksPtr = ptr;
		ptr += subsize;

		if (ptr < rowBlockSize){
			int xadjPtr1 =  ((rowBlocks[ptr] >> (64-32)) & ((1UL << 32) - 1UL));
			int xadjPtr2 =  ((rowBlocks[ptr+1] >> (64-32)) & ((1UL << 32) - 1UL));

	//		cout << ptr <<" xadjPtr1 "  << xadjPtr1 <<endl; 
	//		cout << ptr+1 <<" xadjPtr2 "  << xadjPtr2 <<endl; 
			verifi = false;
			while (xadjPtr1 == xadjPtr2){

				ptr++;
				xadjPtr1 =  ((rowBlocks[ptr] >> (64-32)) & ((1UL << 32) - 1UL));
				xadjPtr2 =  ((rowBlocks[ptr+1] >> (64-32)) & ((1UL << 32) - 1UL));
	//			cout << " -  xadjPtr1 "  << xadjPtr1 <<endl;
	//			cout << " -  xadjPtr2 "  << xadjPtr2 <<endl; 
			}
			

		if(ptr < rowBlockSize){
			t.rowBlockSize = (ptr - subRowBlockSize)   ;
//			std::cout << id << " rowBlocks[Ptr] " << rowBlocks[ptr]  << " Ptr "  << t.rowBlocksPtr << " - subRowBlockSize " << subRowBlockSize  << " t.rowBlockSize : "  << t.rowBlockSize << endl;
			subRowBlockSize = ptr;
			}else{
				t.rowBlockSize = rowBlockSize - subRowBlockSize  ;
			//ptr++;
			}
		}else{
//			std::cout << id << "end ... " << endl;
			t.rowBlockSize = rowBlockSize - subRowBlockSize  ;
//			std::cout << id << " rowBlocks[Ptr] " << rowBlocks[ptr] << " Ptr "  << t.rowBlocksPtr << " - subRowBlockSize " << subRowBlockSize  << " t.rowBlockSize : "  << t.rowBlockSize << endl;
			//ptr++;
		}

		tasks.push_back(t); 
	//	cerr << "t.id="<<t.id<<" t.rowBlocksPtr="<<t.rowBlocksPtr<<" t.rowBlockSize="<<t.rowBlockSize<<endl;
	}
	return id;
}


Task get_task(list<Task>* tasks,int index) {

	//Task<VertexType, EdgeType, Scalar> t = tasks.front();
	//tasks.pop_front();
	//template <typename VertexType, typename EdgeType, typename Scalar>
	typename list<Task>::iterator it = tasks->begin();
	advance(it, index);
	return *it;
}


template <typename VertexType, typename EdgeType, typename Scalar>
void creat_stream(unsigned long *d_rowBlocks, Scalar* alpha, Scalar* beta, Scalar* d_val, EdgeType* d_xadj, VertexType *d_adj, Scalar* d_prin, Scalar* d_prout, unsigned int* d_blkSize, unsigned int* d_rows_for_vector, unsigned int* d_blkMultiplier, tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, int stream_number ) {

	for(int i=0 ; i < stream_number ; i++ ) {
		cout <<" id "<< i << endl;
		stream_container<VertexType, EdgeType, Scalar> * stream;
		stream = (stream_container<VertexType, EdgeType, Scalar> * ) malloc(1*sizeof(stream_container<VertexType, EdgeType, Scalar>));

		stream->id = i;
		stream->alpha = alpha;
		stream->beta = beta;
		stream->d_val = d_val;
		stream->d_xadj = d_xadj;
		stream->d_adj = d_adj;
		stream->d_prin = d_prin;
		stream->d_prout = d_prout;
		stream->d_rowBlocks = d_rowBlocks;
		stream->d_blkSize = d_blkSize;
		stream->d_rows_for_vector = d_rows_for_vector;
		stream->d_blkMultiplier = d_blkMultiplier;
		stream->streams = streams;
		cudaStream_t new_stream;
		cudaStreamCreate( &new_stream );
		stream->stream = new_stream;
		add_new_idle_stream<VertexType, EdgeType, Scalar>(streams, stream);
		//streams->push(stream);
		//stream->lspmv = new lightSpMVCSRKernel();
		// add_new_idle_stream<VertexType, EdgeType, Scalar>(streams, stream);
		//              std::cout << " creat stream : " << i << std::endl;
	}
}

template <typename VertexType, typename EdgeType, typename Scalar>
void creat_stream_2gpus(unsigned long *d_rowBlocks0, Scalar* alpha0, Scalar* beta0, Scalar* d_val0, EdgeType* d_xadj0, VertexType *d_adj0, Scalar* d_prin0, Scalar* d_prout0, unsigned int* d_blkSize0, unsigned int* d_rows_for_vector0, unsigned int* d_blkMultiplier0, unsigned long *d_rowBlocks1, Scalar* alpha1, Scalar* beta1, Scalar* d_val1, EdgeType* d_xadj1, VertexType *d_adj1, Scalar* d_prin1, Scalar* d_prout1, unsigned int* d_blkSize1, unsigned int* d_rows_for_vector1, unsigned int* d_blkMultiplier1, tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, int stream_number ) {

	for(int i=0 ; i < stream_number ; i++ ) {

                cudaSetDevice(0);
		stream_container<VertexType, EdgeType, Scalar> * stream0;
		stream0 = (stream_container<VertexType, EdgeType, Scalar> * ) malloc(1*sizeof(stream_container<VertexType, EdgeType, Scalar>));

		stream0->id = i;
		stream0->alpha = alpha0;
		stream0->beta = beta0;
		stream0->d_val = d_val0;
		stream0->d_xadj = d_xadj0;
		stream0->d_adj = d_adj0;
		stream0->d_prin = d_prin0;
		stream0->d_prout = d_prout0;
		stream0->d_rowBlocks = d_rowBlocks0;
		stream0->d_blkSize = d_blkSize0;
		stream0->d_rows_for_vector = d_rows_for_vector0;
		stream0->d_blkMultiplier = d_blkMultiplier0;
		stream0->streams = streams;
		cudaStream_t new_stream0;
		cudaStreamCreate( &new_stream0 );
		stream0->stream = new_stream0;
		stream0->device = 0;
		add_new_idle_stream<VertexType, EdgeType, Scalar>(streams, stream0);

		cudaSetDevice(1);
                stream_container<VertexType, EdgeType, Scalar> * stream1;
                stream1 = (stream_container<VertexType, EdgeType, Scalar> * ) malloc(1*sizeof(stream_container<VertexType, EdgeType, Scalar>));

                stream1->id = i;
                stream1->alpha = alpha1;
                stream1->beta = beta1;
                stream1->d_val = d_val1;
                stream1->d_xadj = d_xadj1;
                stream1->d_adj = d_adj1;
                stream1->d_prin = d_prin1;
                stream1->d_prout = d_prout1;
                stream1->d_rowBlocks = d_rowBlocks1;
                stream1->d_blkSize = d_blkSize1;
                stream1->d_rows_for_vector = d_rows_for_vector1;
                stream1->d_blkMultiplier = d_blkMultiplier1;
                stream1->device = 1;
                stream1->streams = streams;
                cudaStream_t new_stream1;
                cudaStreamCreate( &new_stream1 );
                stream1->stream = new_stream1;
                add_new_idle_stream<VertexType, EdgeType, Scalar>(streams, stream1);

	}
}







template <typename VertexType, typename EdgeType, typename Scalar>
void put_work_on_stream(stream_container<VertexType, EdgeType, Scalar>* current_stream, Task current_task){

	current_stream->rowBlockSize = current_task.rowBlockSize;
	current_stream->rowBlocksPtr = current_task.rowBlocksPtr;
}

template <typename VertexType, typename EdgeType, typename Scalar>
void add_new_idle_stream(tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, stream_container <VertexType, EdgeType, Scalar>* stream) {

	//	std::cout << "add stream " << stream->id << " terminer on GPU " << stream->device << std::endl;
	streams->push(stream);
}


template 
void ComputeRowBlocks<int,int>(unsigned long * rowBlocks, int& rowBlockSize, const int* xadj,
		const int nRows, const int blkSize, const int blkMultiplier,
		const int rows_for_vector, int WGSIZE, const bool allocate_row_blocks = true );


template 
size_t ComputeRowBlocksSize<int,int>( const int* rowDelimiters, const int nRows, const unsigned int blkSize, const unsigned int blkMultiplier, const unsigned int rows_for_vector, int WGSIZE );

template       
void creat_stream(unsigned long *d_rowBlocks, float* alpha, float* beta, float* d_val, int* d_xadj, int *d_adj, float* d_prin, float* d_prout, unsigned int* d_blkSize, unsigned int* d_rows_for_vector, unsigned int* d_blkMultiplier, tbb::concurrent_bounded_queue<stream_container<int, int,float>*>* streams, int stream_number );


template 
void creat_stream_2gpus(unsigned long *d_rowBlocks0, float* alpha0, float* beta0, float* d_val0, int* d_xadj0, int *d_adj0, float* d_prin0, float* d_prout0, unsigned int* d_blkSize0, unsigned int* d_rows_for_vector0, unsigned int* d_blkMultiplier0, unsigned long *d_rowBlocks1, float* alpha1, float* beta1, float* d_val1, int* d_xadj1, int *d_adj1, float* d_prin1, float* d_prout1, unsigned int* d_blkSize1, unsigned int* d_rows_for_vector1, unsigned int* d_blkMultiplier1, tbb::concurrent_bounded_queue<stream_container<int, int, float>*>* streams, int stream_number ); 


template 
void put_work_on_stream(stream_container<int, int, float>* current_stream, Task current_task);

template
void add_new_idle_stream(tbb::concurrent_bounded_queue<stream_container<int,int,float>*>* streams, stream_container <int, int, float>* stream);

void CUDART_CB call_back(cudaStream_t Stream, cudaError_t err, void* data);

