#include <cuda_runtime_api.h>
#include <iostream>
#include <list>
#include "tbb/concurrent_queue.h"
#include "math.h"
#include "timestamp.hpp"
#include "AdaptativeUtils.hpp"
#include <stdio.h>



__device__ void csr_stream(double* partialSums, double* vals, int* cols, int* rowPtrs, double* vec, double* out, unsigned long* rowBlocks, double alpha, double beta, const unsigned int BLOCKSIZE , const unsigned int ROWS_FOR_VECTOR, const unsigned int BLOCK_MULTIPLIER, const unsigned int Bid, const unsigned Tid, unsigned int row, unsigned int stop_row, unsigned int wg, int WG_SIZE  ){

	double temp_sum = 0.;
	//int WG_SIZE = blockDim.x;
	const unsigned int numThreadsForRed = wg;
	const unsigned int col = rowPtrs[row] + Tid;

	if (Bid != (gridDim.x - 1))
	{
		for(int i = 0; i < BLOCKSIZE; i += WG_SIZE)
			partialSums[Tid + i] = alpha * vals[col + i] * vec[cols[col + i]];
	}
	else
	{
		// This is required so that we stay in bounds for vals[] and cols[].
		// Otherwise, if the matrix's endpoints don't line up with BLOCKSIZE,
		// we will buffer overflow. On today's dGPUs, this doesn't cause problems.
		// The values are within a dGPU's page, which is zeroed out on allocation.
		// However, this may change in the future (e.g. with shared virtual memory.)
		// This causes a minor performance loss because this is the last workgroup
		// to be launched, and this loop can't be unrolled.
		const unsigned int max_to_load = rowPtrs[stop_row] - rowPtrs[row];
		for(int i = 0; i < max_to_load; i += WG_SIZE)
			partialSums[Tid + i] = alpha * vals[col + i] * vec[cols[col + i]];
	}
	__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);  	

	if(numThreadsForRed > 1)
	{
		const unsigned int local_row = row + (Tid >> (31 - __clz(numThreadsForRed)));
		const unsigned int local_first_val = rowPtrs[local_row] - rowPtrs[row];
		const unsigned int local_last_val = rowPtrs[local_row + 1] - rowPtrs[row];
		const unsigned int threadInBlock = Tid & (numThreadsForRed - 1);

		if(local_row < stop_row)
		{
			// This is dangerous -- will infinite loop if your last value is within
			// numThreadsForRed of MAX_UINT. Noticable performance gain to avoid a
			// long induction variable here, though.
			for(unsigned int local_cur_val = local_first_val + threadInBlock; local_cur_val < local_last_val; local_cur_val += numThreadsForRed)
			//	temp_sum += partialSums[local_cur_val] ; 	
				temp_sum = two_sum(partialSums[local_cur_val], temp_sum, &sumk_e);
			
				
		}
		__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       

		partialSums[Tid] = temp_sum;


		// Step one of this two-stage reduction is done. Now each row has {numThreadsForRed}
		// values sitting in the local memory. This means that, roughly, the beginning of
		// LDS is full up to {workgroup size} entries.
		// Now we perform a parallel reduction that sums together the answers for each
		// row in parallel, leaving us an answer in 'temp_sum' for each row.
		for (unsigned long i = (WG_SIZE >> 1); i > 0; i >>= 1)
		{
			__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE); 

			if ( numThreadsForRed > i ){
				temp_sum += partialSums[Tid + i];
				__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       
				partialSums[Tid] = temp_sum;

			}	
		}

		if (threadInBlock == 0 && local_row < stop_row)
		{
			// All of our write-outs check to see if the output vector should first be zeroed.
			// If so, just do a write rather than a read-write. Measured to be a slight (~5%)
			// performance improvement.
			if (beta != 0.)
				temp_sum = std::fma(beta, out[local_row], temp_sum);
			//	out[local_row] = beta* out[local_row] + temp_sum;
			else
			//	out[local_row] = temp_sum;
				out[local_row] = temp_sum + sumk_e;

		}
	}else{
		// In this case, we want to have each thread perform the reduction for a single row.
		// Essentially, this looks like performing CSR-Scalar, except it is computed out of local memory.
		// However, this reduction is also much faster than CSR-Scalar, because local memory
		// is designed for scatter-gather operations.
		// We need a while loop because there may be more rows than threads in the WG.
		unsigned int local_row = row + Tid;
		while(local_row < stop_row)
		{
			int local_first_val = (rowPtrs[local_row] - rowPtrs[row]);
			int local_last_val = rowPtrs[local_row + 1] - rowPtrs[row];
			temp_sum = 0.;
			for (int local_cur_val = local_first_val; local_cur_val < local_last_val; local_cur_val++)
				temp_sum += partialSums[local_cur_val];

			// After you've done the reduction into the temp_sum register,
			// put that into the output for each row.
			if (beta != 0.)
			//	out[local_row] = beta* out[local_row] + temp_sum;
				temp_sum = two_fma(beta, out[local_row], temp_sum, &sumk_e);
			else
				out[local_row] = temp_sum + sumk_e;
				//out[local_row] = temp_sum;

			local_row += WG_SIZE;
		}
	}
}

__device__ void csr_vector(double* partialSums, double* vals, int* cols, int* rowPtrs, double* vec, double* out,
		unsigned long* rowBlocks, double alpha, double beta, const unsigned int BLOCKSIZE , const unsigned int ROWS_FOR_VECTOR, const unsigned int BLOCK_MULTIPLIER, const unsigned int Bid, const unsigned Tid, unsigned int row, unsigned int stop_row, unsigned int wg, int WG_SIZE   ){

	double temp_sum = 0.;

	while (row < stop_row){
		temp_sum = 0.;

		// Load in a bunch of partial results into your register space, rather than LDS (no contention)
		// Then dump the partially reduced answers into the LDS for inter-work-item reduction.
		// Using a long induction variable to make sure unsigned int overflow doesn't break things.

		unsigned int vecStart = rowPtrs[row];
		unsigned int vecEnd = rowPtrs[row+1];

		for (long j = vecStart + Tid; j < vecEnd; j+=WG_SIZE)
		{
			const unsigned int col = cols[(unsigned int)j];
			temp_sum += alpha*vals[(unsigned int)j]*vec[col];
		}

		partialSums[Tid] = temp_sum;

		// Reduce partial sums

		for (unsigned long i = (WG_SIZE >> 1); i > 0; i >>= 1)
		{
			__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       
			//temp_sum = sum2_reduce(temp_sum, &new_error, partialSums, lid, lid, WG_SIZE, i);
			if ( WG_SIZE > i ){
				temp_sum += partialSums[Tid + i];
				__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       
				partialSums[Tid] = temp_sum;
			}  
		}

		if (Tid == 0UL)
		{
			if (beta != 0.)
			//	out[row] = beta* out[row] + temp_sum;
			        temp_sum = two_fma(beta, out[local_row], temp_sum, &sumk_e);
			else
				out[row] = temp_sum;

		}
		row++;
	}

}

__device__ void csr_vectorL(double* partialSums, double* vals, int* cols, int* rowPtrs, double* vec, double* out,
		unsigned long* rowBlocks, double alpha, double beta, const unsigned int BLOCKSIZE , const unsigned int ROWS_FOR_VECTOR, const unsigned int BLOCK_MULTIPLIER, const unsigned int Bid, const unsigned Tid, unsigned int row, unsigned int stop_row, unsigned int wg, unsigned int vecStart,  unsigned int vecEnd, int WG_SIZE ){

	// In CSR-LongRows, we have more than one workgroup calculating this row.
	// The output values for those types of rows are stored using atomic_add, because
	// more than one parallel workgroup's value makes up the final answer.
	// Unfortunately, this makes it difficult to do y=Ax, rather than y=Ax+y, because
	// the values still left in y will be added in using the atomic_add.
	//
	// Our solution is to have the first workgroup in one of these long-rows cases
	// properly initaizlie the output vector. All the other workgroups working on this
	// row will spin-loop until that workgroup finishes its work.

	// First, figure out which workgroup you are in the row. Bottom 24 bits.
	// You can use that to find the global ID for the first workgroup calculating
	// this long row.

	double temp_sum = 0.;
	const unsigned int first_wg_in_row = Bid - (rowBlocks[Bid] & ((1UL << 24) - 1UL)); //  WGBITS = 24
	const unsigned int compare_value = rowBlocks[Bid] & (1UL << 24);


	// Bit 24 in the first workgroup is the flag that everyone waits on.
	if(Bid == first_wg_in_row && Tid == 0UL)
	{
		// The first workgroup handles the output initialization.
		volatile double out_val = out[row];
		temp_sum = (beta - 1.) * out_val;
		atomicXor( (unsigned int*)  &rowBlocks[first_wg_in_row], (unsigned int) (1UL << 24)); // Release other workgroups.
	}

	// For every other workgroup, bit 24 holds the value they wait on.
	// If your bit 24 == first_wg's bit 24, you spin loop.
	// The first workgroup will eventually flip this bit, and you can move forward.
	__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       

	while(Bid != first_wg_in_row && 
			Tid == 0U && 
			((atomicMax((unsigned int*) &rowBlocks[first_wg_in_row],(unsigned int) 0UL) & (1UL << 24)) == compare_value)); //WGBITS = 24

	__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       

	// After you've passed the barrier, update your local flag to make sure that
	// the next time through, you know what to wait on.
	if (Bid != first_wg_in_row && Tid == 0UL)
		rowBlocks[Bid] ^= (1UL << 24); //  WGBITS = 24


	// All but the final workgroup in a long-row collaboration have the same start_row
	// and stop_row. They only run for one iteration.
	// Load in a bunch of partial results into your register space, rather than LDS (no contention)
	// Then dump the partially reduced answers into the LDS for inter-work-item reduction.
	//	unsigned int vecStart = wg*(unsigned int)(BLOCK_MULTIPLIER*BLOCKSIZE) + rowPtrs[row];
	//	unsigned int vecEnd = (rowPtrs[row + 1] > vecStart + BLOCK_MULTIPLIER*BLOCKSIZE) ? vecStart + BLOCK_MULTIPLIER*BLOCKSIZE : rowPtrs[row+1];

	const unsigned int col = vecStart + Tid;

	if (row == stop_row) // inner thread, we can hardcode/unroll this loop
	{
		// Don't put BLOCK_MULTIPLIER*BLOCKSIZE as the stop point, because
		// some GPU compilers will *aggressively* unroll this loop.
		// That increases register pressure and reduces occupancy.
		for (int j = 0; j < (int)(vecEnd - col); j += WG_SIZE)
		{
			temp_sum += alpha*vals[col + j]*vec[cols[col + j]];
		}
	}
	else
	{
		for(int j = 0; j < (int)(vecEnd - col); j += WG_SIZE)
			temp_sum += alpha*vals[col + j]*vec[cols[col + j]];
	}

	partialSums[Tid] = temp_sum;

	// Reduce partial sums
	for (unsigned long i = (WG_SIZE >> 1); i > 0; i >>= 1)
	{
		__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       
		//	temp_sum = sum2_reduce(temp_sum, &new_error, partialSums, lid, lid, WG_SIZE, i);
		if ( WG_SIZE > i ){
			temp_sum += partialSums[Tid + i];
			__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       
			partialSums[Tid] = temp_sum;
		}

	}


	if (Tid == 0UL)
	{
		atomicAdd(&out[row], temp_sum);

	}



}



__global__ void csr_adaptative(double* vals, int* cols, int* rowPtrs, double* vec, double* out,
		unsigned long* rowBlocks, double* d_alpha, double* d_beta, unsigned int* d_blkSize, 
		unsigned int* d_blkMultiple, unsigned int* d_rowForVector, int rowBlockSize, int * method){

	const unsigned int blkSize = *d_blkSize;
	const unsigned int blkMultiple = *d_blkMultiple;
	const unsigned int rowForVector = *d_rowForVector;

	extern __shared__ double partialSums[];
	int Bid = blockIdx.x ;
	int Tid = threadIdx.x;
	const double alpha = *d_alpha;
	const double beta = *d_beta;

	int WGSIZE = blockDim.x;

	if (Bid < rowBlockSize - 1) {
		unsigned int row = ((rowBlocks[Bid] >> 32) & ((1UL << 32) - 1UL));	 // OWBITS = 32
		unsigned int stop_row = ((rowBlocks[Bid + 1] >> 32) & ((1UL << 32) - 1UL));
		unsigned int num_rows = stop_row - row;


		unsigned int wg = rowBlocks[Bid] & ((1 << 24) - 1);  // WGBITS = 24

		unsigned int vecStart = wg*(unsigned int)(blkSize*blkMultiple) + rowPtrs[row];
		unsigned int vecEnd = (rowPtrs[row + 1] > vecStart + blkSize*blkMultiple) ? vecStart + blkSize*blkMultiple : rowPtrs[row+1];

	
		if(Tid == 0 && Bid == 1)
				atomicAdd(&method[1], 1);

		/*	if (num_rows == 0 || (num_rows == 1 && wg)) // CSR-LongRows case
			{
			num_rows = rowForVector;
			stop_row = (wg ? row : (row + 1));
			wg = 0;
		//	tab[Bid] = 15;	
		}*/

		if(row <= stop_row){
			if (num_rows > rowForVector ) //CSR-Stream case
			{

				atomicAdd(&method[0], 1);
				csr_stream(partialSums, vals, cols, rowPtrs, vec, out, rowBlocks, alpha, beta, blkSize, rowForVector, blkMultiple, Bid, Tid, row, stop_row, wg, WGSIZE);
			}else if (num_rows >= 1 && !wg){ // CSR-Vector case.
				atomicAdd(&method[1], 1);
				//	csr_vector(partialSums, vals, cols, rowPtrs, vec, out, rowBlocks, alpha, beta, blkSize, rowForVector, blkMultiple, Bid, Tid, row, stop_row, wg, WGSIZE);
			}else{ //CSR-LongRows
				//	csr_vectorL(partialSums, vals, cols, rowPtrs, vec, out, rowBlocks, alpha, beta, blkSize, rowForVector, blkMultiple, Bid, Tid, row, stop_row, wg, vecStart, vecEnd, WGSIZE);
			}
		}
	}else{
		atomicAdd(&method[2], 1);
	}
}
	__global__ void csr_adaptativeT(int *a){

		int index = blockIdx.x*blockDim.x +  threadIdx.x;
		a[index] = __clz(512)  ;


	}
