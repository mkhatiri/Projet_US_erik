#include <cuda_runtime_api.h>
#include <iostream>
#include <list>
#include "tbb/concurrent_queue.h"
#include "math.h"
#include "timestamp.hpp"
#include "AdaptativeUtils.hpp"
#include <stdio.h>
#include <float.h>


__device__ float two_fma( const float x_vals, const float x_vec, float y, float * sumk_err )
{
	float x = x_vals * x_vec;
	const float sumk_s = x + y;
	if (fabs(x) < fabs(y))
	{
		const float swap = x;
		x = y;
		y = swap;
	}
	(*sumk_err) += (y - (sumk_s - x));
	// 2Sum in the FMA case. Poor performance on low-DPFP GPUs.
	//const float bp = fma(-x_vals, x_vec, sumk_s);
	//(*sumk_err) += (fma(x_vals, x_vec, -(sumk_s - bp)) + (y - bp));
	return sumk_s;
}



__device__ float two_sum( float x, float y, float *sumk_err )
{
	const float sumk_s = x + y;
	// We use this 2Sum algorithm to perform a compensated summation,
	// which can reduce the cummulative rounding errors in our SpMV summation.
	// Our compensated sumation is based on the SumK algorithm (with K==2) from
	// Ogita, Rump, and Oishi, "Accurate Sum and Dot Product" in
	// SIAM J. on Scientific Computing 26(6) pp 1955-1988, Jun. 2005.
	// 2Sum can be done in 6 FLOPs without a branch. However, calculating
	// double precision is slower than single precision on every existing GPU.
	// As such, replacing 2Sum with Fast2Sum when using DPFP results in better
	// performance (~5% higher total). This is true even though we must ensure
	// that |a| > |b|. Branch divergence is better than the DPFP slowdown.
	// Thus, for DPFP, our compensated summation algorithm is actually described
	// by both Pichat and Neumaier in "Correction d'une somme en arithmetique
	// a virgule flottante" (J. Numerische Mathematik 19(5) pp. 400-406, 1972)
	// and "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher
	// Summen (ZAMM Z. Angewandte Mathematik und Mechanik 54(1) pp. 39-51,
	// 1974), respectively.
	if (fabs(x) < fabs(y))
	{
		const float swap = x;
		x = y;
		y = swap;
	}
	(*sumk_err) += (y - (sumk_s - x));
	// Original 6 FLOP 2Sum algorithm.
	//const float bp = sumk_s - x;
	//(*sumk_err) += ((x - (sumk_s - bp)) + (y - bp));
	return sumk_s;
}

__device__ float sum2_reduce( float cur_sum,   float * err, 
		float *  partial,
		const unsigned int lid,
		const unsigned int thread_lane,
		const unsigned int max_size,
		const unsigned int reduc_size )
{
	if ( max_size > reduc_size )
	{
		//#ifdef EXTENDED_PRECISION
		const unsigned int partial_dest = lid + reduc_size;
		if (thread_lane < reduc_size)
			cur_sum  = two_sum(cur_sum, partial[partial_dest], err);
		// We reuse the LDS entries to move the error values down into lower
		// threads. This saves LDS space, allowing higher occupancy, but requires
		// more barriers, which can reduce performance.
		__syncthreads();
		//barrier(CLK_LOCAL_MEM_FENCE);
		// Have all of those upper threads pass their temporary errors
		// into a location that the lower threads can read.
		partial[lid] = (thread_lane >= reduc_size) ? *err : partial[lid];
		__syncthreads();

		//barrier(CLK_LOCAL_MEM_FENCE);
		if (thread_lane < reduc_size) // Add those errors in.
		{
			*err += partial[partial_dest];
			partial[lid] = cur_sum;
		}
		//#else
		//        cur_sum += partial[lid + reduc_size];
		//        barrier( CLK_LOCAL_MEM_FENCE );
		//        partial[lid] = cur_sum;
		//#endif
	}
	return cur_sum;
}

static __inline__ __device__ float cuMax(float a, float b)
{
	return a < b ? b : a;
}


__device__ float atomic_add_float_extended( float * ptr,  float temp, float * old_sum ){
	unsigned int *address_as_ul = reinterpret_cast<unsigned int *>(ptr);
	unsigned int old = *address_as_ul, assumed;


	//	printf(" 1%dX%d *adr=%1.9f, temp=%1.9f \n",blockIdx.x, threadIdx.x, * ptr, temp );
	do
	{
		assumed = old;
		old = atomicCAS(address_as_ul, assumed, __float_as_int(temp + __int_as_float(assumed)));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);
	if (old_sum != 0)
		*old_sum = __int_as_float(old);


	//	printf(" 2%dX%d,  *adr=%1.9f, temp=%1.9f, \n", blockIdx.x, threadIdx.x, *ptr, temp);
	return __int_as_float(old);
}

/*__device__ float atomic_add_float_extended44( float * ptr,  float temp, float * old_sum ){
	unsigned int newVal;
	unsigned int prevVal;

	do{
		prevVal = __float_as_uint(*ptr);
		newVal =  __float_as_uint( __int_as_float(temp) + *ptr);
		if(threadIdx.x == 0)
		printf(" Bid%dXTid%d prevVal=%d, newVal=%d \n", blockIdx.x, threadIdx.x , __int_as_float(prevVal) ,__int_as_float(newVal));
	}while( atomicCAS((unsigned int *)(ptr) , prevVal, newVal) != prevVal);
	if (old_sum != 0)
		*old_sum = __int_as_float(prevVal);
	return __int_as_float(newVal);


}

*/


__device__ float atomicMaxFloat(float *address, float val)
{
	unsigned int *address_as_ul = reinterpret_cast<unsigned int *>(address);
	unsigned int old = *address_as_ul, assumed;

	do
	{
		assumed = old;
		old = atomicCAS(address_as_ul, assumed, __float_as_int(cuMax(val, __int_as_float(assumed))));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __int_as_float(old);
}



__device__ float atomic_two_sum_float( float * x_ptr,
		float y,
		float * sumk_err )
{
	// Have to wait until the return from the atomic op to know what X was.
	float sumk_s = 0.;
	float x;
		//printf(" 1-atomic_two_sum_float BID%d  x_ptr=%0.9f y=%0.9f x=%0.9f sumk_s=%0.9f \n  ", blockIdx.x, * x_ptr, y, x, sumk_s);
	sumk_s = atomic_add_float_extended(x_ptr, y, &x);
	//if(threadIdx.x == 0)
	//	printf(" 2-atomic_two_sum_float BID%d  x_ptr=%0.9f y=%0.9f x=%0.9f sumk_s=%0.9f \n  ", blockIdx.x, * x_ptr, y, x, sumk_s);
	if (fabs(x) < fabs(y))
	{
		const float swap = x;
		x = y;
		y = swap;
	}
	(*sumk_err) += (y - (sumk_s - x));
	return sumk_s;
}



__device__ void csr_stream(float* partialSums, float* vals, int* cols, int* rowPtrs, float* vec, float* out, unsigned long* rowBlocks, float alpha, float beta, const unsigned int BLOCKSIZE , const unsigned int ROWS_FOR_VECTOR, const unsigned int BLOCK_MULTIPLIER, const unsigned int Bid, const unsigned Tid, unsigned int row, unsigned int stop_row, unsigned int wg, int WG_SIZE, float temp_sum, float sumk_e, float new_error, int* method  ){


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
			{	
				//temp_sum = two_sum(partialSums[local_cur_val], temp_sum, &sumk_e);
				temp_sum += partialSums[local_cur_val] ; 
			}

		}
		__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       
		//temp_sum = two_sum(temp_sum, sumk_e, &new_error);
		partialSums[Tid] = temp_sum;
		
		// Step one of this two-stage reduction is done. Now each row has {numThreadsForRed}
		// values sitting in the local memory. This means that, roughly, the beginning of
		// LDS is full up to {workgroup size} entries.
		// Now we perform a parallel reduction that sums together the answers for each
		// row in parallel, leaving us an answer in 'temp_sum' for each row.
		//	for (unsigned long i = (WG_SIZE >> 1); i > 0; i >>= 1)
		for (int i = (WG_SIZE >> 1); i > 0; i >>= 1)
		{
			__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE); 

			//	/	if ( numThreadsForRed > i ){
			//	temp_sum += partialSums[Tid + i];
			//	__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       
			//	partialSums[Tid] = temp_sum;
			//
			//			}	
			temp_sum = sum2_reduce(temp_sum, &new_error, partialSums, Tid, threadInBlock, numThreadsForRed, i);
		}

		if (threadInBlock == 0 && local_row < stop_row)
		{
			// All of our write-outs check to see if the output vector should first be zeroed.
			// If so, just do a write rather than a read-write. Measured to be a slight (~5%)
			// performance improvement.
			if (beta != 0.)
				temp_sum = two_fma(beta, out[local_row], temp_sum, &new_error);
			//out[local_row] = beta* out[local_row] + temp_sum;
			//	else
			out[local_row] = temp_sum + new_error;

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
			sumk_e = 0.;
			for (int local_cur_val = local_first_val; local_cur_val < local_last_val; local_cur_val++)
				temp_sum = two_sum(partialSums[local_cur_val], temp_sum, &sumk_e);
			//temp_sum += partialSums[local_cur_val];

			// After you've done the reduction into the temp_sum register,
			// put that into the output for each row.
			if (beta != 0.)
				//		out[local_row] = beta* out[local_row] + temp_sum;
				temp_sum = two_fma(beta, out[local_row], temp_sum, &sumk_e);
			//	else
			//	out[local_row] = temp_sum;

			out[local_row] = temp_sum + sumk_e;
			local_row += WG_SIZE;
		}
	}
}

__device__ void csr_vector(float* partialSums, float* vals, int* cols, int* rowPtrs, float* vec, float* out,
		unsigned long* rowBlocks, float alpha, float beta, const unsigned int BLOCKSIZE , const unsigned int ROWS_FOR_VECTOR, const unsigned int BLOCK_MULTIPLIER, const unsigned int Bid, const unsigned Tid, unsigned int row, unsigned int stop_row, unsigned int wg, int WG_SIZE   , float temp_sum, float sumk_e, float new_error ){

	while (row < stop_row){

		temp_sum = 0.;
		sumk_e = 0.;
		new_error = 0.;
		// Load in a bunch of partial results into your register space, rather than LDS (no contention)
		// Then dump the partially reduced answers into the LDS for inter-work-item reduction.
		// Using a long induction variable to make sure unsigned int overflow doesn't break things.

		unsigned int vecStart = rowPtrs[row];
		unsigned int vecEnd = rowPtrs[row+1];

		// Load in a bunch of partial results into your register space, rather than LDS (no contention)
		// Then dump the partially reduced answers into the LDS for inter-work-item reduction.
		// Using a long induction variable to make sure unsigned int overflow doesn't break things
		//printf("******* row=%d \n", row);
		for (int j = vecStart + Tid; j < vecEnd; j+=WG_SIZE)
		{
			const unsigned int col = cols[(unsigned int)j];
			temp_sum = two_fma(alpha*vals[(unsigned int)j], vec[col], temp_sum, &sumk_e);
			//temp_sum += alpha*vals[(unsigned int)j]*vec[col];
		}
		temp_sum = two_sum(temp_sum, sumk_e, &new_error);
		partialSums[Tid] = temp_sum;
		//printf("TID=%d temp_sum=%0.3e, sumk_e=%0.3e new_error=%0.3e \n", Tid , temp_sum, sumk_e, new_error );

		// Reduce partial sums

		//	for (unsigned long i = (WG_SIZE >> 1); i > 0; i >>= 1)
		for (int i = (WG_SIZE >> 1); i > 0; i >>= 1)
		{
			__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       
			temp_sum = sum2_reduce(temp_sum, &new_error, partialSums, Tid, Tid, WG_SIZE, i);
	/*		if ( WG_SIZE > i ){
			  temp_sum += partialSums[Tid + i];
			  __syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       
			  partialSums[Tid] = temp_sum;
			  }  
	*/	}

		if (Tid == 0UL)
		{
			if (beta != 0.)
				//	out[row] = beta* out[row] + temp_sum;
				temp_sum = two_fma(beta, out[row], temp_sum, &sumk_e);
			//	else
			out[row] = temp_sum + new_error;

		}
		row++;
	}

}

__device__ void csr_vectorL(float* partialSums, float* vals, int* cols, int* rowPtrs, float* vec, float* out,
		unsigned long* rowBlocks, float alpha, float beta, const unsigned int BLOCKSIZE , const unsigned int ROWS_FOR_VECTOR, const unsigned int BLOCK_MULTIPLIER, const unsigned int Bid, const unsigned Tid, unsigned int row, unsigned int stop_row, unsigned int wg, unsigned int vecStart,  unsigned int vecEnd, int WG_SIZE , float *temp_sum, float sumk_e, float new_error, float *rowErr ){

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
	//	printf("%d ici rowerr=%f", rowErr[0] );
	//	temp_sum = 0.;
	//	sumk_e = 0.;
	//	new_error = 0.;
	const unsigned int first_wg_in_row = Bid - (rowBlocks[Bid] & ((1UL << 24) - 1UL)); //  WGBITS = 24
	const unsigned int compare_value = rowBlocks[Bid] & (1UL << 24);


	// Bit 24 in the first workgroup is the flag that everyone waits on.
	if(Bid == first_wg_in_row && Tid == 0UL)
	{
		// The first workgroup handles the output initialization.
		volatile float out_val = out[row];
		*temp_sum = (beta - 1.) * out_val;
		rowErr[gridDim.x + Bid + 1] = 0UL;

		atomicXor( (unsigned int*)  &rowBlocks[first_wg_in_row], (unsigned int) (1UL << 24)); // Release other workgroups.
		//atomicXor(&rowBlocks[first_wg_in_row], (1UL << 24)); // Release other workgroups.
	}

	// For every other workgroup, bit 24 holds the value they wait on.
	// If your bit 24 == first_wg's bit 24, you spin loop.
	// The first workgroup will eventually flip this bit, and you can move forward.
	__syncthreads(); // barrier(CLK_GLOBAL_MEM_FENCE );       
// __threadfence_block() ;

	while(Bid != first_wg_in_row && 
			Tid == 0U && 
			((atomicMax(( unsigned int*) &rowBlocks[first_wg_in_row],(unsigned int) 0UL)  & (1UL << 24)  ) == compare_value)); //WGBITS = 24
// __threadfence_block(); 
	__syncthreads(); // barrier(CLK_GLOBAL_MEM_FENCE );       

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
			*temp_sum = two_fma(alpha*vals[col + j], vec[cols[col + j]], *temp_sum, &sumk_e);
			//		temp_sum += alpha*vals[col + j]*vec[cols[col + j]];


				if(2*WG_SIZE <= BLOCK_MULTIPLIER*BLOCKSIZE){
					j += WG_SIZE;
					*temp_sum = two_fma(alpha*vals[col + j], vec[cols[col + j]], *temp_sum, &sumk_e);
				}


		}
	}
	else
	{
		for(int j = 0; j < (int)(vecEnd - col); j += WG_SIZE)
			*temp_sum = two_fma(alpha*vals[col + j], vec[cols[col + j]], *temp_sum, &sumk_e);
		//O	temp_sum += alpha*vals[col + j]*vec[cols[col + j]];
	}
//	printf("A-Bid%dXTID%d , temps_sum=%1.9f, new_error=%1.9f, sumk_e=%1.9f \n", Bid, Tid, *temp_sum, new_error, sumk_e);

	*temp_sum = two_sum(*temp_sum, sumk_e, &new_error);
//	printf("B-Bid%dXTID%d , temps_sum=%1.9f, new_error=%1.9f, sumk_e=%1.9f \n", Bid, Tid, *temp_sum, new_error, sumk_e);

	partialSums[Tid] = *temp_sum;

	// Reduce partial sums
	for (int i = (WG_SIZE >> 1); i > 0; i >>= 1)
	{
		__syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);       
		*temp_sum = sum2_reduce(*temp_sum, &new_error, partialSums, Tid, Tid, WG_SIZE, i);
	}


	if (Tid == 0UL)
	{
		atomic_two_sum_float(&out[row], *temp_sum, &new_error);
		unsigned int error_loc = gridDim.x + first_wg_in_row + 1;
		atomic_add_float_extended((float *) &(rowErr[error_loc]), new_error, 0);

		if (row != stop_row)
		{	

			do{
				printf("Bid=%d max=%d, wg=%d\n", Bid, (int) rowErr[first_wg_in_row], wg );
			}	
			while( rowErr[first_wg_in_row] !=  wg );
			
//
		//	while( atomicMax((int *) &rowErr[first_wg_in_row]  ,0) != 0  ){
		//		printf("rowErr[first_wg_in_row]=%.9f", rowErr[first_wg_in_row]);
		//		}

			new_error = rowErr[error_loc];
		//	printf("\t\t1-Bid%d, row=%d, out[row]=%.*e , new_error=%1.9f , wg=%d\n", Bid, row, out[row], new_error, wg);
			out[row] += new_error;
		//	printf("\t\t2-Bid%d, row=%d, out[row]=%.*e, new_error=%1.9f , wg=%d\n", Bid, row, out[row], new_error, wg);
			rowErr[error_loc] = 0UL;
	//		rowBlocks[first_wg_in_row] = rowBlocks[Bid] - wg;
		}else{
		//	printf("1-else BID%d , wg=%d max=%f \n", Bid , (int)  wg, (cuMax(rowErr[first_wg_in_row], 0)), cuMax(rowErr[first_wg_in_row], 0)  );
			atomic_add_float_extended((float *) &(rowErr[first_wg_in_row]), 1, 0); 
	//		atomicInc((unsigned int *) &rowErr[first_wg_in_row], 1UL);
		//	printf("2-else BID%d , wg=%d max=%f \n", Bid, (int) wg,  (cuMax(rowErr[first_wg_in_row], 0)), cuMax(rowErr[first_wg_in_row], 0) );



		}
	}
}



__global__ void csr_adaptative(float* vals, int* cols, int* rowPtrs, float* vec, float* out,
		unsigned long* rowBlocks, float* d_alpha, float* d_beta, unsigned int* d_blkSize, 
		unsigned int* d_blkMultiple, unsigned int* d_rowForVector, int rowBlockSize, int * method, float * rowErr){

	const unsigned int blkSize = *d_blkSize;
	const unsigned int blkMultiple = *d_blkMultiple;
	const unsigned int rowForVector = *d_rowForVector;

	extern __shared__ float partialSums[];
	int Bid = blockIdx.x ;
	int Tid = threadIdx.x;
	const float alpha = *d_alpha;
	const float beta = *d_beta;

	int WGSIZE = blockDim.x;
	float temp_sum = 0.;
	float sumk_e = 0.;
	float new_error = 0.;
	

	if (Bid < rowBlockSize) {

		unsigned int row = ((rowBlocks[Bid] >> 32) & ((1UL << 32) - 1UL));	 // OWBITS = 32
		unsigned int stop_row = ((rowBlocks[Bid + 1] >> 32) & ((1UL << 32) - 1UL));
		unsigned int num_rows = stop_row - row;


		unsigned int wg = rowBlocks[Bid] & ((1 << 24) - 1);  // WGBITS = 24

		unsigned int vecStart = wg*(unsigned int)(blkSize*blkMultiple) + rowPtrs[row];
		unsigned int vecEnd = (rowPtrs[row + 1] > vecStart + blkSize*blkMultiple) ? vecStart + blkSize*blkMultiple : rowPtrs[row+1];

	//	printf("Tid=%d row=%d, stop_row=%d \n ", Tid, row, stop_row);
		
		   if (num_rows == 0 || (num_rows == 1 && wg)) // CSR-LongRows case
		   {
		   num_rows = rowForVector;
		   stop_row = (wg ? row : (row + 1));
		   wg = 0;
		//	tab[Bid] = 15;	
		}
		
		if(row <= stop_row){

			if (num_rows > rowForVector ) //CSR-Stream case
			{
				atomicAdd(&method[0], 1);
				//		if(Tid==0)
				//printf("stream : BID=%d, TID=%d \n", Bid, Tid);
				csr_stream(partialSums, vals, cols, rowPtrs, vec, out, rowBlocks, alpha, beta, blkSize, rowForVector, blkMultiple, Bid, Tid, row, stop_row, wg, WGSIZE, temp_sum, sumk_e, new_error, method);
			}else if (num_rows >= 1 && !wg){ // CSR-Vector case.
				atomicAdd(&method[1], 1);
				//		if(Tid==0)
				//		printf("Vector : BID=%d, TID=%d \n", Bid, Tid);
				csr_vector(partialSums, vals, cols, rowPtrs, vec, out, rowBlocks, alpha, beta, blkSize, rowForVector, blkMultiple, Bid, Tid, row, stop_row, wg, WGSIZE, temp_sum, sumk_e, new_error);
			}else{ //CSR-LongRows
				atomicAdd(&method[2], 1);
				//	if(Tid==0)
				//	printf("VL : BID=%d, TID=%d \n", Bid, Tid);
				csr_vectorL(partialSums, vals, cols, rowPtrs, vec, out, rowBlocks, alpha, beta, blkSize, rowForVector, blkMultiple, Bid, Tid, row, stop_row, wg, vecStart, vecEnd, WGSIZE, &temp_sum, sumk_e, new_error, rowErr);
			}
		}
	}else{
		atomicAdd(&method[3], 1);
	}
}
__global__ void csr_adaptativeT(int *a){

	int index = blockIdx.x*blockDim.x +  threadIdx.x;
	a[index] = __clz(512)  ;


}
