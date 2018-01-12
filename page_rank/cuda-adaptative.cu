#include "main-pr.hpp"

#define THROW_AWAY 0
#include "Padded2DArray.hpp"
#include <omp.h>
#include "memutils.hpp"
#include <cmath>

//#define SHOWLOADBALANCE
#include "logged_array.hpp"

//#define LOG
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "helper_cuda.h"
#include "math.h"
#include "tbb/concurrent_queue.h"
#include "math.h"
#include "AdaptativeUtils.hpp"

	template <typename VertexType, typename EdgeType, typename Scalar>
int main_pr(VertexType nVtx, EdgeType* xadj_, VertexType *adj_, Scalar* val_, Scalar *prior_, Scalar* pr_,
		Scalar lambda,
		int nTry, //algo parameter
		util::timestamp& totaltime, std::string& 
	   )
{


	int blk_size = 0;
	int nb_threads = 0;

	{
		char* str = getenv ("NBTHREAD");
		if (str) {
			std::stringstream ss (str);
			ss>>nb_threads;
			if (!ss)
				std::cerr<<"NBTHREAD invalid"<<std::endl;
		}
	}



	{
		char* str = getenv ("BLKSIZE");
		if (str) {
			std::stringstream ss (str);
			ss>>blk_size;
			if (!ss)
				std::cerr<<"SUBSIZE invalid"<<std::endl;
		}
	}


	if(nb_threads == 0 && blk_size == 0){
		std::cerr<<"BLKSIZE=??? or  NBTHREAD=???"<<std::endl;
		exit(0);
	}



	bool coldcache = true;

	util::timestamp start(0,0);

	//cpuside variables  
	Scalar* prin_ = new Scalar[nVtx];
	EdgeType* xadj = xadj_;
	VertexType *adj = adj_;
	Scalar* val = val_;
	Scalar* prior = prior_;
	Scalar* prin = prin_;
	Scalar* prout = pr_;
	Scalar alpha = lambda;
	Scalar beta = 1-lambda;


	//cuda side variable
	EdgeType* d_xadj ;
	VertexType *d_adj ;
	Scalar* d_val ;
	Scalar* d_prior ;
	Scalar* d_prin ;
	Scalar* d_prout ;
	Scalar *d_alpha;
	Scalar *d_beta;

	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);

	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	//memalloc

	checkCudaErrors( cudaMalloc((void**)&d_xadj, (nVtx+1)*sizeof(*xadj)) );
	checkCudaErrors( cudaMalloc((void**)&d_adj, (xadj[nVtx])*sizeof(*adj)) );
	checkCudaErrors( cudaMalloc((void**)&d_val, (xadj[nVtx])*sizeof(*val)) );
	checkCudaErrors( cudaMalloc((void**)&d_prior, (nVtx*sizeof(*prior))));
	checkCudaErrors( cudaMalloc((void**)&d_prin, (nVtx*sizeof(*prin)) ));
	checkCudaErrors( cudaMalloc((void**)&d_prout, (nVtx*sizeof(*prout)) ));

	//cpu to gpu copies

	checkCudaErrors( cudaMemcpy(d_xadj, xadj, (nVtx+1)*sizeof(*xadj), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_adj, adj, (xadj[nVtx])*sizeof(*adj), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_val, val, (xadj[nVtx])*sizeof(*val), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_prior, prior, nVtx*sizeof(*prior), cudaMemcpyHostToDevice) );



	int nRows = nVtx;
	unsigned long* rowBlocks;
	const int nThreadPerBlock = nb_threads; 
	const unsigned int blkSize = blk_size;
	const unsigned int blkMultiplier = 3 ;
	const unsigned int rows_for_vector = 2 ;
	const bool allocate_row_blocks = true;

	//device variable
	unsigned long* d_rowBlocks;
	unsigned int* d_blkSize;
	unsigned int* d_rows_for_vector;
	unsigned int* d_blkMultiplier;
	float* d_a;
	float* d_b;
	int rowBlockSize1;
	int rowBlockSize2;


	//calculer rowBlockSize
	rowBlockSize1 = ComputeRowBlocksSize<int,int>(xadj, nVtx, blkSize, blkMultiplier, rows_for_vector, nThreadPerBlock);
	//cout << "rowBlockSize1 : " << rowBlockSize1 << endl;

	//declarer rowBlocks
	rowBlocks = (unsigned long*) calloc(sizeof(unsigned long),rowBlockSize1);

	//calculer rowBlocks
	ComputeRowBlocks<int,int>( rowBlocks, rowBlockSize2, xadj, nVtx, blkSize, blkMultiplier, rows_for_vector, nThreadPerBlock, allocate_row_blocks);

	//	if(rowBlocks[rowBlockSize1] == 0){
	//		rowBlockSize1--;
	//	}



	//malloc for device variable
	checkCudaErrors( cudaMalloc((void**)&d_rowBlocks, (rowBlockSize1*sizeof(unsigned long))));
	checkCudaErrors( cudaMalloc((void**)&d_blkSize, 1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_rows_for_vector,1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_blkMultiplier, 1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_a, 1*sizeof(float)));
	checkCudaErrors( cudaMalloc((void**)&d_b, 1*sizeof(float)));


	//send data to device
	checkCudaErrors( cudaMemcpy(d_rowBlocks, rowBlocks, rowBlockSize1*sizeof(unsigned long), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_blkSize, &blkSize, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_rows_for_vector, &rows_for_vector, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_blkMultiplier, &blkMultiplier, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_a, &alpha, 1*sizeof(Scalar), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_b, &beta, 1*sizeof(Scalar), cudaMemcpyHostToDevice) );

	int mmshared_size =  (blkSize) * sizeof(float);



	//	csr_adaptative<<<(rowBlockSize1 + 1) , nThreadPerBlock, size >>>(d_val, d_adj, d_xadj, d_prior, d_prout1, d_rowBlocks, d_a,  d_b, d_blkSize, d_blkMultiplier, d_rows_for_vector, rowBlockSize1);



	for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
	{
		if (TRY >= THROW_AWAY)
			start = util::timestamp();

		for (int iter = 0; iter < 40 ; ++ iter) {

			//setup prin
			if (iter == 0)
				//std::copy (prior, prior+nVtx, prin);
				checkCudaErrors(cudaMemcpy(d_prin, d_prior, nVtx*sizeof(*prior), cudaMemcpyDeviceToDevice));
			else
				//std::copy (prout, prout+nVtx, prin);
				checkCudaErrors(cudaMemcpy(d_prin, d_prout, nVtx*sizeof(*prout), cudaMemcpyDeviceToDevice));

			Scalar eps = 0.;

			//prout = A prin
			//prout = lambda * prout + (1-lambda) prior

			checkCudaErrors(cudaMemcpy(d_prout, d_prior, nVtx*sizeof(*prior), cudaMemcpyDeviceToDevice));


			//for float it is S.
			//does prout = alpha A prin + beta prout
			csr_adaptative<<<(rowBlockSize1 + 1) , nThreadPerBlock, mmshared_size >>>(d_val, d_adj, d_xadj, d_prin, d_prout, d_rowBlocks, d_a,  d_b, d_blkSize, d_blkMultiplier, d_rows_for_vector, rowBlockSize1);


			cudaThreadSynchronize();

			cudaPrintError("after Kernel");

			if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
				std::cerr<<"err"<<std::endl;



			//compute epsilon
			//using prin to compute epsilon
			float epsalpha = -1;
			cublasStatus = cublasSaxpy (cublasHandle, nVtx, &epsalpha, d_prout, 1, d_prin, 1); // d_prin = d_prout*-1 + d_prin

			if (cublasStatus != CUBLAS_STATUS_SUCCESS)
				std::cerr<<"err 1"<<std::endl;

			cublasStatus = cublasSasum(cublasHandle, nVtx, d_prin, 1, &eps);
			if (cublasStatus != CUBLAS_STATUS_SUCCESS)
				std::cerr<<"err 2"<<std::endl;

			//stopping condition
			if (eps < 0) // deactivited for testing purposes
				iter = 20;

			std::cerr<<eps<<std::endl;

		}

		checkCudaErrors(cudaMemcpy(prout, d_prout, nVtx*sizeof(*prout), cudaMemcpyDeviceToHost));

		std::cerr<<"PR[0]="<<prout[0]<<std::endl;

		if (TRY >= THROW_AWAY)
		{
			util::timestamp stop;  
			totaltime += stop - start;
		}



		/*    
#ifndef LOG
if (coldcache) {
#pragma omp parallel
{
evict_array_from_cache(adj, xadj[nVtx]*sizeof(*adj));
evict_array_from_cache(xadj, (nVtx+1)*sizeof(*xadj));
evict_array_from_cache(val, xadj[nVtx]*sizeof(*val));
evict_array_from_cache(prior, nVtx*sizeof(*prior));
evict_array_from_cache(prin, nVtx*sizeof(*prin));
evict_array_from_cache(prout, nVtx*sizeof(*prout));

#pragma omp barrier
}
}
#endif
		 */
}


cudaFree(d_rowBlocks);
cudaFree(d_blkSize);
cudaFree(d_rows_for_vector);
cudaFree(d_blkMultiplier);
cudaFree(d_a);
cudaFree(d_b);


#ifdef SHOWLOADBALANCE
std::cout<<"load balance"<<std::endl;
for (int i=0; i< 244; ++i)
std::cout<<count[i]<<std::endl;
#endif

delete[] prin_;



return 0;
}



