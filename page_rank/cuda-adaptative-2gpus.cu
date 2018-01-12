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
//#include "streamUtils.hpp"
#include "tbb/concurrent_queue.h"
#include "math.h"
#include "AdaptativeUtils.hpp"

	template <typename VertexType, typename EdgeType, typename Scalar>
int main_pr(VertexType nVtx, EdgeType* xadj_, VertexType *adj_, Scalar* val_, Scalar *prior_, Scalar* pr_,
		Scalar lambda,
		int nTry, //algo parameter
		util::timestamp& totaltime, std::string& out 
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

	Scalar alpha1 = lambda;
	Scalar beta1 = 1-lambda;
	Scalar epsalpha = -1;

	Scalar *h_eps0;
	Scalar *h_eps1;


	//cuda side variable
	EdgeType* d_xadj0 ;
	VertexType *d_adj0 ;
	Scalar* d_val0 ;
	Scalar* d_prior0 ;
	Scalar* d_prin0 ;
	Scalar* d_prout0 ;
	Scalar *d_alpha0;
	Scalar *d_beta0;
	Scalar *d_epsalpha0;
	Scalar *d_eps0;

	EdgeType* d_xadj1 ;
	VertexType *d_adj1 ;
	Scalar* d_val1 ;
	Scalar* d_prior1 ;
	Scalar* d_prin1 ;
	Scalar* d_prout1 ;
	Scalar *d_alpha1;
	Scalar *d_beta1;

	Scalar *d_epsalpha1;
	Scalar *d_eps1;

	/* Get handle to the CUBLAS context */
	cudaSetDevice(0);
	cublasHandle_t cublasHandle0 = 0;
	cublasStatus_t cublasStatus0;
	cublasStatus0 = cublasCreate(&cublasHandle0);
	cublasSetPointerMode(cublasHandle0, CUBLAS_POINTER_MODE_DEVICE);

	checkCudaErrors( cudaSetDevice(1));
	cublasHandle_t cublasHandle1 = 0;
	cublasStatus_t cublasStatus1;
	cublasStatus1 = cublasCreate(&cublasHandle1);
	cublasSetPointerMode(cublasHandle1, CUBLAS_POINTER_MODE_DEVICE);

	/* Get handle to the CUSPARSE context */
	cudaSetDevice(0);

	cusparseHandle_t cusparseHandle0 = 0;
	cusparseStatus_t cusparseStatus0;
	cusparseStatus0 = cusparseCreate(&cusparseHandle0);

	cusparseMatDescr_t descr0 = 0;
	cusparseStatus0 = cusparseCreateMatDescr(&descr0);

	cusparseSetMatType(descr0,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr0,CUSPARSE_INDEX_BASE_ZERO);


	cudaSetDevice(1);

	cusparseHandle_t cusparseHandle1 = 0;
	cusparseStatus_t cusparseStatus1;
	cusparseStatus1 = cusparseCreate(&cusparseHandle1);

	cusparseMatDescr_t descr1 = 0;
	cusparseStatus1 = cusparseCreateMatDescr(&descr1);

	cusparseSetMatType(descr1,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr1,CUSPARSE_INDEX_BASE_ZERO);


	//cuda stream

	cudaSetDevice(0);

	cudaStream_t stream0;
	cudaStreamCreate(&stream0);

	cudaSetDevice(1);

	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	//memalloc

	cudaSetDevice(0);

	checkCudaErrors( cudaMalloc((void**)&d_xadj0, (nVtx+1)*sizeof(*xadj)) );
	checkCudaErrors( cudaMalloc((void**)&d_adj0, (xadj[nVtx])*sizeof(*adj)) );
	checkCudaErrors( cudaMalloc((void**)&d_val0, (xadj[nVtx])*sizeof(*val)) );
	checkCudaErrors( cudaMalloc((void**)&d_prior0, (nVtx*sizeof(*prior))));
	checkCudaErrors( cudaMalloc((void**)&d_prin0, (nVtx*sizeof(*prin)) ));
	checkCudaErrors( cudaMalloc((void**)&d_prout0, (nVtx*sizeof(*prout)) ));
	checkCudaErrors( cudaMalloc((void**)&d_epsalpha0, (sizeof(epsalpha)) ));
	checkCudaErrors( cudaMallocHost((void**)&h_eps0, (sizeof(*h_eps0)) ));
	checkCudaErrors( cudaMalloc((void**)&d_eps0, (sizeof(*h_eps0)) ));


	cudaSetDevice(1);

	checkCudaErrors( cudaMalloc((void**)&d_xadj1, (nVtx+1)*sizeof(*xadj)) );
	checkCudaErrors( cudaMalloc((void**)&d_adj1, (xadj[nVtx])*sizeof(*adj)) );
	checkCudaErrors( cudaMalloc((void**)&d_val1, (xadj[nVtx])*sizeof(*val)) );
	checkCudaErrors( cudaMalloc((void**)&d_prior1, (nVtx*sizeof(*prior))));
	checkCudaErrors( cudaMalloc((void**)&d_prin1, (nVtx*sizeof(*prin)) ));
	checkCudaErrors( cudaMalloc((void**)&d_prout1, (nVtx*sizeof(*prout)) ));
	checkCudaErrors( cudaMalloc((void**)&d_epsalpha1, (sizeof(epsalpha)) ));
	checkCudaErrors( cudaMallocHost((void**)&h_eps1, (sizeof(*h_eps1)) ));
	checkCudaErrors( cudaMalloc((void**)&d_eps1, (sizeof(*h_eps1)) ));


	//cpu to gpu copies

	cudaSetDevice(0);


	checkCudaErrors( cudaMemcpy(d_xadj0, xadj, (nVtx+1)*sizeof(*xadj), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_adj0, adj, (xadj[nVtx])*sizeof(*adj), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_val0, val, (xadj[nVtx])*sizeof(*val), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_prior0, prior, nVtx*sizeof(*prior), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_epsalpha0, &epsalpha, sizeof(epsalpha), cudaMemcpyHostToDevice) );


	cudaSetDevice(1);


	checkCudaErrors( cudaMemcpy(d_xadj1, xadj, (nVtx+1)*sizeof(*xadj), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_adj1, adj, (xadj[nVtx])*sizeof(*adj), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_val1, val, (xadj[nVtx])*sizeof(*val), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_prior1, prior, nVtx*sizeof(*prior), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_epsalpha1, &epsalpha, sizeof(epsalpha), cudaMemcpyHostToDevice) );




	int nRows = nVtx;
	unsigned long* rowBlocks;
	const int nThreadPerBlock = nb_threads; 
	const unsigned int blkSize = blk_size; 
	const unsigned int blkMultiplier = 3;
	const unsigned int rows_for_vector = 1; 
	const bool allocate_row_blocks = true;

	//device 0 variable 
	unsigned long* d_rowBlocks0;
	unsigned int* d_blkSize0;
	unsigned int* d_rows_for_vector0;
	unsigned int* d_blkMultiplier0;
	float* d_a0;
	float* d_b0;

	//device 1 variable 
	unsigned long* d_rowBlocks1;
	unsigned int* d_blkSize1;
	unsigned int* d_rows_for_vector1;
	unsigned int* d_blkMultiplier1;
	float* d_a1;
	float* d_b1;

	int rowBlockSize1;
	int rowBlockSize2;


	//calculer rowBlockSize
	rowBlockSize1 = ComputeRowBlocksSize<int,int>(xadj, nVtx, blkSize, blkMultiplier, rows_for_vector, nThreadPerBlock);
	//cout << "rowBlockSize1 : " << rowBlockSize1 << endl;

	//declarer rowBlocks
	rowBlocks = (unsigned long*) calloc(sizeof(unsigned long),rowBlockSize1);

	//calculer rowBlocks
	ComputeRowBlocks<int,int>( rowBlocks, rowBlockSize2, xadj, nVtx, blkSize, blkMultiplier, rows_for_vector, nThreadPerBlock, allocate_row_blocks);
	//cout << "rowBlockSize2 : " << rowBlockSize2 <<endl;

        int end = ((rowBlocks[rowBlockSize1] >> (64-32)) & ((1UL << 32) - 1UL));
//	cout << " end : " << end <<endl;
//		if(end == 0){
//			rowBlockSize1--;
//		}

	int mediumRowblocks = cutRowBlocks(rowBlocks, rowBlockSize1);
	int part2 = rowBlockSize1 - mediumRowblocks;
	
	int medium =  ((rowBlocks[mediumRowblocks] >> (64-32)) & ((1UL << 32) - 1UL));
	end = ((rowBlocks[rowBlockSize1] >> (64-32)) & ((1UL << 32) - 1UL));
	
//	cout << "rowBlockSize1 : " << rowBlockSize1 << endl;
//	cout << "mediumRowBlocks :" << mediumRowblocks << endl;
//	cout << " - medium : " << medium <<endl;
//	cout << " - part2 = " << part2 << endl;
//	cout << " - end : -- > " << end << endl;

	//malloc for device 0 variable
	cudaSetDevice(0);
	checkCudaErrors( cudaMalloc((void**)&d_rowBlocks0, (rowBlockSize1*sizeof(unsigned long))));
	checkCudaErrors( cudaMalloc((void**)&d_blkSize0, 1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_rows_for_vector0,1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_blkMultiplier0, 1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_a0, 1*sizeof(float)));
	checkCudaErrors( cudaMalloc((void**)&d_b0, 1*sizeof(float)));

	//malloc for device 1 variable
	cudaSetDevice(1);
	checkCudaErrors( cudaMalloc((void**)&d_rowBlocks1, (rowBlockSize1*sizeof(unsigned long))));
	checkCudaErrors( cudaMalloc((void**)&d_blkSize1, 1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_rows_for_vector1,1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_blkMultiplier1, 1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_a1, 1*sizeof(float)));
	checkCudaErrors( cudaMalloc((void**)&d_b1, 1*sizeof(float)));




	//send data to device 0 
	cudaSetDevice(0);
	checkCudaErrors( cudaMemcpy(d_rowBlocks0, rowBlocks, rowBlockSize1*sizeof(unsigned long), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_blkSize0, &blkSize, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_rows_for_vector0, &rows_for_vector, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_blkMultiplier0, &blkMultiplier, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_a0, &alpha, 1*sizeof(Scalar), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_b0, &beta, 1*sizeof(Scalar), cudaMemcpyHostToDevice) );

	//send data to device 1 
	cudaSetDevice(1);
	checkCudaErrors( cudaMemcpy(d_rowBlocks1, rowBlocks, rowBlockSize1*sizeof(unsigned long), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_blkSize1, &blkSize, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_rows_for_vector1, &rows_for_vector, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_blkMultiplier1, &blkMultiplier, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_a1, &alpha, 1*sizeof(Scalar), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_b1, &beta, 1*sizeof(Scalar), cudaMemcpyHostToDevice) );


	int size =  (blkSize) * sizeof(float);

//		csr_adaptative<<<(rowBlockSize1 + 1) , nThreadPerBlock, size >>>(d_val, d_adj, d_xadj, d_prior, d_prout1, d_rowBlocks, d_a,  d_b, d_blkSize, d_blkMultiplier, d_rows_for_vector, rowBlockSize1);


/*	cudaSetDevice(0);
	csr_adaptative<<<(mediumRowblocks + 1) , nThreadPerBlock, size, stream0 >>>(d_val0, d_adj0, d_xadj0, d_prior0, d_prout0, d_rowBlocks0, d_a0,  d_b0, d_blkSize0, d_blkMultiplier0, d_rows_for_vector0, mediumRowblocks);

	cudaSetDevice(1);
	csr_adaptative<<<(part2 + 1) , nThreadPerBlock, size, stream1 >>>(d_val1, d_adj1, d_xadj1, d_prior1, d_prout1, (d_rowBlocks1 + mediumRowblocks) , d_a1,  d_b1, d_blkSize1, d_blkMultiplier1, d_rows_for_vector1, part2);


        cudaSetDevice(0);
	checkCudaErrors(cudaMemcpy(prout, d_prout0, nVtx*sizeof(*prout), cudaMemcpyDeviceToHost));

        for(int i=0; i< nVtx; i++)
                cout << "pr0["<< i <<"] = "<<prout[i]<<endl;
	cout << "   deuxiem ------ "<< endl  ;     
 
	cudaSetDevice(1);
	checkCudaErrors(cudaMemcpy(prout, d_prout1, nVtx*sizeof(*prout), cudaMemcpyDeviceToHost));


	for(int i=0; i< nVtx; i++)
		cout << "pr1["<< i <<"] = "<<prout[i]<<endl;
*/ 	

	for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
	{
		if (TRY >= THROW_AWAY)
			start = util::timestamp();

		int maxiter = 40;


		//for GPU0

		cudaSetDevice(0);
		//setup prin
		//cudaMemcpyAsync(d_prin0, d_prior0, nVtx*sizeof(*prior), cudaMemcpyDeviceToDevice,stream0);
		cudaMemcpyAsync(d_prin0, d_prior0, (medium)*sizeof(*prior), cudaMemcpyDeviceToDevice,stream0);


		cudaSetDevice(1);
		//setup prin
		cudaMemcpyAsync(d_prin1+medium, d_prior1+medium, (nVtx-medium)*sizeof(*prior), cudaMemcpyDeviceToDevice,stream1);


		cudaSetDevice(1);
		checkCudaErrors( cudaStreamSynchronize(stream1));


		cudaSetDevice(0);
		checkCudaErrors( cudaStreamSynchronize(stream0));


		for (int iter = 0; iter < maxiter ; ++ iter) {

			//exchange data	
			cudaSetDevice(1);
			cudaMemcpyAsync(d_prin1, d_prin0, (medium)*sizeof(*d_prin0), cudaMemcpyDeviceToDevice, stream1);//probably incorrect

			cudaSetDevice(0);
			cudaMemcpyAsync(d_prin0+medium, d_prin1+medium, (nVtx-medium)*sizeof(*d_prin0), cudaMemcpyDeviceToDevice, stream0);//probably incorrect


			cudaSetDevice(0);
			//prout = A prin
			//prout = lambda * prout + (1-lambda) prior

			cudaMemcpyAsync(d_prout0, d_prior0, (medium)*sizeof(*prior), cudaMemcpyDeviceToDevice, stream0);



/*		 cusparseStatus0 = cusparseScsrmv(cusparseHandle0, CUSPARSE_OPERATION_NON_TRANSPOSE,
					1+lastinzero, nVtx, xadj[1+lastinzero+1], &alpha,
					descr0,
					d_val0, d_xadj0, d_adj0,
					d_prin0, &beta,
					d_prout0);
*/

		csr_adaptative<<<(mediumRowblocks + 1) , nThreadPerBlock, size, stream0 >>>(d_val0, d_adj0, d_xadj0, d_prin0, d_prout0, d_rowBlocks0, d_a0,  d_b0, d_blkSize0, d_blkMultiplier0, d_rows_for_vector0, mediumRowblocks);


			//compute epsilon
			//using prin to compute epsilon

			cublasSetStream(cublasHandle0, stream0);

			cublasSaxpy (cublasHandle0, medium, d_epsalpha0, d_prout0, 1, d_prin0, 1); // d_prin = d_prout*-1 + d_prin

			cublasSasum (cublasHandle0, medium, d_prin0, 1, d_eps0);

			cudaMemcpyAsync(h_eps0, d_eps0, sizeof(*d_eps0), cudaMemcpyDeviceToHost, stream0);

			//	cudaMemcpyAsync(d_prin0, d_prout0, nVtx*sizeof(*prout), cudaMemcpyDeviceToDevice, stream0);//prepare prin for next iteration
			cudaMemcpyAsync(d_prin0, d_prout0, (medium)*sizeof(*prout), cudaMemcpyDeviceToDevice, stream0);//prepare prin for next iteration


			//for GPU1

			cudaSetDevice(1);


			//prout = A prin
			//prout = lambda * prout + (1-lambda) prior

			cudaMemcpyAsync(d_prout1+medium, d_prior1+medium, (nVtx-medium)*sizeof(*prior), cudaMemcpyDeviceToDevice, stream1);


/*			cusparseScsrmv(cusparseHandle1, CUSPARSE_OPERATION_NON_TRANSPOSE,
					nVtx-1-lastinzero, nVtx, xadj[nVtx]-xadj[lastinzero+1+1], &alpha1,
					descr1,
					d_val1, d_xadj1+1+lastinzero, d_adj1,
					d_prin1, &beta1,
					d_prout1+1+lastinzero);
*/
			
			csr_adaptative<<<(part2 + 1) , nThreadPerBlock, size, stream1 >>>(d_val1, d_adj1, d_xadj1, d_prin1, d_prout1, (d_rowBlocks1 + mediumRowblocks) , d_a1,  d_b1, d_blkSize1, d_blkMultiplier1, d_rows_for_vector1, part2);


			//using prin to compute epsilon

			cublasSetStream(cublasHandle1, stream1);


			cublasSaxpy (cublasHandle1, (nVtx-medium), d_epsalpha1, d_prout1+medium, 1, d_prin1+medium, 1); // d_prin = d_prout*-1 + d_prin

			cublasSasum(cublasHandle1, nVtx-medium, d_prin1+medium, 1, d_eps1);

			cudaMemcpyAsync(h_eps1, d_eps1, sizeof(*h_eps1), cudaMemcpyDeviceToHost, stream1);

			cudaMemcpyAsync(d_prin1+medium, d_prout1+medium, (nVtx-medium)*sizeof(*prout), cudaMemcpyDeviceToDevice,stream1);//prepare prin for next iteration

			cudaSetDevice(1);
			checkCudaErrors( cudaStreamSynchronize(stream1));


			cudaSetDevice(0);
			checkCudaErrors( cudaStreamSynchronize(stream0));


			//stopping condition
			if (*h_eps0 +*h_eps1 < 0) // deactivited for testing purposes
				iter = maxiter;

			std::cerr<<*h_eps0+*h_eps1<<std::endl;

		}

		cudaSetDevice(0);

		checkCudaErrors(cudaMemcpy(prout, d_prout0, nVtx*sizeof(*prout), cudaMemcpyDeviceToHost));

		std::cerr<<"PR[0]="<<prout[0]<<std::endl;

		if (TRY >= THROW_AWAY)
		{
			util::timestamp stop;  
			totaltime += stop - start;
		}

	}


	cudaStreamDestroy(stream0);

	delete[] prin_;


	{
		std::stringstream ss;
		ss<<"part1V: "<< medium <<" part1E: "<<xadj[medium+1]
			<<" part2V: "<<nVtx-(medium)<<" part2E: "<< xadj[nVtx] - xadj[medium+1];
		out = ss.str();
	}




	return 0;
}



