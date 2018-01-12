#include "main-pr.hpp"

#define THROW_AWAY 0
#include "Padded2DArray.hpp"
#include <omp.h>
#include "memutils.hpp"
#include <cmath>

//#define SHOWLOADBALANCE

//#define LOG

#ifdef LOG
#include "logged_array.hpp"
#endif

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "helper_cuda.h"
#include "LightSpMV_interface.hpp"

template <typename VertexType, typename EdgeType, typename Scalar>
int main_pr(VertexType nVtx, EdgeType* xadj_, VertexType *adj_, Scalar* val_, Scalar *prior_, Scalar* pr_,
	    Scalar lambda,
	      int nTry, //algo parameter
	      util::timestamp& totaltime, std::string& 
	      )
{
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
  //  Scalar *d_alpha;
  //Scalar *d_beta;

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

  lightSpMVCSRKernel lspmv;
  

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
	// cusparseStatus = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
	// 	       nVtx, nVtx, xadj[nVtx], &alpha,
	// 	       descr,
	// 	       d_val, d_xadj, d_adj,
	// 	       d_prin, &beta,
	// 	       d_prout);
	// cudaThreadSynchronize();


	lspmv.spmvBLAS(nVtx, nVtx, xadj[nVtx],
		       (uint32_t*)d_xadj, (uint32_t*)d_adj, d_val,
		       d_prin, d_prout,
		       alpha, beta, 0);
	cudaThreadSynchronize();




	//compute epsilon
	//using prin to compute epsilon
	float epsalpha = -1;
	cublasStatus = cublasSaxpy (cublasHandle, nVtx, &epsalpha, d_prout, 1, d_prin, 1); // d_prin = d_prout*-1 + d_prin

	if (cublasStatus != CUBLAS_STATUS_SUCCESS)
	  std::cerr<<"err"<<std::endl;

	cublasStatus = cublasSasum(cublasHandle, nVtx, d_prin, 1, &eps);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS)
	  std::cerr<<"err"<<std::endl;
	
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

    }

#ifdef SHOWLOADBALANCE
  std::cout<<"load balance"<<std::endl;
  for (int i=0; i< 244; ++i)
    std::cout<<count[i]<<std::endl;
#endif

  delete[] prin_;

  return 0;
}

  
  
