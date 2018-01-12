#include "main-pr.hpp"

#define THROW_AWAY 3
#include "Padded2DArray.hpp"
#include <omp.h>
#include "memutils.hpp"
#include <cmath>

//#define SHOWLOADBALANCE
#include "logged_array.hpp"

//#define LOG

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "helper_cuda.h"


template <typename VertexType, typename EdgeType, typename Scalar>
int main_pr(VertexType nVtx, EdgeType* xadj_, VertexType *adj_, Scalar* val_, Scalar *prior_, Scalar* pr_,
	    Scalar lambda,
	      int nTry, //algo parameter
	      util::timestamp& totaltime, std::string& out
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


  VertexType lastinzero = 0;
  //using miguet pierson load balancing algorithm. cut at WORK/P
  //TODO:should probably bin search it
  for (lastinzero = 0; lastinzero != nVtx; ++lastinzero) {
    if (xadj[lastinzero] > xadj[nVtx]/2)
      break;
  }

  std::cerr<<"last in gpu 0: "<<lastinzero<<" edges: "<<xadj[lastinzero]<<std::endl;

  std::cout << "nVtx =" << nVtx << std::endl;
  std::cout << "lastinzero =" << lastinzero << std::endl;
  std::cout << "xadj[nVtx] =" << xadj[nVtx] << std::endl;
  std::cout << "GPU 0 "<< 1+lastinzero <<" , "<<  xadj[1+lastinzero+1]  << std::endl;
  std::cout << "GPU 1 "<< nVtx-1-lastinzero <<" , "<< xadj[nVtx]-xadj[lastinzero+1+1]  << std::endl;
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
  

  for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
    {
      if (TRY >= THROW_AWAY)
	start = util::timestamp();

      int maxiter = 40;


	//for GPU0

	cudaSetDevice(0);
	//setup prin
	//cudaMemcpyAsync(d_prin0, d_prior0, nVtx*sizeof(*prior), cudaMemcpyDeviceToDevice,stream0);
	cudaMemcpyAsync(d_prin0, d_prior0, (1+lastinzero)*sizeof(*prior), cudaMemcpyDeviceToDevice,stream0);


	cudaSetDevice(1);
	//setup prin
	cudaMemcpyAsync(d_prin1+lastinzero+1, d_prior1+lastinzero+1, (nVtx-lastinzero-1)*sizeof(*prior), cudaMemcpyDeviceToDevice,stream1);


	cudaSetDevice(1);
	checkCudaErrors( cudaStreamSynchronize(stream1));
	

	cudaSetDevice(0);
	checkCudaErrors( cudaStreamSynchronize(stream0));


      for (int iter = 0; iter < maxiter ; ++ iter) {

	//exchange data	
	cudaSetDevice(1);
	cudaMemcpyAsync(d_prin1, d_prin0, (1+lastinzero)*sizeof(*d_prin0), cudaMemcpyDeviceToDevice, stream1);//probably incorrect

	cudaSetDevice(0);
	cudaMemcpyAsync(d_prin0+1+lastinzero, d_prin1+1+lastinzero, (nVtx-lastinzero-1)*sizeof(*d_prin0), cudaMemcpyDeviceToDevice, stream0);//probably incorrect


	cudaSetDevice(0);
	//prout = A prin
	//prout = lambda * prout + (1-lambda) prior

	cudaMemcpyAsync(d_prout0, d_prior0, (1+lastinzero)*sizeof(*prior), cudaMemcpyDeviceToDevice, stream0);



	//for float it is S.
	//does prout = alpha A prin + beta prout
	cusparseSetStream(cusparseHandle0, stream0);

	// cusparseStatus0 = cusparseScsrmv(cusparseHandle0, CUSPARSE_OPERATION_NON_TRANSPOSE,
	// 	       nVtx, nVtx, xadj[nVtx], &alpha,
	// 	       descr0,
	// 	       d_val0, d_xadj0, d_adj0,
	// 	       d_prin0, &beta,
	// 	       d_prout0);


	cusparseStatus0 = cusparseScsrmv(cusparseHandle0, CUSPARSE_OPERATION_NON_TRANSPOSE,
		       1+lastinzero, nVtx, xadj[1+lastinzero+1], &alpha,
		       descr0,
		       d_val0, d_xadj0, d_adj0,
		       d_prin0, &beta,
		       d_prout0);





	//compute epsilon
	//using prin to compute epsilon

	cublasSetStream(cublasHandle0, stream0);
	
	cublasSaxpy (cublasHandle0, 1+lastinzero, d_epsalpha0, d_prout0, 1, d_prin0, 1); // d_prin = d_prout*-1 + d_prin

	cublasSasum (cublasHandle0, 1+lastinzero, d_prin0, 1, d_eps0);

	cudaMemcpyAsync(h_eps0, d_eps0, sizeof(*d_eps0), cudaMemcpyDeviceToHost, stream0);

	//	cudaMemcpyAsync(d_prin0, d_prout0, nVtx*sizeof(*prout), cudaMemcpyDeviceToDevice, stream0);//prepare prin for next iteration
	cudaMemcpyAsync(d_prin0, d_prout0, (1+lastinzero)*sizeof(*prout), cudaMemcpyDeviceToDevice, stream0);//prepare prin for next iteration


	//for GPU1

	cudaSetDevice(1);

	
	//prout = A prin
	//prout = lambda * prout + (1-lambda) prior
	
	cudaMemcpyAsync(d_prout1+1+lastinzero, d_prior1+1+lastinzero, (nVtx-1-lastinzero)*sizeof(*prior), cudaMemcpyDeviceToDevice, stream1);
	
	
	//for float it is S.
	//does prout = alpha A prin + beta prout
	cusparseSetStream(cusparseHandle1, stream1);
	
	// cusparseScsrmv(cusparseHandle1, CUSPARSE_OPERATION_NON_TRANSPOSE,
	// 	       nVtx, nVtx, xadj[nVtx], &alpha1,
	// 	       descr1,
	// 	       d_val1, d_xadj1, d_adj1,
	// 	       d_prin1, &beta1,
	// 	       d_prout1);
	

	cusparseScsrmv(cusparseHandle1, CUSPARSE_OPERATION_NON_TRANSPOSE,
		       nVtx-1-lastinzero, nVtx, xadj[nVtx]-xadj[lastinzero+1+1], &alpha1,
		       descr1,
		       d_val1, d_xadj1+1+lastinzero, d_adj1,
		       d_prin1, &beta1,
		       d_prout1+1+lastinzero);

	
	//compute epsilon
	//using prin to compute epsilon
	
	cublasSetStream(cublasHandle1, stream1);
	
	
	cublasSaxpy (cublasHandle1, (nVtx-1-lastinzero), d_epsalpha1, d_prout1+1+lastinzero, 1, d_prin1+1+lastinzero, 1); // d_prin = d_prout*-1 + d_prin
	
	cublasSasum(cublasHandle1, nVtx-1-lastinzero, d_prin1+1+lastinzero, 1, d_eps1);

	cudaMemcpyAsync(h_eps1, d_eps1, sizeof(*h_eps1), cudaMemcpyDeviceToHost, stream1);

	cudaMemcpyAsync(d_prin1+1+lastinzero, d_prout1+1+lastinzero, (nVtx-lastinzero-1)*sizeof(*prout), cudaMemcpyDeviceToDevice,stream1);//prepare prin for next iteration

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
    ss<<"part1V: "<<lastinzero+1<<" part1E: "<<xadj[lastinzero+2]
      <<" part2V: "<<nVtx-(lastinzero+1)<<" part2E: "<< xadj[nVtx] - xadj[lastinzero+2];
    out = ss.str();
  }



  return 0;
}

  
  
