#include "main-pr.hpp"

#define THROW_AWAY 0 
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
#include "LightSpMV_interface.hpp"
#include "streamUtils.hpp"
#include "tbb/concurrent_queue.h"
#include "math.h"
#include "streamUtils.hpp"
#include "tbb/concurrent_queue.h"



	template <typename VertexType, typename EdgeType, typename Scalar>
int main_pr(VertexType nVtx, EdgeType* xadj_, VertexType *adj_, Scalar* val_, Scalar *prior_, Scalar* pr_,
		Scalar lambda,
		int nTry, //algo parameter
		util::timestamp& totaltime, std::string& out
	   )
{

	int subsize = 0;
	int nb_blocks = 0;
	int stream_number = 0;

	{
		char* str = getenv ("NBBLOCK");
		if (str) {
			std::stringstream ss (str);
			ss>>nb_blocks;
			if (!ss)
				std::cerr<<"NBBLOCK invalid"<<std::endl;
		}
	}



	{
		char* str = getenv ("SUBSIZE");
		if (str) {
			std::stringstream ss (str);
			ss>>subsize;
			if (!ss)
				std::cerr<<"SUBSIZE invalid"<<std::endl;
		}
	}





	{
		char* str = getenv ("NBSTREAM");
		if (str) {
			std::stringstream ss (str);
			ss>>stream_number;
			if (!ss)
				std::cerr<<"NBSTREAM invalid"<<std::endl;
		}
	}

	if(nb_blocks == 0 && subsize == 0){
		std::cerr<<"SUBSIZE=??? or  NBBLOCK=???"<<std::endl;
		exit(0);
	}

	if(stream_number == 0){
		std::cerr<<"NBSTREAM=???? "<<std::endl;
		exit(0);
	}



	if(subsize == 0){
		int X;
		X = (int) nVtx/(nb_blocks*2) ;
		X = X / 32 ;
		subsize = (X+1) * 32;
	}

		std::cout << "subsize " << subsize <<std::endl;
		list<Task<int,int,float> > *tasks = new  list<Task<int,int,float> >;

		tbb::concurrent_bounded_queue<stream_container<int,int,float>* >* streams = new tbb::concurrent_bounded_queue<stream_container<int,int,float>* >;

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

		checkCudaErrors(cudaSetDevice(1));
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
//i		vector<lightSpMVCSRKernel> lspmv1(stream_number)  ;

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

		streams->set_capacity(stream_number*3);
		creat_stream_2gpus<int, int, float>(nVtx, &alpha, &beta, d_val0, d_xadj0, d_adj0, d_prin0, d_prout0, d_val1, d_xadj1, d_adj1, d_prin1, d_prout1, &cusparseHandle0, &cusparseHandle1, &descr0, &descr1, streams, stream_number );


		int nb_tasks =	split_input_to_tasks<int, int, float>(xadj, nVtx, subsize, *tasks);

		//	std::cout << " number of tasks " << nb_tasks << std::endl;


		//	int nb_tasks =	smart_split_input_to_tasks<int, int, float>(xadj, nVtx, subsize, *tasks);

		std::cout << "number-blocks: " << nb_tasks << std::endl;

		int medium;
		for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
		{
                cudaSetDevice(0);
		vector<lightSpMVCSRKernel> lspmv0(stream_number)  ;
                cudaSetDevice(1);
		vector<lightSpMVCSRKernel> lspmv1(stream_number)  ;

			if (TRY >= THROW_AWAY)
				start = util::timestamp();

			int maxiter = 40;

			nVtx % 2 == 0 ? medium = (nVtx/2) : medium = ((nb_tasks - 1)/2);

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

				int top = 0;
				int bottom = nb_tasks;
				//exchange data	
				cudaSetDevice(1);
				cudaMemcpyAsync(d_prin1, d_prin0, (medium)*sizeof(*d_prin0), cudaMemcpyDeviceToDevice, stream1);

				cudaSetDevice(0);
				cudaMemcpyAsync(d_prin0+medium, d_prin1+medium, (nVtx-medium)*sizeof(*d_prin0), cudaMemcpyDeviceToDevice, stream0); 

				//prout = A prin
				//prout = lambda * prout + (1-lambda) prior
				cudaSetDevice(0);
				cudaMemcpyAsync(d_prout0, d_prior0, (medium)*sizeof(*prior), cudaMemcpyDeviceToDevice, stream0);

				cudaSetDevice(1);
				cudaMemcpyAsync(d_prout1+medium, d_prior1+medium, (nVtx-medium)*sizeof(*prior), cudaMemcpyDeviceToDevice, stream1);

				cudaSetDevice(1);
				checkCudaErrors( cudaStreamSynchronize(stream1));


				cudaSetDevice(0);
				checkCudaErrors( cudaStreamSynchronize(stream0));
			
				int iteration = 0;
				while(top < bottom){
					iteration++;
					//std::cout << " while : "<<  std::endl;
					stream_container<int, int, float> *current_stream;
					streams->pop(current_stream);
					if(current_stream->device == 0){
				//		std::cout << "0 top++ : " << top <<std::endl;
						Task <int,int,float> t = get_task<int,int,float>(tasks, top++);
						put_work_on_stream<int,int,float>(current_stream,t);
						cudaSetDevice(0);					
						lspmv0[current_stream->id].spmvBLAS(current_stream->m, current_stream->n, current_stream->nnz,
								(uint32_t *)(current_stream->d_xadj+current_stream->RowPtr)
								,(uint32_t *)current_stream->d_adj, current_stream->d_val,
								current_stream->d_prin,
								current_stream->d_prout+current_stream->RowPtr,
								*current_stream->alpha, *current_stream->beta,current_stream->stream);

						checkCudaErrors(cudaStreamAddCallback(current_stream->stream, call_back , current_stream , 0));
					}else{
						//	std::cout << "1 bottom-- " << bottom << std::endl;
						Task <int,int,float> t = get_task<int,int,float>(tasks, --bottom);
						put_work_on_stream<int,int,float>(current_stream,t);
						
						cudaSetDevice(1);					
						lspmv1[current_stream->id].spmvBLAS(current_stream->m, current_stream->n, current_stream->nnz,
								(uint32_t *)(current_stream->d_xadj+current_stream->RowPtr)
								,(uint32_t *)current_stream->d_adj, current_stream->d_val,
								current_stream->d_prin,
								current_stream->d_prout+current_stream->RowPtr,
								*current_stream->alpha, *current_stream->beta,current_stream->stream);

						checkCudaErrors(cudaStreamAddCallback(current_stream->stream, call_back , current_stream , 0));

						medium = current_stream->RowPtr;

					}		
					//for float it is S.
					//does prout = alpha A prin + beta prout
					//std::cout << iteration << " ---  GPU " << current_stream->device << " stream "<<current_stream->id<<" prend nVtx " << current_stream->m  << " and NNZ  "  << current_stream->nnz << std::endl;
				}

				cudaSetDevice(0);
				cudaDeviceSynchronize();
				cudaSetDevice(1);
				cudaDeviceSynchronize();			

				//      std::cout << " medium : "<< medium << std::endl;

				//compute epsilon
				//using prin to compute epsilon
				cudaSetDevice(0);
				cublasSetStream(cublasHandle0, stream0);

				cublasSaxpy (cublasHandle0, medium, d_epsalpha0, d_prout0, 1, d_prin0, 1); // d_prin = d_prout*-1 + d_prin

				cublasSasum (cublasHandle0, medium, d_prin0, 1, d_eps0);

				cudaMemcpyAsync(h_eps0, d_eps0, sizeof(*d_eps0), cudaMemcpyDeviceToHost, stream0);

				//	cudaMemcpyAsync(d_prin0, d_prout0, nVtx*sizeof(*prout), cudaMemcpyDeviceToDevice, stream0);//prepare prin for next iteration


				//compute epsilon
				//using prin to compute epsilon
				cudaSetDevice(1);
				cublasSetStream(cublasHandle1, stream1);
				cublasSaxpy (cublasHandle1, (nVtx-medium), d_epsalpha1, d_prout1+medium, 1, d_prin1+medium, 1); // d_prin = d_prout*-1 + d_prin

				cublasSasum(cublasHandle1, nVtx-medium, d_prin1+medium, 1, d_eps1);

				cudaMemcpyAsync(h_eps1, d_eps1, sizeof(*h_eps1), cudaMemcpyDeviceToHost, stream1);
				cudaSetDevice(1);
				cudaMemcpyAsync(d_prin1+medium, d_prout1+medium, (nVtx-medium)*sizeof(*prout), cudaMemcpyDeviceToDevice,stream1);//prepare prin for next iteration

				cudaSetDevice(0);
				cudaMemcpyAsync(d_prin0, d_prout0, (medium)*sizeof(*prout), cudaMemcpyDeviceToDevice, stream0);//prepare prin for next iteration

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



		cudaSetDevice(0);
		cudaDeviceReset();
		cudaSetDevice(1);
		cudaDeviceReset();



		delete[] prin_;

		int lastinzero  = 1; 
		{
			std::stringstream ss;
			ss<<"part1V: "<< medium <<" part1E: "<<xadj[medium]
				<<" part2V: "<< nVtx-(medium) <<" part2E: "<< xadj[nVtx] - xadj[medium];
			out = ss.str();
		}


		return 0;
	}



