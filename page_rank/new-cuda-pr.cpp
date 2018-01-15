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
#include "math.h"
#include "streamUtils.hpp"
#include "tbb/concurrent_queue.h"
#include "math.h"
#include "streamUtils.hpp"
#include "tbb/concurrent_queue.h"



	template <typename VertexType, typename EdgeType, typename Scalar>
int main_pr(VertexType nVtx, EdgeType* xadj_, VertexType *adj_, Scalar* val_, Scalar *prior_, Scalar* pr_,
		Scalar lambda,
		int nTry, //algo parameter
		util::timestamp& totaltime, std::string& 
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
                X = (int) nVtx/(nb_blocks) ;
                X = X / 32 ;
                subsize = (X+1) * 32;
        }



	list<Task<int,int,float> > *tasks = new  list<Task<int,int,float> >;

	tbb::concurrent_bounded_queue<stream_container<int,int,float>* >* streams = new tbb::concurrent_bounded_queue<stream_container<int,int,float>* >;

	//
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

//cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
//cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);

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

	// creat stream and tasks
	creat_stream<int, int, float>(nVtx, &alpha, &beta, d_val, d_xadj, d_adj, d_prin, d_prout, streams, stream_number );
	int nb_tasks =	split_input_to_tasks<int, int, float>(xadj, nVtx, subsize, *tasks);

	cout << "number-blocks: " << nb_tasks << endl;
	/*	int index = 0;
		while(index < nb_tasks){
		stream_container<int, int, float> *current_stream;
		Task <int,int,float> t = get_task<int,int,float>(tasks, index++);
	//streams->pop(current_stream);
	//put_work_on_stream<int,int,float>(current_stream,t);
	cout << "id : ------------------ "<< t.id << " ---------------------------------" << endl;
	cout << "       m : "<< t.nVtx << endl;
	cout << "       n : "<< t.nnz << endl;
	cout << "       nnz : "<< t.nnz << endl;
	cout << "       xdaj : "<< *t.xadj << endl;
	cout << "       prin: "<< *t.prin << endl;
	cout << "       prout : "<< *t.prout << endl;
	}
	 */


	for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
	{
		if (TRY >= THROW_AWAY)
			start = util::timestamp();

		for (int iter = 0; iter < 40 ; ++ iter) {

			int index = 0;

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

			while(index < nb_tasks){
				stream_container<int, int, float> *current_stream;
				Task <int,int,float> t = get_task<int,int,float>(tasks, index++);
				streams->pop(current_stream);
				put_work_on_stream<int,int,float>(current_stream,t);

	//			cout << "m : " << current_stream->m <<", n : "<< current_stream->n<< ", nnz " << current_stream->nnz << endl;
				cusparseSetStream(cusparseHandle, current_stream->stream);
				cusparseStatus = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						current_stream->m, current_stream->n, current_stream->nnz, current_stream->alpha,
						descr,
						current_stream->d_val, current_stream->d_xadj+current_stream->RowPtr, current_stream->d_adj,
						current_stream->d_prin, current_stream->beta,
						current_stream->d_prout+current_stream->RowPtr);

				cudaStreamAddCallback(current_stream->stream, call_back , current_stream , 0);
}
				cudaDeviceSynchronize();

				if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
					std::cerr<<"err"<<std::endl;



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

				//std::cerr<<eps<<std::endl;

			}

			checkCudaErrors(cudaMemcpy(prout, d_prout, nVtx*sizeof(*prout), cudaMemcpyDeviceToHost));

			std::cerr<<"PR[0]="<<prout[0]<<std::endl<<std::endl;

			if (TRY >= THROW_AWAY)
			{
				util::timestamp stop;  
				totaltime += stop - start;
			}

#ifndef LOG
			if (coldcache) {
#pragma omp parallel
				{
					evict_array_from_cache(prior, nVtx*sizeof(*prior));
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



