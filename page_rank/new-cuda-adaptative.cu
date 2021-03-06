#include "main-pr.hpp"

#define THROW_AWAY 1
#include "Padded2DArray.hpp"
#include <omp.h>
#include "memutils.hpp"
#include <cmath>

//#define SHOWLOADBALANCE
#include "logged_array.hpp"

//#define LOG
//#include <cuda_runtime.h>
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

	int nb_blocks = 0;
	int blk_size = 0;
	int short_blk_size = 0;
	int nb_threads = 0;
	int V = 0;

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
				std::cerr<<"BLKSIZE invalid"<<std::endl;
		}
	}

	{
		char* str = getenv ("SHORTBLKSIZE");
		if (str) {
			std::stringstream ss (str);
			ss>>short_blk_size;
			if (!ss)
				std::cerr<<"SHORTBLKSIZE invalid"<<std::endl;
		}
	}



	{
                char* str = getenv ("NBBLOCK");
                if (str) {
                        std::stringstream ss (str);
                        ss>>nb_blocks;
                        if (!ss)
                                std::cerr<<"NBBLOCK invalid"<<std::endl;
                }
        }



	if(nb_threads == 0 ){
		std::cerr<<" NBTHREAD=??? "<<std::endl;
		exit(0);
	}

	if(blk_size == 0 || short_blk_size == 0 ){
                std::cerr<<" BLKSIZE or SHORTBLKSIZE=??? "<<std::endl;
                exit(0);
        }
	if(nb_blocks == 0 ){
                std::cerr<<" NBBLOCK=??? "<<std::endl;
                exit(0);
        }

	//cudaPrintError(" cudaDeviceReset() -1 ");
	//cudaDeviceReset();
	//cudaPrintError(" cudaDeviceReset() -2 ");

	bool coldcache = true;

	util::timestamp start(0,0);
	util::timestamp start1(0,0);

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

	
	cudaSetDevice(0);
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
	const unsigned int shortBlkSize = short_blk_size;
	const unsigned int blkMultiplier = 3 ;
	const unsigned int rows_for_vector = 1 ;
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


//**************************************************
	int *method;
	int *d_method;
//	float *rowErr;

	method = (int*) calloc(sizeof(int), 5);
	checkCudaErrors( cudaMalloc((void**)&d_method, 5*sizeof(int)));

	checkCudaErrors( cudaMemcpy(d_method, method, 5*sizeof(int), cudaMemcpyHostToDevice) );
//*********************************************


//	util::timestamp start_2(0,0);
//	start_2 = util::timestamp(); 
	unsigned long* rowBlockstest;

//	cout <<" comput rowBlocks " << endl;
	//calculer rowBlockSize
	rowBlockSize1 = ComputeRowBlocksSize<int,int>(xadj, nVtx, blkSize, shortBlkSize , blkMultiplier, rows_for_vector, nThreadPerBlock);
	//declarer rowBlocks

	//cerr << " rowBlockSize1=" << rowBlockSize1;
	
	rowBlockSize2 = rowBlockSize1;


	rowBlocks = (unsigned long*) calloc(sizeof(unsigned long),rowBlockSize1);
//	rowBlockstest = (unsigned long*) calloc(sizeof(unsigned long),rowBlockSize1);

	//calculer rowBlocks
	ComputeRowBlocks<int,int>( rowBlocks, rowBlockSize2, xadj, nVtx, blkSize, shortBlkSize, blkMultiplier, rows_for_vector, nThreadPerBlock, allocate_row_blocks);

//	cout << "fin de calcule de rowBlocks" <<endl;



	//malloc for device variable
//	checkCudaErrors( cudaMalloc((void**)&rowErr, (rowBlockSize1+1)*sizeof(float)));
	checkCudaErrors( cudaMalloc((void**)&d_rowBlocks, ((rowBlockSize1)*sizeof(unsigned long))));
	checkCudaErrors( cudaMalloc((void**)&d_blkSize, 1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_rows_for_vector,1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_blkMultiplier, 1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_a, 1*sizeof(float)));
	checkCudaErrors( cudaMalloc((void**)&d_b, 1*sizeof(float)));


	//send data to device
	checkCudaErrors( cudaMemcpy(d_rowBlocks, rowBlocks, (rowBlockSize1)*sizeof(unsigned long), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_blkSize, &blkSize, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_rows_for_vector, &rows_for_vector, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_blkMultiplier, &blkMultiplier, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_a, &alpha, 1*sizeof(Scalar), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_b, &beta, 1*sizeof(Scalar), cudaMemcpyHostToDevice) );

	int mmshared_size =  (blkSize + 1) * sizeof(float);


	// prepar stream config

	list<Task> *tasks = new  list<Task>;

	tbb::concurrent_bounded_queue<stream_container<int,int,float>* >* streams = new tbb::concurrent_bounded_queue<stream_container<int,int,float>* >;

//	int nb_blocks = 128;
	int stream_number = 5;
	

	int X, subsize;
	X = (int) rowBlockSize1/(nb_blocks) ;


	if(X > 32 && nb_blocks != 1){
		if(X % 32 == 0){
			subsize = X;
		}else{  
			X = X / 32 ;
			subsize = (X+1) * 32;
		}
	}else{
		subsize=rowBlockSize1;
	}








	creat_stream<int, int, float>(d_rowBlocks, d_a, d_b, d_val, d_xadj, d_adj, d_prin, d_prout, d_blkSize, d_rows_for_vector, d_blkMultiplier, streams, stream_number );
	int nb_tasks = split_input_to_tasks(rowBlocks, rowBlockSize1, subsize, *tasks);
//	cout << "end creat stream " <<endl;
//
//	cout << "start split task " <<endl;
//	int cum=0;
//	int S=0;
//	unsigned int total_row=0;

	cout << "nb_blocks: " << nb_blocks << " subsize: " << subsize << " rowBlockSize: " << rowBlockSize1 << " ";
	cout << "nThreadPerBlock: " << nThreadPerBlock << " shortBlockSize: " << shortBlkSize << " BlkSize: " <<  blkSize << " ";
	cout << "nb_task: " << nb_tasks << " ";



int m1=0, m2=0, m3=0;

for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
{
	if (TRY >= THROW_AWAY){
	//	std::cout<<"start measuring"<<std::endl;
		start = util::timestamp();
}
	
	for (int iter = 0; iter < 40; ++iter){
	
		int index =0;

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
				Task t = get_task(tasks, index);
				streams->pop(current_stream);
				put_work_on_stream<int,int,float>(current_stream,t);
				//cudaPrintError("before kernel");


				csr_adaptative<<< current_stream->rowBlockSize , nThreadPerBlock, mmshared_size, current_stream->stream >>>(current_stream->d_val, current_stream->d_adj, current_stream->d_xadj, current_stream->d_prin, current_stream->d_prout, (current_stream->d_rowBlocks + current_stream->rowBlocksPtr), current_stream->alpha, current_stream->beta, current_stream->d_blkSize, current_stream->d_blkMultiplier, current_stream->d_rows_for_vector, current_stream->rowBlockSize, d_method);

					cudaPrintError("after kernel1");

				//	cudaPrintError("befor callback");
				cudaStreamAddCallback(current_stream->stream, call_back , current_stream , 0);
				//	cudaPrintError("after callback");


				index++;
		}	


		//		cudaPrintError("befor Synch");
				cudaDeviceSynchronize();
		//		cudaPrintError("after Synch");

		if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
			std::cerr<<"err-1"<<std::endl;



		//compute epsilon
		//using prin to compute epsilon
		float epsalpha = -1.;
	
		cublasStatus = cublasSaxpy (cublasHandle, nVtx, &epsalpha, d_prout, 1, d_prin, 1); // d_prin = d_prout*-1 + d_prin


		if (cublasStatus != CUBLAS_STATUS_SUCCESS)
			std::cerr<<"err-2"<<std::endl;

		start1 = util::timestamp();
		cublasStatus = cublasSasum(cublasHandle, nVtx, d_prin, 1, &eps);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS)
			std::cerr<<"err-3"<<std::endl;


//		util::timestamp stop1;  
//		cout << "cublasSum : totaltime = " << stop1 - start1 << endl;
		//	cudaPrintError("cublas");



		//stopping condition
		//	if (eps < 0) // deactivited for testing purposes
		//		iter = 20;

		std::cerr << eps << endl; 


	}
	checkCudaErrors(cudaMemcpy(prout, d_prout, 1*sizeof(*prout), cudaMemcpyDeviceToHost));

	//std::cerr<<"PR[0]="<<prout[0]<<std::endl;
	//	checkCudaErrors(cudaMemcpy(method, d_method, 5*sizeof(int), cudaMemcpyDeviceToHost));
	//	std::cerr << " method Stm="<< method[0] << " V ="<< method[1] << " VL="<< method[2] << " V3="<< method[3] << " V4=" <<method[5] <<endl ;

	for(int i=0; i<1; i++)
	{
		std::cerr.precision(10);
		std::cerr <<"PR["<< i <<"]="<<prout[i]<<std::endl;
	}
	if (TRY >= THROW_AWAY)
	{
		util::timestamp stop;  
		totaltime += stop - start;
//		cout << "cuda-adap : totaltime = " << stop - start << endl;
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

free(rowBlocks);
free(method);
free(rowBlockstest);

#ifdef SHOWLOADBALANCE
std::cout<<"load balance"<<std::endl;
for (int i=0; i< 244; ++i)
std::cout<<count[i]<<std::endl;
#endif

//delete[] prin_;



return 0;
}



