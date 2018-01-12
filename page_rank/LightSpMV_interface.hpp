/*
 * PageRankLightSpMV.cu
 *
 *  Created on: May 29, 2015
 *      Author: Yongchao Liu
 *      Affiliation: School of Computational Science & Engineering, Georgia Institute of Technology
 *      Email: yliu@cc.gatech.edu
 */

#include "LightSpMVCore.h"

/*formula Y = AX*/
class lightSpMVCSRKernel {
public:
  ~lightSpMVCSRKernel() {
    cudaFree(_cudaRowCounters_reset);
    CudaCheckError();
    cudaFree(_cudaRowCounters);
    CudaCheckError();
  }

	lightSpMVCSRKernel() {
		/*allocate space*/
	  //_cudaRowCounters.resize(1);

	  cudaMalloc((void**)&_cudaRowCounters_reset, sizeof(*_cudaRowCounters));
	  uint32_t reset = 0;
	  cudaMalloc((void**)&_cudaRowCounters, sizeof(*_cudaRowCounters));

	  cudaMemcpy (_cudaRowCounters_reset, &reset, sizeof(reset), cudaMemcpyHostToDevice);

		/*specify the texture object parameters*/
	  //_texVectorX = 0;
	  //memset(&_texDesc, 0, sizeof(_texDesc));
	  //_texDesc.addressMode[0] = cudaAddressModeClamp;
	  //_texDesc.addressMode[1] = cudaAddressModeClamp;
	  //_texDesc.filterMode = cudaFilterModePoint;
	  //_texDesc.readMode = cudaReadModeElementType;

		/*clear*/
	  //memset(&_resDesc, 0, sizeof(_resDesc));

		/*get GPU information*/
		int device;
		cudaGetDevice(&device);
		CudaCheckError();

		/*get the device property*/
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, device);
		_numThreadsPerBlock = prop.maxThreadsPerBlock;
		_numThreadBlocks = prop.multiProcessorCount
				* (prop.maxThreadsPerMultiProcessor / _numThreadsPerBlock);
		//cerr << _numThreadsPerBlock << " " << _numThreadBlocks << endl;
	}

  inline void spmv(uint32_t num_rows, uint32_t num_cols, uint32_t num_entries, //num entries is the effective number of non zero
		   const uint32_t* row_offsets, const uint32_t* column_indices, const float* values, //graph on device
		   const float*  x, float* y, //on the device
		   cudaStream_t stream = 0) {
    /*initialize the row counter*/
    //_cudaRowCounters[0] = 0;
    cudaMemcpyAsync( _cudaRowCounters, _cudaRowCounters_reset, sizeof(*_cudaRowCounters_reset), cudaMemcpyDeviceToDevice, stream);


    //Erik: let's forget the texturing to begin with
		/*texture object*/
		// _resDesc.resType = cudaResourceTypeLinear;
		// _resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0,
		// 		cudaChannelFormatKindFloat);
		// _resDesc.res.linear.devPtr = (void*) thrust::raw_pointer_cast(x.data());
		// _resDesc.res.linear.sizeInBytes = x.size() * sizeof(float);
		// cudaCreateTextureObject(&_texVectorX, &_resDesc, &_texDesc, NULL);
		// CudaCheckError();

		int meanElementsPerRow = rint((double) num_entries / num_rows);

		/*invoke the kernel*/
		if (meanElementsPerRow <= 2) {
			lightspmv::csr32DynamicWarp<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
			  _numThreadBlocks, _numThreadsPerBlock, 0, stream>>>(
					_cudaRowCounters,
					num_rows, num_cols,
					row_offsets, column_indices, values,
					x, y);
		} else if (meanElementsPerRow <= 4) {
			lightspmv::csr32DynamicWarp<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
			  _numThreadBlocks, _numThreadsPerBlock, 0, stream>>>(
					_cudaRowCounters,
					num_rows, num_cols,
					row_offsets, column_indices, values,
					x, y);
		} else if (meanElementsPerRow <= 64) {
			lightspmv::csr32DynamicWarp<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
			  _numThreadBlocks, _numThreadsPerBlock, 0, stream>>>(
					_cudaRowCounters,
					num_rows, num_cols,
					row_offsets, column_indices, values,
					x, y);

		} else {
			lightspmv::csr32DynamicWarp<float, 32,
					MAX_NUM_THREADS_PER_BLOCK / 32><<<_numThreadBlocks,
			  _numThreadsPerBlock, 0, stream>>>(
					_cudaRowCounters,
					num_rows, num_cols,
					row_offsets, column_indices, values,
					x, y);
		}
		/*Do Not Synchronize*/
		//cudaDeviceSynchronize();
	}

	inline void spmvBLAS(uint32_t num_rows, uint32_t num_cols, uint32_t num_entries,
			     const uint32_t* row_offsets, const uint32_t* column_indices, const float* values, //graph on device
			     const float* x, float* y, //on teh device
			     const float alpha, const float beta,
			     cudaStream_t stream = 0 ) {
		/*initialize the row counter*/
	  cudaMemcpyAsync( _cudaRowCounters, _cudaRowCounters_reset, sizeof(*_cudaRowCounters_reset), cudaMemcpyDeviceToDevice, stream);

		/*texture object*/
		// _resDesc.resType = cudaResourceTypeLinear;
		// _resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0,
		// 		cudaChannelFormatKindFloat);
		// _resDesc.res.linear.devPtr = (void*) x;
		// _resDesc.res.linear.sizeInBytes = num_cols * sizeof(float);
		// cudaCreateTextureObject(&_texVectorX, &_resDesc, &_texDesc, NULL);
		// CudaCheckError();

		// std::cout<<"SHOULD NOT APPEAR"<<std::endl;

	  const float* _texVectorX = x; //Hack to avoid texture memory allocation


		int meanElementsPerRow = rint((double) num_entries / num_rows);

		/*invoke the kernel*/
		if (meanElementsPerRow <= 2) {
			lightspmv::csr32DynamicWarpBLAS<float, 2,
					MAX_NUM_THREADS_PER_BLOCK / 2><<<_numThreadBlocks,
			  _numThreadsPerBlock, 0, stream>>>(
					_cudaRowCounters,
					num_rows, num_cols,
					row_offsets, column_indices, values,
					//x, y,
					_texVectorX, y,
					alpha, beta);
		} else if (meanElementsPerRow <= 4) {
			lightspmv::csr32DynamicWarpBLAS<float, 4,
					MAX_NUM_THREADS_PER_BLOCK / 4><<<_numThreadBlocks,
			  _numThreadsPerBlock, 0, stream>>>(
					_cudaRowCounters,
					num_rows, num_cols,
					row_offsets, column_indices, values,
					//x, y,
					_texVectorX, y,
					alpha, beta);
		} else if (meanElementsPerRow <= 64) {
			lightspmv::csr32DynamicWarpBLAS<float, 8,
							MAX_NUM_THREADS_PER_BLOCK / 8><<<_numThreadBlocks, _numThreadsPerBlock, 0, stream>>>(
					_cudaRowCounters,
					num_rows, num_cols,
					row_offsets, column_indices, values,
					//x, y,
					_texVectorX, y,
					alpha, beta);

		} else {
			lightspmv::csr32DynamicWarpBLAS<float, 32,
					MAX_NUM_THREADS_PER_BLOCK / 32><<<_numThreadBlocks,
			  _numThreadsPerBlock, 0, stream>>>(
					_cudaRowCounters,
					num_rows, num_cols,
					row_offsets, column_indices, values,
					//x, y,
					_texVectorX, y,
					alpha, beta);
		}
		/*DO NOT synchronize*/
		//cudaDeviceSynchronize();
	}


private:
  uint32_t* _cudaRowCounters_reset; //on host
  uint32_t* _cudaRowCounters; //on device memory

	/*for texture object*/
  //cudaTextureDesc _texDesc;
  //cudaResourceDesc _resDesc;
  //cudaTextureObject_t _texVectorX;
	int _numThreadsPerBlock;
	int _numThreadBlocks;
};
