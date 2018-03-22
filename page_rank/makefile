

CXXFLAGS += 

MKL_LDFLAGS += 

INCLUDES   = -I./

TARGET     =  omp-pr

CXXFILES   = omp-pr

SRCS	= $(CFILES:%=%.c) $(CXXFILES:%=%.cpp)

TBBLDFLAGS= -L /opt/intel/composerxe/tbb/lib/mic -ltbb 


all: $(TARGET)


mpi2-omp-pr: mpi2-omp-pr.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o mpi2-omp-pr.o $(LIBS) 

mpiN-omp-pr: mpiN-omp-pr.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o mpiN-omp-pr.o $(LIBS) 


omp-pr: omp-pr.o graphIO.o logged_array.o
	$(LD) $(LDFLAGS) -o $@ graphIO.o omp-pr.o logged_array.o $(LIBS) 

cachecount: cachecount.o graphIO.o logged_array.o
	$(LD) $(LDFLAGS) -o $@ graphIO.o cachecount.o logged_array.o $(LIBS) 


cuda-pr: cuda-pr.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o cuda-pr.o $(LIBS) $(CUDALDFLAGS)

new-cuda-pr: new-cuda-pr.o graphIO.o streamUtils.o
	$(LD) $(LDFLAGS) -o $@ graphIO.o streamUtils.o new-cuda-pr.o $(LIBS) $(CUDALDFLAGS) -ltbb


cuda-adaptative: cuda-adaptative.o graphIO.o AdaptativeUtils.o Adaptative_Kernal.o  
	$(LD) $(LDFLAGS) -m64 -o $@ graphIO.o AdaptativeUtils.o cuda-adaptative.o Adaptative_Kernal.o $(LIBS) $(CUDALDFLAGS) -ltbb  

cuda-adaptative-2gpus: cuda-adaptative-2gpus.o graphIO.o AdaptativeUtils.o Adaptative_Kernal.o  
	$(LD) $(LDFLAGS) -m64 -o $@ graphIO.o AdaptativeUtils.o cuda-adaptative-2gpus.o Adaptative_Kernal.o $(LIBS) $(CUDALDFLAGS) -ltbb  

new-cuda-adaptative: new-cuda-adaptative.o graphIO.o AdaptativeUtils.o Adaptative_Kernal.o
	$(LD) $(LDFLAGS) -m64 -o $@ graphIO.o AdaptativeUtils.o new-cuda-adaptative.o Adaptative_Kernal.o $(LIBS) $(CUDALDFLAGS) -ltbb  

new-cuda-adaptative-2gpus: new-cuda-adaptative-2gpus.o graphIO.o AdaptativeUtils.o Adaptative_Kernal.o
	$(LD) $(LDFLAGS) -m64 -o $@ graphIO.o AdaptativeUtils.o new-cuda-adaptative-2gpus.o Adaptative_Kernal.o $(LIBS) $(CUDALDFLAGS) -ltbb

new-cuda-adaptative-2gpus-ws: new-cuda-adaptative-2gpus-ws.o graphIO.o AdaptativeUtils.o Adaptative_Kernal.o
	$(LD) $(LDFLAGS) -m64 -o $@ graphIO.o AdaptativeUtils.o new-cuda-adaptative-2gpus-ws.o Adaptative_Kernal.o $(LIBS) $(CUDALDFLAGS) -ltbb

new-cuda-adaptative-test: new-cuda-adaptative-test.o graphIO.o  AdaptativeUtils.o Adaptative_Kernal.o
	$(LD) $(LDFLAGS)  -m64 -o $@ graphIO.o AdaptativeUtils.o new-cuda-adaptative-test.o Adaptative_Kernal.o $(LIBS) $(CUDALDFLAGS) -ltbb

new-cuda-pr-2gpus: new-cuda-pr-2gpus.o graphIO.o streamUtils.o
	$(LD) $(LDFLAGS) -o $@ graphIO.o streamUtils.o new-cuda-pr-2gpus.o $(LIBS) $(CUDALDFLAGS) -ltbb

cuda-lightspmv-pr: cuda-lightspmv-pr.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o cuda-lightspmv-pr.o $(LIBS) $(CUDALDFLAGS)

cuda-lightspmv-pr-2gpus: cuda-lightspmv-pr-2gpus.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o cuda-lightspmv-pr-2gpus.o $(LIBS) $(CUDALDFLAGS)

new-cuda-lightspmv-pr: new-cuda-lightspmv-pr.o graphIO.o streamUtils.o
	$(LD) $(LDFLAGS) -o $@ graphIO.o streamUtils.o new-cuda-lightspmv-pr.o $(LIBS) $(CUDALDFLAGS) -ltbb

new-cuda-lightspmv-pr-2gpus: new-cuda-lightspmv-pr-2gpus.o graphIO.o streamUtils.o
	$(LD) $(LDFLAGS) -o $@ graphIO.o new-cuda-lightspmv-pr-2gpus.o streamUtils.o $(LIBS) $(CUDALDFLAGS) -ltbb

cuda-pr-2gpus: cuda-pr-2gpus.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o cuda-pr-2gpus.o $(LIBS) $(CUDALDFLAGS)


include makefile.in
# DO NOT DELETE

omp-pr.o: main-pr.hpp timestamp.hpp graphIO.hpp Padded2DArray.hpp
omp-pr.o: memutils.hpp logged_array.hpp Model.hpp
