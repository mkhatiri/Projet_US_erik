#include "timestamp.hpp"
#include <string>
//#include "graph.hpp"
#include "graphIO.hpp"
#include <fstream>
#include <string.h>

#include <xmmintrin.h>



template<typename VertexType, typename EdgeType, typename Scalar>
int main_pr(VertexType nVtx, EdgeType*xadj, VertexType *adj, Scalar* val, Scalar *prior, Scalar* pr,
	    Scalar lambda,
	      int nTry, //algo parameter
	      util::timestamp& totaltime, std::string& algo_out
	      );

int main(int argc, char *argv[])
{
#ifdef MPI_EN
  MPI_Init (&argc, &argv);

  int rank = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  int nVtx, nEdge, *xadj=nullptr, *adj=nullptr, *tadj=nullptr,  nTry = 1;
  char *filename = argv[1];
  char timestr[20];
  
  if (argc < 2) {
      fprintf(stderr, "usage: %s <filename> [nTry]\n", argv[0]);
      exit(1);
  }
    
  if (argc >= 3) {
      nTry = atoi(argv[2]);
      if (!nTry) {
          fprintf(stderr, "nTry was 0, setting it to 1\n");
          nTry = 1;
      } 
  } 


  float* val=nullptr;

  //  ReadGraph<int,int,float>(argv[1], &nVtx, &nCol, &xadj, &adj, &val, NULL);
  ReadGraph<int,int,float>(argv[1], &nVtx, &xadj, &adj, &tadj, &val, NULL, NULL);

  //  generateDense<int,int,float>(&nVtx, &nCol, &xadj, &adj, &val, 16*1024);
  //  generateBanded<int,int,float>(&nVtx, &nCol, &xadj, &adj, &val, 16*1024, 128);

  nEdge = xadj[nVtx];

  //  WriteBinary<int,int,float>("foo.bin", nVtx, nCol, xadj, adj, val);
  //  return 0;

  // float* out = new float[nVtx];
  // float* in = new float[nVtx];

  int nVtxalloc = nVtx;
  nVtxalloc += 64-(nVtx%64);

  float* pr = (float*) _mm_malloc(nVtxalloc*sizeof(float), 64);
  float* prior = (float*) _mm_malloc(nVtxalloc*sizeof(float), 64);
  float lambda = .95;

  for (int i=0; i<nVtx; ++i)
    prior[i] = (float) (1./nVtx);

  std::cerr.precision(10);
  std::cout << "prior=" << prior[0] << std::endl;

  //alarm(2400); //most likely our run will not take that long. When the alarm is triggered, there will be no callback to process it and th eprocess will die.

  if (strrchr(argv[1], '/'))
	  filename = strrchr(argv[1], '/') + 1;

  util::timestamp totaltime(0,0);  

  std::string algo_out;

  std::cout<<"graph read"<<std::endl;

  //fixing nnz values to match page rank
  if (!val)
	  val = new float[xadj[nVtx]];
  for (int i=0; i< nVtx; ++i) {
	  for (int p = xadj[i]; p< xadj[i+1]; ++p)
		  val[p] = 1./(xadj[i+1] - xadj[i]);
  }


  std::cerr<<"filename: "<< filename << " ";
  main_pr<int,int,float>(nVtx, xadj, adj, val, prior, pr, lambda, nTry, totaltime, algo_out);

  totaltime /= nTry;
  totaltime.to_c_str(timestr, 20);

#ifdef MPI_EN
  if (rank == 0) {
#endif
	  std::cout<<"filename: "<<filename
		  <<" nVtx: "<<nVtx
		  <<" nonzero: "<<nEdge
		  <<" AvgTime: "<<(float)totaltime
		  <<" "<<algo_out; //<<std::endl;


#ifdef MPI_EN
  }

  MPI_Finalize();
#endif

  //   std::ofstream outfile ("a");
  //   for (int i=0; i< nVtx; ++i) {
  //     outfile<<out[i]<<'\n';
  //   }
  //   outfile<<std::flush;
  //   outfile.close();



  // delete[] in;
  // delete[] out;

  //  GraphFree<int,int,float> (xadj, adj, val);

  return 0;
}
