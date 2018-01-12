#define MPI_EN
#include <mpi.h>

#include "main-pr.hpp"

#define THROW_AWAY 0
#include "Padded2DArray.hpp"
#include <omp.h>
#include "memutils.hpp"
#include <cmath>
#include <sstream>

//#define LOG

#include "Model.hpp"

template <typename VertexType, typename EdgeType, typename Scalar>
int main_pr(VertexType nVtx, EdgeType* xadj_, VertexType *adj_, Scalar* val_, Scalar *prior_, Scalar* pr_,
	    Scalar lambda,
	      int nTry, //algo parameter
	      util::timestamp& totaltime, std::string& out
	      )
{
  bool coldcache = true;
  
  util::timestamp start(0,0);

  Scalar* prin_ = new Scalar[nVtx];

#ifdef LOG
  LoggedArray<EdgeType, logtextomp> xadj (xadj_, (nVtx+1)*sizeof(xadj_[0]), "xadj");
  LoggedArray<VertexType, logtextomp> adj (adj_, xadj_[nVtx]*sizeof(adj_[0]), "adj");
  LoggedArray<Scalar, logtextomp> val (val_, xadj_[nVtx]*sizeof(val_[0]), "val");

  LoggedArray<Scalar, logtextomp> prior (prior_, nVtx*sizeof(in_[0]), "prior");
  LoggedArray<Scalar, logtextomp> prin (prin_, nVtx*sizeof(out_[0]), "prin");

  LoggedArray<Scalar, logtextomp> prout (pr_, nVtx*sizeof(out_[0]), "prout");

#else
  EdgeType* xadj = xadj_;
  VertexType *adj = adj_;
  Scalar* val = val_;
  Scalar* prior = prior_;
  Scalar* prin = prin_;
  Scalar* prout = pr_;
#endif

  VertexType lastinzero = 0;
  //using miguet pierson load balancing algorithm. cut at WORK/P                                                                                               
  //TODO:should probably bin search it                                                                                                                         
  for (lastinzero = 0; lastinzero != nVtx; ++lastinzero) {
    if (xadj[lastinzero] > xadj[nVtx]/2)
      break;
  }

  std::cerr<<"last in node 0: "<<lastinzero<<" edges: "<<xadj[lastinzero]<<std::endl;

  int rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  VertexType startv, endv;
  VertexType remotestartv, remoteendv;

  if (rank == 0) {
    startv = 0;
    endv = lastinzero;
    
    remotestartv = lastinzero+1;
    remoteendv = nVtx-1;
  }
  else {
    remotestartv = 0;
    remoteendv = lastinzero;
    
    startv = lastinzero+1;
    endv = nVtx-1;
  }

  std::cerr<<"startv:"<<startv<<" endv:"<<endv<<" remotestartv:"<<remotestartv<<" remoteendv:"<<remoteendv<<std::endl;


  Model m;

  for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
    {
      if (TRY >= THROW_AWAY)
	start = util::timestamp();

      std::copy (prior+startv, prior+endv+1, prin+startv);

      for (int iter = 0; iter < 40 ; ++ iter) {

	m.begin("iter");

	//setup prin

	m.begin("recv");
	MPI_Request request_r;
	MPI_Request request_s;

	MPI_Status sta;



	//exchange half prin with other node
	int err;
	err = MPI_Irecv (prin+remotestartv, (remoteendv-remotestartv+1), MPI_FLOAT, (rank+1)%2, 0, MPI_COMM_WORLD, &request_r);
	if (err != MPI_SUCCESS)
	  std::cerr<<"plop"<<std::endl;
	err = MPI_Isend (prin+startv, (endv-startv+1), MPI_FLOAT, (rank+1)%2, 0, MPI_COMM_WORLD, &request_s);
	if (err != MPI_SUCCESS)
	  std::cerr<<"plop"<<std::endl;
	//half of prin comes from the previous iteration

	
	//sync
	err = MPI_Wait (&request_r, &sta);
	if (err != MPI_SUCCESS)
	  std::cerr<<"plop"<<std::endl;

	m.end("recv");

	Scalar eps = 0.;

	//prout = A prin
	{	  
	  m.begin("spmv");

#pragma omp parallel for schedule(runtime)
	  for (VertexType i = startv; i < endv+1; ++i)
	    {
	      
	      Scalar output = 0.;
	      
	      EdgeType beg = xadj[i];
	      EdgeType end = xadj[i+1];
	      
	      for (EdgeType p = beg; p < end; ++p)
		{
		  output += prin[adj[p]] * val[p];
		}
	      
	      prout[i] = output;
	    }

	  m.end("spmv");

	  m.begin("prior+eps");

	  //prout = lambda * prout + (1-lambda) prior
#pragma omp parallel for schedule(dynamic,512)
	  for (VertexType i = startv; i < endv+1; ++i) {
	    prout[i] = lambda * prout[i] + (1.-lambda)*prior[i];
	  }	  

	  //compute epsilon
	  eps = 0.;
#pragma omp parallel for schedule(dynamic,512) reduction(+:eps)
	  for (VertexType i = startv; i < endv+1; ++i) {
	    eps += std::abs(prout[i] - prin[i]);
	  }	  
	  
	  m.end("prior+eps");

	err = MPI_Wait (&request_s, &sta);
	if (err != MPI_SUCCESS)
	  std::cerr<<"plop"<<std::endl;

	  float remote_eps;

	  if (rank == 0) {
	    MPI_Recv(&remote_eps, 1, MPI_FLOAT, (rank+1)%2, 0, MPI_COMM_WORLD, &sta);
	    eps += remote_eps;
	  }
	  else
	    MPI_Send(&eps, 1, MPI_FLOAT, (rank+1)%2, 0, MPI_COMM_WORLD);
	  }

	  
	  //	  std::copy (prout+startv, prout+endv+1, prin+startv);	  
	  std::swap(prout,prin);
	  
	
	//stopping condition
	if (eps < 0) // deactivited for testing purposes
	  iter = 40;

	if (rank == 0)
	  std::cerr<<eps<<std::endl;
	
	m.end("iter");
      }

      std::cout<<"rank 0"<<std::endl;
      if (rank == 0)
      	m.dump();
      MPI_Barrier(MPI_COMM_WORLD);
      std::cout<<"rank 1"<<std::endl;
      if (rank == 1) {
      	m.dump();
      }

      MPI_Barrier(MPI_COMM_WORLD);
      
      if (TRY >= THROW_AWAY)
	{
	  util::timestamp stop;  
	  totaltime += stop - start;
	}
      

    }


  delete[] prin_;


  {
    std::stringstream ss;
    ss<<"part1V: "<<lastinzero+1<<" part1E: "<<xadj[lastinzero+2]
      <<" part2V: "<<nVtx-(lastinzero+1)<<" part2E: "<< xadj[nVtx] - xadj[lastinzero+2];
    out = ss.str();
  }


  return 0;
}

  
  
