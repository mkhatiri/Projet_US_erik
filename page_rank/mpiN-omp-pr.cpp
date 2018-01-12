#define MPI_EN
#include <mpi.h>

#include "main-pr.hpp"

#define THROW_AWAY 0
#include "Padded2DArray.hpp"
#include <omp.h>
#include "memutils.hpp"
#include <cmath>
#include <sstream>
#include <vector>

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

  int nbRanks;
  MPI_Comm_size(MPI_COMM_WORLD, &nbRanks);

  std::vector<VertexType> partition (nbRanks+1);
  std::vector<VertexType> partsize (nbRanks);

  int PART = 0;
  if (PART == 0) {
    //using miguet pierson load balancing algorithm. cut at WORK/P                                                                                               
    //TODO:should probably bin search it

    partition[0] = 0;                                                
    for (int i = 1 ; i < nbRanks; ++i) {
      VertexType end = partition[i-1];
      EdgeType remainingwork = xadj[nVtx] - xadj[partition[i-1]];
      int remainingRanks = nbRanks - (i - 1);
      double balanced = ((double) remainingwork) / remainingRanks;
      while (end <= nVtx && xadj[end] - xadj[partition[i-1]] < balanced) {
	end++;
      }
      end++;
      if (end > nVtx)
	end = nVtx;
      partition[i] = end;
    }
    partition[nbRanks]=nVtx;
  }
  if (PART == 1) {
    double netBW  = 1.05*1000*1000*1000;
    double coreBW = 24. *1000*1000*1000;

    partition[0] = 0;                                                
    for (int i = 1 ; i < nbRanks; ++i) {

      VertexType candidate = partition[i-1];
      int Pr = nbRanks - (i - 1) - 1; //remaining processor after the current one
      double cand_value = std::max( 
				   //realistic me
				   std::max(4.*(candidate-partition[i-1]+1)*(nbRanks-1)/netBW,//send
					    4.*(nVtx-(candidate-partition[i-1]+1))/netBW)//receive
				   +  (4. + 32.*(candidate-partition[i-1])+8.*(xadj[candidate+1]-partition[i-1]))/coreBW,//comp
				   //optimistic rest
				   std::max( ((double)nbRanks-1)*(4.*(nVtx-candidate-1)/Pr)/netBW,//send
					     4.*(nVtx-((double)nVtx-candidate-1)/Pr)/netBW)//receive
				   +  (4. + 32./Pr*(nVtx-candidate-1)+8./Pr*(xadj[nVtx]-xadj[candidate+1]))/coreBW//comp
				    );
			      

      for (int tryout = candidate+1; tryout<nVtx; ++tryout) {
	double value = std::max( 
				//realistic me
				std::max(4.*(tryout-partition[i-1]+1)*(nbRanks-1)/netBW,
					 4.*(nVtx-(tryout-partition[i-1]+1))/netBW)
				+  (4. + 32.*(tryout-partition[i-1])+8.*(xadj[tryout+1]-partition[i-1]))/coreBW,
				//optimistic rest
				std::max( ((double)nbRanks-1)*(4.*(nVtx-tryout-1)/Pr)/netBW,
					  4.*(nVtx-((double)nVtx-tryout-1)/Pr)/netBW)
				+  (4. + 32./Pr*(nVtx-tryout-1)+8./Pr*(xadj[nVtx]-xadj[tryout+1]))/coreBW
				 );

	if (value < cand_value) {
	  cand_value = value;
	  candidate = tryout;
	}

      }
      
      if (candidate > nVtx)
	candidate = nVtx;
      partition[i] = candidate;
    }
    partition[nbRanks]=nVtx;
  }
  if (PART == 2) {
    double netBW  = 1.05*1000*1000*1000;
    double coreBW = 24. *1000*1000*1000;






    partition[0] = 0;                                                
    for (int i = 1 ; i < nbRanks; ++i) {
      VertexType end = partition[i-1];
      VertexType remainingvertices = nVtx-end;
      EdgeType remainingedge = xadj[nVtx] - xadj[partition[i-1]];
      int remainingRanks = nbRanks - (i - 1);
      double remainingWork = 4.*(nbRanks-1)*remainingvertices/netBW
	+ (4 + 32.*remainingvertices + 8.*remainingedge)/coreBW;

      double balanced = ((double) remainingWork) / remainingRanks;


      while (end <= nVtx && 
	     (4.*(nbRanks-1)*(end-partition[i-1]))/netBW
	     + (4 + 32.*(end-partition[i-1])  + 8.*(xadj[end] - xadj[partition[i-1]]))/coreBW
	     < balanced) {
	end++;
      }
      end++;
      if (end > nVtx)
	end = nVtx;
      partition[i] = end;
    }
    partition[nbRanks]=nVtx;




    // partition[0] = 0;                                                
    // for (int i = 1 ; i < nbRanks; ++i) {

    //   VertexType candidate = partition[i-1];
    //   int Pr = nbRanks - (i - 1) - 1; //remaining processor after the current one
    //   double cand_value = std::max( 
    // 				   //realistic me
    // 				   4.*(candidate-partition[i-1]+1)*(nbRanks-1)/netBW+//send
    // 				   (4. + 32.*(candidate-partition[i-1])+8.*(xadj[candidate+1]-partition[i-1]))/coreBW,//comp
    // 				   //optimistic rest
    // 				   ((double)nbRanks-1)*(4.*(nVtx-candidate-1)/Pr)/netBW+//send
    // 				   (4. + 32./Pr*(nVtx-candidate-1)+8./Pr*(xadj[nVtx]-xadj[candidate+1]))/coreBW//comp
    // 				    );
			      

    //   for (int tryout = candidate+1; tryout<nVtx; ++tryout) {
    // 	double value = std::max( 
    // 				//realistic me
    // 				4.*(tryout-partition[i-1]+1)*(nbRanks-1)/netBW+
    // 				 (4. + 32.*(tryout-partition[i-1])+8.*(xadj[tryout+1]-partition[i-1]))/coreBW,
    // 				//optimistic rest
    // 				((double)nbRanks-1)*(4.*(nVtx-tryout-1)/Pr)/netBW+
    // 				(4. + 32./Pr*(nVtx-tryout-1)+8./Pr*(xadj[nVtx]-xadj[tryout+1]))/coreBW
    // 				 );

    // 	if (value < cand_value) {
    // 	  cand_value = value;
    // 	  candidate = tryout;
    // 	}

    //   }
      
    //   if (candidate > nVtx)
    // 	candidate = nVtx;
    //   partition[i] = candidate;
    // }
    // partition[nbRanks]=nVtx;
  }



  for (int i=0; i<nbRanks; ++i)
    partsize[i] = partition[i+1]-partition[i];

  //partition is set so that "i" is  responsible for [partition[i]; partition[i+1][

  //std::cerr<<"last in node 0: "<<lastinzero<<" edges: "<<xadj[lastinzero]<<std::endl;

  int rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);


  //std::cerr<<"startv:"<<startv<<" endv:"<<endv<<" remotestartv:"<<remotestartv<<" remoteendv:"<<remoteendv<<std::endl;

  Model m;


  std::cerr<<"maxthread: "<<omp_get_max_threads()<<std::endl;

  for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
    {

      MPI_Barrier(MPI_COMM_WORLD);

      if (TRY >= THROW_AWAY)
	start = util::timestamp();

      //std::copy (prior+startv, prior+endv+1, prin+startv);
      std::copy (prior+partition[rank], prior+partition[rank+1], prin+partition[rank]);

      std::vector<MPI_Request> request_r (nbRanks);
      std::vector<MPI_Request> request_s (nbRanks);
      for (int iter = 0; iter < 40 ; ++ iter) {

	m.begin("iter");

	//setup prin

	m.begin("recv");


	//exchange prin with other nodes
	//half of prin comes from the previous iteration
	MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
		       prin, &(partsize[0]), &(partition[0]), MPI_FLOAT,
		       MPI_COMM_WORLD);

	m.end("recv");

	Scalar eps = 0.;

	//prout = A prin
	VertexType startv = partition[rank];
	VertexType endv = partition[rank+1]-1;
      
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
	}

	float reduced_eps;
	
	//if (0)
	MPI_Reduce(&eps, &reduced_eps, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	
	eps = reduced_eps;
	
	//	  std::copy (prout+startv, prout+endv+1, prin+startv);	  
	std::swap(prout,prin);
	
	
	//stopping condition
	if (eps < 0) // deactivited for testing purposes
	  iter = 40;
	
	if (rank == 0)
	  std::cerr<<eps<<std::endl;
	
	m.end("iter");
      }
      

      {
	std::stringstream ss;
	ss<<rank;
	std::ofstream ofi (ss.str());
	m.dump(ofi);
      }
      // std::cout<<"rank 1"<<std::endl;
      // if (rank == 1) {
      // 	m.dump();
      // }

	//MPI_Barrier(MPI_COMM_WORLD);
      
      if (TRY >= THROW_AWAY)
	{
	  util::timestamp stop;  
	  totaltime += stop - start;
	}
      

    }


  delete[] prin_;


  {
    std::stringstream ss;
    for (int i=0; i< nbRanks; ++i)
      ss<<"part"<<i+1<<"V: "<<partition[i+1]-partition[i]<<" part"<<i+1<<"E: "<<xadj[partition[i+1]] - xadj[partition[i]]<<" ";

    out = ss.str();
  }


  return 0;
}

  
  
