#include "main-pr.hpp"

#define THROW_AWAY 0
#include "Padded2DArray.hpp"
#include <omp.h>
#include "memutils.hpp"
#include <cmath>
#include <list>
#include <map>

//#define SHOWLOADBALANCE
#include "logged_array.hpp"
#include "cache-simulator.hpp"

#define MODEL
#include "Model.hpp"
//#define LOG

template <typename VertexType, typename EdgeType, typename Scalar>
VertexType cachelines_on_row (VertexType nVtx, EdgeType* xadj_, VertexType *adj_, Scalar* prin_, VertexType row) {

  const int CACHELINESIZE = 64;


#ifdef LOG
  LoggedArray<EdgeType, logtextomp> xadj (xadj_, (nVtx+1)*sizeof(xadj_[0]), "xadj");
  LoggedArray<VertexType, logtextomp> adj (adj_, xadj_[nVtx]*sizeof(adj_[0]), "adj");
  LoggedArray<Scalar, logtextomp> prin (prin_, nVtx*sizeof(out_[0]), "prin");


#else
  EdgeType* xadj = xadj_;
  VertexType *adj = adj_;
  Scalar* prin = prin_;
#endif

  
  VertexType cachelines = 0;
  VertexType last;
  
  EdgeType beg = xadj[row];
  EdgeType end = xadj[row+1];
  
  if (beg == end)
    return cachelines;
  
  
  last = adj[beg];
  cachelines ++;
  
  for (EdgeType p = beg+1; p < end; ++p) {
    VertexType loc = adj[p];
    
    if (((long long int)(prin+last))/CACHELINESIZE != ((long long int)(prin+loc))/CACHELINESIZE) {
      cachelines++;
    }
    
    last = loc;
  }
  return cachelines;
}

template <typename VertexType, typename EdgeType, typename Scalar>
int main_pr(VertexType nVtx, EdgeType* xadj_, VertexType *adj_, Scalar* val_, Scalar *prior_, Scalar* pr_,
	    Scalar lambda,
	      int nTry, //algo parameter
	      util::timestamp& totaltime, std::string& outstr
	      )
{
 
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


  //per row count
  EdgeType cachelines = 0;
#pragma omp parallel for schedule(runtime) reduction(+:cachelines)
  for (VertexType i = 0; i < nVtx; ++i) {
    cachelines += cachelines_on_row (nVtx, xadj_, adj_, prin_, i);
  }

  //per row estimate
  EdgeType cachelines_est = 0;
  double fraction = 0.01;
  EdgeType considered_edges = 0;
#pragma omp parallel for schedule(runtime) reduction(+:cachelines_est) reduction(+:considered_edges)
  for (VertexType i = 0; i < nVtx; ++i) {
    if (drand48() < fraction) {
      cachelines_est += cachelines_on_row (nVtx, xadj_, adj_, prin_, i);
      considered_edges += xadj[i+1]-xadj[i];
    }
  }
  cachelines_est *= (xadj[nVtx] - xadj[0])/(double)considered_edges;

  //per block L2 simulation
  VertexType blockSize = 32;
  int CACHELINE = 64;
  long CacheCapacity= 256*1024;
  EdgeType potential_cachemisses = 0;
#pragma omp parallel for schedule(runtime) reduction(+:potential_cachemisses)
  for (VertexType beg_block = 0; beg_block< nVtx; beg_block+= blockSize) {
    CacheSimulatorFast cache (CACHELINE, CacheCapacity);
    VertexType end_block = std::min(beg_block + blockSize, nVtx);
    
    for (VertexType row = beg_block; row < end_block; ++row) {

      EdgeType beg = xadj[row];
      EdgeType end = xadj[row+1];
      
      for (EdgeType p = beg; p<end; ++p) {
	cache.touch(adj[p]);
      }
      
    }

    potential_cachemisses += cache.getMiss();
  }

  

  {
    std::stringstream ss;

    ss<<"veccachelines: "<<cachelines;
    ss<<" veccachelines_est: "<<cachelines_est;
    ss<<" potentialL2: "<<potential_cachemisses;
    outstr = ss.str();

  }

  delete[] prin_;

  return 0;
}

  
  
