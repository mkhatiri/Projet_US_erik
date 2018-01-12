#include "main-pr.hpp"

#define THROW_AWAY 0
#include "Padded2DArray.hpp"
#include <omp.h>
#include "memutils.hpp"
#include <cmath>
#include <list>
#include <map>
#include <atomic>

//#define SHOWLOADBALANCE
#include "logged_array.hpp"

#define MODEL
#include "Model.hpp"
//#define LOG


template <typename VertexType, typename EdgeType, typename Scalar>
int main_pr(VertexType nVtx, EdgeType* xadj_, VertexType *adj_, Scalar* val_, Scalar *prior_, Scalar* pr_,
	    Scalar lambda,
	      int nTry, //algo parameter
	      util::timestamp& totaltime, std::string& 
	      )
{
  Model m;
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
  
#ifdef SHOWLOADBALANCE
  Padded1DArray<int> count (244);

  for (int i=0; i<244; ++i)
    count[i] = 0;
#endif


  for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
    {
      if (TRY >= THROW_AWAY)
	start = util::timestamp();

      for (int iter = 0; iter < 40 ; ++ iter) {

	m.begin("iter");

	Scalar eps = 0.;

	
	//prout = A prin
	{
	  if (iter == 0) {
	    Scalar* from ;
	    //setup prin
	    from = prior;
	    
#pragma omp parallel for schedule(runtime)
	    for (VertexType i=0; i<nVtx; ++i)
	      prin[i] = from[i];
	  }
	  else {
	    std::swap(prin, prout);
	  }
	}

	m.begin("spmv");

	//#define SPMV_OPT

#ifdef SPMV_OPT
	

	{
	  double frac_threshold = 0.01;
	  VertexType* longs = new VertexType[(EdgeType)(1./frac_threshold) + 2];
	  std::atomic<VertexType> endlong;
	  endlong = 0;

#pragma omp parallel for schedule(runtime)
	  for (VertexType i = 0; i < nVtx; ++i)
	    {
	      
	      Scalar output = 0.;
	      


	      EdgeType beg = xadj[i];
	      EdgeType end = xadj[i+1];

	      if (end-beg > frac_threshold * (xadj[nVtx]-xadj[0]) )  {
		//very long row
		
		VertexType wheretowrite = std::atomic_fetch_add(&endlong, 1);
		longs[wheretowrite] = i;
	      }
	      else {
		//average row
		for (EdgeType p = beg; p < end; ++p)
		  {
		    output += prin[adj[p]] * val[p];
		  }
	      
		prout[i] = output;
	      }
	    }

	  //process longs
	  for (VertexType i = 0; i < endlong; ++i) {
	    VertexType to = longs [i];

	    Scalar output = 0.;
	    EdgeType beg = xadj[to];
	    EdgeType end = xadj[to+1];

#pragma omp parallel for schedule(dynamic,128) reduction(+:output)
	    for (EdgeType p = beg; p < end; ++p) {
		output += prin[adj[p]] * val[p];
	      }


	    prout[to] = output;
	  }

	  delete[] longs;
	}

#else

#pragma omp parallel for schedule(runtime)
	for (VertexType i = 0; i < nVtx; ++i)
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




#endif
	
	m.end("spmv");

	m.begin("prior+eps");

	//prout = lambda * prout + (1-lambda) prior
	//#pragma omp parallel for schedule(runtime)
	#pragma omp parallel for schedule(dynamic,512)
	for (VertexType i = 0; i < nVtx; ++i) {
	  prout[i] = lambda * prout[i] + ((Scalar)(1.)-lambda)*prior[i];
	}	  
	
	//compute epsilon
	eps = 0.;
	//#pragma omp parallel for schedule(runtime) reduction(+:eps)
	#pragma omp parallel for schedule(dynamic,512) reduction(+:eps)
	for (VertexType i = 0; i < nVtx; ++i) {
	  eps += std::abs(prout[i] - prin[i]);
	}	  
	
	m.end("prior+eps");

	//stopping condition
	if (eps < 0) // deactivited for testing purposes
	  iter = 40;

	std::cerr<<eps<<std::endl;

	m.end("iter");
      }

      if (TRY >= THROW_AWAY)
	{
	  util::timestamp stop;  
	  totaltime += stop - start;
	}
      
      m.dump();

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

    }

#ifdef SHOWLOADBALANCE
  std::cout<<"load balance"<<std::endl;
  for (int i=0; i< 244; ++i)
    std::cout<<count[i]<<std::endl;
#endif

  delete[] prin_;

  return 0;
}

  
  
