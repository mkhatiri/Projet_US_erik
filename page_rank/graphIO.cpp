/*
---------------------------------------------------------------------
 This file is a part of the source code for the paper "Betweenness
 Centrality on GPUs and Heterogeneous Architectures", published in
 GPGPU'13 workshop. If you use the code, please cite the paper.

 Copyright (c) 2013,
 By:    Ahmet Erdem Sariyuce,
        Kamer Kaya,
        Erik Saule,
        Umit V. Catalyurek
---------------------------------------------------------------------
 This file is licensed under the Apache License. For more licensing
 information, please see the README.txt and LICENSE.txt files in the
 main directory.
---------------------------------------------------------------------
*/


#include <vector>
#include <map>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <cmath>
#include <algorithm>
#include <list>
#include <omp.h>
#include <sys/time.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string.h>
//#include "ulib.h"
//#include "timestamp.hpp"
#include <assert.h>

#define MAXLINE 128*(1024*1024)

#include "graphIO.hpp"

using namespace std;


template <typename VtxType, typename EdgeType, typename WeightType>
void ReadGraphFromFile(FILE *fpin, VtxType *numofvertex, EdgeType **pxadj, VtxType **padjncy, WeightType **padjncyw, WeightType **ppvw) { //chaco
	VtxType *adjncy,  nvtxs, fmt, readew, readvw, k, ncon;
	EdgeType *xadj, nedges; 
	WeightType*adjncyw=NULL, *pvw=NULL;
	char *line;

	line = (char *)malloc(sizeof(char)*(MAXLINE+1));

	do {
	  (void)fgets(line, MAXLINE, fpin);
	} while (line[0] == '%' && !feof(fpin));

	if (feof(fpin)){
		printf("empty graph!!!");
		exit(1);
	}

	fmt = 0;
	{
		std::string s = line;
		std::stringstream ss (s);
		ss>>nvtxs>>nedges>>fmt>>ncon;
	}
	*numofvertex = nvtxs;

	readew = (fmt%10 > 0);
	readvw = ((fmt/10)%10 > 0);
	if (fmt >= 100) {
		printf("invalid format");
		exit(1);
	}

	nedges *=2;

	xadj = *pxadj = (EdgeType*) malloc((nvtxs+2)*sizeof(EdgeType));
	adjncy = *padjncy = (VtxType*) malloc(nedges*sizeof(VtxType));
	if (padjncyw)
	  adjncyw = *padjncyw = (WeightType*) malloc(nedges*sizeof(WeightType));
	if (ppvw)
	  pvw = *ppvw = (WeightType*)malloc((nvtxs+1)*sizeof(WeightType));
	
	xadj[0]=0;
	k=0;
	for (VtxType i=0; i<nvtxs; i++) {
	  //char *oldstr=line, *newstr=NULL;
		WeightType vw=1;

		VtxType edge;

		do {
		  (void)fgets(line, MAXLINE, fpin);
		} while (line[0] == '%' && !feof(fpin));

		if (strlen(line) >= MAXLINE-5){
			printf("\nBuffer for fgets not big enough!\n");
			exit(1);
		}
		
		std::string s = line;
		std::stringstream ss (s);

		if (readvw) {
		  ss>>vw;
		  //vw = (int)strtol(oldstr, &newstr, 10);
		  //oldstr = newstr;
		}

		if (ppvw)
			pvw[i] = vw;

		for (;;) {
		  WeightType ewgt=1;		  
		  ss>>edge;
		  edge -= 1; //in chaco format, the first edge is index 1. We need it indexed 0
		  //edge = (int)strtol(oldstr, &newstr, 10) -1;
		  //oldstr = newstr;

			if (readew) {
			  ss>>ewgt;
			  //ewgt = (int)strtol(oldstr, &newstr, 10);
			  //oldstr = newstr;
			}

			if (!ss) //if (edge < 0)
			  {
			    break;
			  }

			if (edge==i){
			  //printf("Self loop in the graph for vertex %d\n", i);
			  std::cout<<"Self loop in the graph for vertex"<<i<<std::endl;
				exit(1);
			}
			adjncy[k] = edge;
			if (padjncyw)
				adjncyw[k] = ewgt;
			k++;
		}
		xadj[i+1] = k;
	}

	if (k != nedges){
	  std::cout<<"k("<<k<<")!=nedges("<<nedges<<")"<<std::endl;
	  exit(1);
	}

	free(line);

	return;
}

template <typename VtxType, typename EdgeType, typename WeightType>
void ReadGraphFromMMFile(FILE *matfp, VtxType *numofvertex, EdgeType **pxadj, VtxType **padjncy, WeightType **padjncyw, WeightType **ppvw, bool do_mapping, long** reverse_map) {

  std::pair<VtxType, VtxType> *coords, *new_coords;
  VtxType n, m;//number of rows and columns
	EdgeType  nnz, tnnz, onnz;
	VtxType *adj;
	WeightType *adjncyw;
	WeightType *pvw;
	
	EdgeType *xadj;

	int maxLineLength = 1000000;
	char line[maxLineLength];

	/* set return null parameter values, in case we exit with errors */
	m = nnz = 0;


	std::cerr<<"do not trust this matrix market reader to read an actual matrix"<<std::endl;

      	/* now continue scanning until you reach the end-of-comments */
	//TODO: This does not read matrix market at all. It discard type of matrix information.
	do {
	  (void)fgets(line, 1000000, matfp);

	} while (line[0] == '%' && !feof(matfp));

	/* line[] is either blank or has M,N, nz */
	{
	  std::string s (line);
	  std::stringstream ss (s);

	  ss>>m>>n>>nnz;

	  if (!ss) {
	    do {
	      char* ret = fgets (line, maxLineLength, matfp);
	      if (ret == NULL)
		return;

	      std::string s (line);
	      std::stringstream ss (s);
	      ss>>m>>n>>nnz;

	      //num_items_read = fscanf(matfp, "%d %d %lld", &m, &n, &nnz);
	    }
	    while (!ss);
	    //	    while (num_items_read != 3);

	  }
	}

	*reverse_map = (long*) malloc (sizeof(long) * n);
	//	printf("matrix banner is read %d - %d, %lld nnz\n", m, n, nnz);
	coords = (std::pair<VtxType, VtxType>*) malloc(sizeof(std::pair<VtxType, VtxType>) * 2 * nnz);

	tnnz = 0;
	for(EdgeType i = 0; i < nnz; i++) {
	  VtxType itemp, jtemp;
	  (void)fgets (line, maxLineLength, matfp);
	  std::string s (line);
	  std::stringstream ss (s);
	  ss>>itemp>>jtemp;
	  if (!ss) {
	    std::cout<<"format"<<std::endl;
	    exit(-1);
	  }
	    
	  //fscanf(matfp, "%d %d\n", &itemp, &jtemp);
		if(itemp != jtemp) {
			coords[tnnz].first = itemp - 1;
			coords[tnnz++].second = jtemp - 1;
			coords[tnnz].first = jtemp - 1;
			coords[tnnz++].second = itemp - 1;
		}
	}
	//	printf("matrix is read %d - %d, %lld nnz with duplicates\n", m, n, tnnz);

	
	std::sort(coords, coords+tnnz, 
		  [] ( const std::pair<VtxType,VtxType> & a, const std::pair<VtxType,VtxType> & b ) 
		  { 
		    if (a.first != b.first) 
		      return a.first < b.first;
		    else
		      return a.second < b.second;
					} );
	//qsort(coords, tnnz, sizeof(Pair), pcmp);

	onnz = 1;
	for(EdgeType i = 1; i < tnnz; i++) {
		if(coords[i].first != coords[onnz-1].first || coords[i].second != coords[onnz-1].second) {
			coords[onnz].first = coords[i].first;
			coords[onnz++].second = coords[i].second;
		}
	}

	*numofvertex = n;
	xadj = *pxadj = (EdgeType*) malloc((n+1) * sizeof(EdgeType));
	adj = *padjncy = (VtxType*) malloc(onnz * sizeof(VtxType));
	if (padjncyw)
	  adjncyw = *padjncyw = (WeightType*) malloc (nnz*sizeof(WeightType));
	if (ppvw)
	  pvw = *ppvw = (WeightType*) malloc ((n+1)*sizeof(WeightType));

	if (do_mapping) {
		map<long, VtxType> reed;
		
		new_coords = (std::pair<VtxType,VtxType>*) malloc(sizeof(std::pair<VtxType,VtxType>) * 2 * nnz);
		long vno = 0;
		// map the ids
		for(EdgeType i = 0; i < onnz; i++) {
			VtxType temp = coords[i].first;
			auto reed_it = reed.find(temp);
			if (reed_it == reed.end()) {
			  std::pair<VtxType,VtxType> temppair;
			  temppair.first = temp;
			  temppair.second = vno;
				reed.insert (temppair);
				(*reverse_map)[vno] = temp;
				new_coords[i].first = vno++;
			}
			else
				new_coords[i].first = reed_it->second;

			temp = coords[i].second;
			reed_it = reed.find(temp);
			if (reed_it == reed.end()) {
			  std::pair<VtxType,VtxType> temppair;
			  temppair.first = temp;
			  temppair.second = vno;
				reed.insert (temppair);
				(*reverse_map)[vno] = temp;
				new_coords[i].second = vno++;
			}
			else
				new_coords[i].second = reed_it->second;
		}
	}

	vector<vector<VtxType> > entire_graph;
	entire_graph.resize(n);
	for(EdgeType i = 0; i < onnz; i++) {
		if (do_mapping)
			entire_graph[new_coords[i].first].push_back(new_coords[i].second);
		else
			entire_graph[coords[i].first].push_back(coords[i].second);
	}
	xadj[0] = 0;
	EdgeType j = 0;
	for(VtxType i = 1; i < n+1; i++) {
		xadj[i] = xadj[i-1] + entire_graph[i-1].size();
		for (VtxType k = 0; k < entire_graph[i-1].size(); k++) {
			adj[j++] = entire_graph[i-1][k];
		}
	}

	//	printf("matrix is read %d - %d, %lld onnz\n", m, n, onnz);
	free(coords);
	//this is useless vector destructor will take care of that
	// for(i = 0; i < m; i++)
	// 	entire_graph[i].clear();

	// entire_graph.clear();

	return;
}

static int really_read(std::istream& is, char* buf, size_t global_size) {
	char* temp2 = buf;
	while (global_size != 0) {
		is.read(temp2, global_size);
		size_t s = is.gcount();
		if (!is)
			return -1;

		global_size -= s;
		temp2 += s;
	}
	return 0;
}

template <typename VtxType, typename EdgeType, typename WeightType>
void ReadBinary(char *filename, VtxType *numofvertex_r, VtxType *numofvertex_c, EdgeType **pxadj, VtxType **padj, WeightType **padjw, WeightType **ppvw) {

	if (ppvw != NULL) {
		cerr<<"vertex weight is unsupported"<<std::endl;
		return;
	}

	std::ifstream in (filename);

	if (!in.is_open()) {
		cerr<<"can not open file:"<<filename<<std::endl;
		return;
	}

 
//	char vtxsize; //in bytes
//	char edgesize; //in bytes
//	char weightsize; //in bytes
	int vtxsize; //in bytes
	int edgesize; //in bytes
	int weightsize; //in bytes

	//reading header
	//in.read(&vtxsize, 1);
	//in.read(&edgesize, 1);
	//in.read(&weightsize, 1);
	in.read((char *)&vtxsize, sizeof(int));
	in.read((char *)&edgesize, sizeof(int));
	in.read((char *)&weightsize, sizeof(int));

//	printf("vtxsize: %d\n", vtxsize);
//	printf("edgesize: %d\n", edgesize);
//	printf("weightsize: %d\n", weightsize);


	std::cout<<"vtxsize: "<<vtxsize<<std::endl;
	std::cout<<"edgesize: "<<edgesize<<std::endl;
	std::cout<<"weightsize: "<<weightsize<<std::endl;

	if (!in) {
		cerr<<"IOError"<<std::endl;
		return;
	}

	if (vtxsize != sizeof(VtxType)) {
		cerr<<"Incompatible VertexSize."<<endl;
		return;
	}

	if (edgesize != sizeof(EdgeType)) {
		cerr<<"Incompatible EdgeSize."<<endl;
		return;
	}

	if (weightsize != sizeof(WeightType)) {
		cerr<<"Incompatible WeightType."<<endl;
		return;
	}

	//reading should be fine from now on.
	in.read((char*)numofvertex_r, sizeof(VtxType));
	in.read((char*)numofvertex_c, sizeof(VtxType));
	EdgeType nnz;
	in.read((char*)&nnz, sizeof(EdgeType));
	if (*numofvertex_c <=0 || *numofvertex_r <=0 || nnz <= 0) {
		cerr<<"graph makes no sense"<<endl;
		return;
	}

	cout<<"nVtx: "<<*numofvertex_r<<endl;
	cout<<"nVtx: "<<*numofvertex_c<<endl;
	cout<<"nEdge: "<<nnz<<endl;
	//printf("nVtx: %d, nVtx: %d, nEdge: %d\n", *numofvertex_r, *numofvertex_c, nnz);
	std::cout<<"nVtx: "<<*numofvertex_r<<", nVtx: "<< *numofvertex_c<<", nEdge: "<< nnz<<std::endl;

	*pxadj = (EdgeType*) malloc (sizeof(EdgeType) * (*numofvertex_r+1));
	*padj =  (VtxType*) malloc (sizeof(VtxType) * (nnz));


	if (padjw) {
		*padjw = new WeightType[nnz];
	}

	int err = really_read(in, (char*)*pxadj, sizeof(EdgeType)*(*numofvertex_r+1));
	err += really_read(in, (char*)*padj, sizeof(VtxType)*(nnz));
	if (padjw)
		err += really_read(in, (char*)*padjw, sizeof(WeightType)*(nnz));
	if (!in || err != 0) {
		cerr<<"IOError"<<endl;
	}

	return;
}

bool has_suffix(const std::string &str, const std::string &suffix)
{
  return str.size() >= suffix.size() &&
    str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

template <typename VtxType, typename EdgeType>
bool isGraphSorted (VtxType nVtx, EdgeType* xadj, VtxType* adj) {
  for(VtxType i = 0; i < nVtx; i++) {
    for(auto ptr = xadj[i]+1; ptr < xadj[i+1]; ptr++) {
      if(adj[ptr] <= adj[ptr-1]) {
	std::cout<<"is not sorted"<<std::endl;
	return false;
      }
    }
  }

  return true;
}

template <typename VtxType, typename EdgeType, typename WeightType>
bool ReadGraph(char *filename, VtxType *numofvertex, EdgeType **pxadj, VtxType **padjncy, VtxType** ptadj, WeightType **padjncyw, WeightType **ppvw, long** reverse_map) {
	FILE *fpin = fopen(filename, "r");
	
	if (fpin == NULL) {
	  std::cerr<<"can't open "<<filename<<std::endl;
	  return false;
	}


	std::string filestr (filename);

	std::string graphsuf = ".graph";
	std::string mtxsuf = ".mtx";
	std::string binsuf = ".bin";

	if (has_suffix(filestr, graphsuf))
	  ReadGraphFromFile<VtxType, EdgeType, WeightType> (fpin, numofvertex, pxadj, padjncy, padjncyw, ppvw);
	else if (has_suffix(filestr, mtxsuf))
		ReadGraphFromMMFile (fpin, numofvertex, pxadj, padjncy, padjncyw, ppvw, false, reverse_map);
	else if (has_suffix(filestr, binsuf))
	  ReadBinary<VtxType, EdgeType, WeightType> (filename, numofvertex, numofvertex, pxadj, padjncy, NULL, NULL);
	else
		ReadGraphFromMMFile (fpin, numofvertex, pxadj, padjncy, padjncyw, ppvw, true, reverse_map);


	
	bool sorted = isGraphSorted<VtxType, EdgeType>(*numofvertex, *pxadj, *padjncy);

	*ptadj = nullptr;

	//This code makes no sense!
	// //Is the goal of the rest to generate the transpose graph?
	// //with an assumption that the graph is symmetric?
	// VtxType* adj = *padjncy;
	// EdgeType* xadj = *pxadj;
	// VtxType n = *numofvertex;
	// *ptadj = (VtxType*) malloc(sizeof(VtxType) * xadj[n]);
	// VtxType* tadj = *ptadj;

	// EdgeType* degs = (EdgeType*)malloc(sizeof(EdgeType) * n);
	// VtxType* myedges = (VtxType*)malloc(sizeof(VtxType) * xadj[n]);

	// std::copy (xadj, xadj+n, degs); //memcpy(degs, xadj, sizeof(VtxType) * n);
	
	// for(VtxType i = 0; i < n; i++) {
	// 	for(auto ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
	// 		VtxType j = adj[ptr];
	// 		myedges[degs[j]++] = i;
	// 	}
	// }

	// for(VtxType i = 0; i < n; i++) {
	// 	if(xadj[i+1] != degs[i]) {
	// 	  std::cout<<"something is wrong, "<<i<<" "<< xadj[i+1] << " "<< degs[i]<<std::endl; //graph is not symetric
	// 	  exit(1);
	// 	}
	// }


	// std::copy(xadj, xadj+n, degs); // memcpy(degs, xadj, sizeof(VtxType) * n);
	// for(VtxType i = 0; i < n; i++) {
	// 	for(auto ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
	// 		VtxType j = adj[ptr];
	// 		if(i < j) {
	// 			tadj[ptr] = degs[j];
	// 			tadj[degs[j]++] = ptr;
	// 		}
	// 	}
	// }

	// free(degs);
	// free(myedges);

	// for(VtxType i = 0; i < n; i++) {
	// 	for(auto ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
	// 		VtxType j = adj[ptr];
	// 		if((adj[tadj[ptr]] != i) || (tadj[ptr] < xadj[j]) || (tadj[ptr] >= xadj[j+1])) {
	// 		  std::cout<<"error i "<<i<<" j "<<j<<" ptr "<<ptr<<std::endl;
	// 		  std::cout<<"error  xadj[j] "<< xadj[j]<<" xadj[j+1] "<<xadj[j+1]<<std::endl;
	// 		  std::cout<<"error tadj[ptr] "<<tadj[ptr]<<std::endl;
	// 		  std::cout<<"error adj[tadj[ptr]] "<< adj[tadj[ptr]]<<std::endl;
	// 			exit(1);
	// 		}
	// 	}
	// }

	fclose(fpin);

	return true;
}

//explicit instanciation of commonly used variants

template bool ReadGraph<int, int, int> (char *filename, int *numofvertex, int **pxadj, int **padjncy, int** ptadj, int **padjncyw, int **ppvw, long** reverse_map);

template bool ReadGraph<int, int, float> (char *filename, int *numofvertex, int **pxadj, int **padjncy, int** ptadj, float **padjncyw, float **ppvw, long** reverse_map);


template 
bool ReadGraph<long int, long int, int>(char *filename, long int *numofvertex, long int **pxadj, long int **padjncy, long int** ptadj, int **padjncyw, int **ppvw, long** reverse_map);

template
bool ReadGraph<int, long int, int> (char *filename, int *numofvertex, long int **pxadj, int **padjncy, int** ptadj, int **padjncyw, int **ppvw, long** reverse_map);
