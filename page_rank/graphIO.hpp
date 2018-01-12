#ifndef GRAPH_IO_HPP
#define GRAPH_IO_HPP

#include <stdio.h>

template <typename VtxType, typename EdgeType, typename WeightType>
bool ReadGraph(char *filename, VtxType *numofvertex, EdgeType **pxadj, VtxType **padjncy, VtxType** ptadj, WeightType **padjncyw, WeightType **ppvw, long** reverse_map);

#endif
