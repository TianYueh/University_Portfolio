#ifndef PART1_H
#include "SolverBase.h"
#include <vector>
#include <map>

using namespace std;

class PartI : public SolverBase
{
int** graph;
int** rev_graph;
int** coarseGraph;
int** scc;
int* visit;
int* order;
int* predecessor;
int* finishOrder;
int n, m;
int Root;
int time = 0, current = 0;
int timeforSCC = 0;
int numofSCC = 0;
int line=0;
int num_cc = 0;

map<int, int> finish;
bool isAcyclic = true;

public:
    void read(std::string);
    void solve();
    void write(std::string);
    void DFS(int);
    void DFSSCC(int,int);
};

#define PART1_H
#endif