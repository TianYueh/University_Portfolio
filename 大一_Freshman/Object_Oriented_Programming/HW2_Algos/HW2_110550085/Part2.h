#ifndef PART2_H
#include "SolverBase.h"

class PartII : public SolverBase
{
public:
    int n, m;
    void read(std::string);
    void solve();
    void write(std::string);
    int** adjmat;
    int* distance;
    bool* visit;
    int dans;
    int* parent;
    bool negativeloop = false;
    int bans;
    int** edgelist;
};

#define PART2_H
#endif