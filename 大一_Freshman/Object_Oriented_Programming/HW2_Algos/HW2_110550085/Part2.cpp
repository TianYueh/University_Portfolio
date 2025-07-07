#include "Part2.h"
#include <iostream>
#include <fstream>
using namespace std;

const int inf = 999999;

void PartII::read(std::string file){
	std::cout << "Part II reading..." << std::endl;
	fstream F;
	F.open(file, ios::in);
	F >> n >> m;
	adjmat = (int**)calloc(n, sizeof(int*));
	for (int i = 0; i < n; i++) {
		adjmat[i] = (int*)calloc(n, sizeof(int));
	}
	distance = (int*)calloc(n, sizeof(int));
	visit = (bool*)calloc(n, sizeof(bool));
	parent = (int*)calloc(n, sizeof(int));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) {
				adjmat[i][j] = 0;
			}
			else {
				adjmat[i][j] = inf;
			}
		}
		visit[i] = false;
		distance[i] = inf;
	}
	edgelist = (int**)calloc(m, sizeof(int*));
	for (int i = 0; i < m; i++) {
		edgelist[i] = (int*)calloc(3, sizeof(int));
	}
	for (int i = 0; i < m; i++) {
		int start, end, weight;
		F >> start >> end >> weight;
		adjmat[start][end] = weight;
		edgelist[i][0] = start;
		edgelist[i][1] = end;
		edgelist[i][2] = weight;
	}
	distance[0] = 0;
	parent[0] = 0;
}
void PartII::solve() {
	std::cout << "Part II solving..." << std::endl;
	//Dijkstra
	for (int i = 0; i < n; i++) {
		int a = -1, b = -1, min = inf;
		for (int j = 0; j < n; j++) {
			if (!visit[j] && distance[j] < min ) {
				a = j;
				min = distance[j];
			}
		}
		if (a == -1) {
			break;
		}
		else {
			visit[a] = true;
		}
		for (b = 0; b < n; b++) {
			int val;
			if (adjmat[a][b] < 0) {
				val = -adjmat[a][b];
			}
			else {
				val = adjmat[a][b];
			}
			if (!visit[b] && distance[a] + val < distance[b]) {
				distance[b] = distance[a] + val;
				//parent[b] = a;
			}
		}
	}
	dans = distance[n-1];

	//clear
	for (int i = 0; i < n; i++) {
		distance[i] = inf;
		visit[i] = false;
	}
	distance[0] = 0;

	//Bellman-Ford
	for (int i = 0; i < n-1; i++) {
		int check = 0;
		for (int j = 0; j < m; j++) {
			if (distance[edgelist[j][0]] != inf && distance[edgelist[j][0]]+edgelist[j][2]<distance[edgelist[j][1]]) {
				distance[edgelist[j][1]] = distance[edgelist[j][0]] + edgelist[j][2];
				//parent[edgelist[j][1]] = edgelist[j][0];
				check = 1;
			}
		}
		if (!check) {
			break;
		}
	}

	//Detect the negative loop
	for (int i = 0; i < m; i++) {
		if (distance[edgelist[i][0]] + edgelist[i][2] < distance[edgelist[i][1]]) {
			negativeloop = true;
		}
	}
	bans = distance[n-1];


	
}
void PartII::write(std::string file) {
	std::cout << "Part II writing..." << std::endl;
	fstream fout;
	fout.open(file, ios::out);
	fout << dans << endl;
	if (negativeloop == true) {
		fout << "Negative loop detected!" << endl;
	}
	else {
		fout << bans << endl;
	}
} 