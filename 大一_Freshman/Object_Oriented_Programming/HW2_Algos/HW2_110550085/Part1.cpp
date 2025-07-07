#include "Part1.h"
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;

void PartI::read(std::string file) {
	std::cout << "Part 1 reading..." << std::endl;
	fstream F;
	F.open(file, ios::in);
	F >> n >> m;

	graph = (int**)calloc(n, sizeof(int*));
	rev_graph = (int**)calloc(n, sizeof(int*));
	scc = (int**)calloc(n, sizeof(int*));
	visit = (int*)calloc(n, sizeof(int));
	order = (int*)calloc(n, sizeof(int));
	finishOrder = (int*)calloc(n, sizeof(int));
	predecessor = (int*)calloc(n, sizeof(int));



	for (int i = 0; i < n; i++) {
		visit[i] = 0;
		predecessor[i] = -1;
		finishOrder[i] = -1;
	}
	for (int i = 0; i < n; i++) {
		graph[i] = (int*)calloc(n, sizeof(int));
		rev_graph[i] = (int*)calloc(n, sizeof(int));
		scc[i] = (int*)calloc(n, sizeof(int));
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			graph[i][j] = 0;
			rev_graph[j][i] = 0;
			scc[i][j] = -1;
		}
	}

	for (int i = 0; i < m; i++) {
		int start, end, useless;
		F >> start >> end >> useless;
		graph[start][end] = 1;
		rev_graph[end][start] = 1;
	}

}
void PartI::solve() {
	std::cout << "Part 1 solving..." << std::endl;
	//DFS
	for (int i = 0; i < n; i++) {
		if (visit[i] == 0) {
			DFS(i);
		}
	}
	if (!isAcyclic) {
		
		int num = 0;
		int ordered = 0;
		while (ordered != n) {
			num++;
			for (int i = 0; i < n; i++) {
				if (finish[i] == num) {
					finishOrder[ordered] = i;
					ordered++;
				}
			}
		}
		reverse(finishOrder, finishOrder + n);
		for (int i = 0; i < n; i++) {
			visit[i] = 0;
			finish[i] = 0;
			predecessor[i] = -1;
		}
		timeforSCC = 0;
		for (int i = 0; i < n; i++) {
			if (visit[finishOrder[i]] == 0) {
				DFSSCC(finishOrder[i], finishOrder[i]);
			}
		}
		for (int i = 0; i < n; i++) {
			if (predecessor[i] == -1) {
				num_cc++;
				scc[num_cc - 1][0] = i;
				int jetzt = 1;
				for (int j = 0; j < n; j++) {
					if (predecessor[j] == i) {
						scc[num_cc - 1][jetzt] = j;
						jetzt++;
					}
				}
			}
		}
		coarseGraph = (int**)calloc(num_cc, sizeof(int*));
		for (int i = 0; i < num_cc; i++) {
			coarseGraph[i] = (int*)calloc(n, sizeof(int));
		}
		for (int i = 0; i < num_cc; i++) {
			for (int j = 0; j < n; j++) {
				coarseGraph[i][j] = 0;
			}
		}
		int found = 0;
		for (int i = 0; i < num_cc; i++) {
			for (int j = 0; j < n; j++) {
				if (scc[i][j] == -1) {
					break;
				}
				else {
					for (int k = 0; k < n; k++) {
						if (graph[scc[i][j]][k] == 1) {
							for (int x = 0; x < num_cc; x++) {
								for (int y = 0; y < n; y++) {
									if (scc[x][y] == k) {
										found = 1;
										coarseGraph[i][x]++;
										break;
									}
								}
								if (found == 1) {
									found = 0;
									break;
								}
							}
						}
					}
				}
			}
		}
		for (int i = 0; i < num_cc; i++) {
			for (int j = 0; j < n; j++) {
				if (coarseGraph[i][j] != 0) {
					if (i != j) {
						line++;
					}
				}
			}
		}
	}
}
void PartI::write(std::string file){
	std::cout << "Part 1 writing..." << std::endl;
	fstream fout;
	fout.open(file, ios::out);
	if (isAcyclic == true) {
		for (int i = n - 1; i >= 0; i--) {
			fout << order[i] << " ";
		}
		fout << endl;
	}
	else {
		fout << num_cc << " " << line << endl;
		for (int i = 0; i < num_cc; i++) {
			for (int j = 0; j < n; j++) {
				if (coarseGraph[i][j] != 0) {
					if (i != j) {
						fout << i << " " << j << " " << coarseGraph[i][j] << endl;
					}
				}
			}
		}
	}
}

void PartI::DFS(int now) {
	if (visit[now] == 1) {
		isAcyclic = false;
		return;
	}
	if (visit[now] == 2) {
		return;
	}
	visit[now] = 1;
	timeforSCC++;
	for (int x = 0; x < n; x++) {
		if (graph[now][x] == 1) {
			DFS(x);
		}
	}
	visit[now] = 2;
	order[time] = now;
	finish[now] = timeforSCC;
	time++;
	timeforSCC++;
}

void PartI::DFSSCC(int now, int ancestor) {
	if (visit[now]) {
		return;
	}
	visit[now] = 1;
	predecessor[now] = ancestor;
	if (now == ancestor) {
		predecessor[now] = -1;
	}
	for (int x = 0; x < n; x++) {
		if (rev_graph[now][x] == 1) {
			DFSSCC(x,ancestor);
		}
	}
	visit[now] = 2;
	finish[now] = timeforSCC;
	timeforSCC++;
}

