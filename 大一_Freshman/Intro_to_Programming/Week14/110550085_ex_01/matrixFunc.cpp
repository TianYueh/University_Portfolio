#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "vector.h"
#include "matrix.h"

// to construct the matrix by given row and column
matrix2D* matrix2D_construct(int m, int n) {
	matrix2D* a=(matrix2D*)calloc(1,sizeof(matrix2D));
	a->row = m;
	a->column = n;
	a->mat = (vector*)calloc(n, sizeof(vector)*1);
	for (int i = 0; i < m; i++) {
		a->mat[i] = *vector_construct(n);
	}
	return a;
}

// fill the matrix with specific number
void matrix2D_fillwith(matrix2D* A, int num) {
	for (int i = 0; i < A->row; i++) {
		for (int j = 0; j < A->column; j++) {
			A->mat[i].vec[j] = num;
		}
	}
}

// check 2 matrix with the right shape, if they can multiple together, return the result, or print shape error
matrix2D* matrix2D_multiple(matrix2D* A, matrix2D* B) {
	matrix2D* ans=(matrix2D*)calloc(1,sizeof(matrix2D));
	if(A->column==B->row) {
		ans = matrix2D_construct(A->row, B->column);
		for (int i = 0; i < A->row; i++) {
			for (int j = 0; j < B->column; j++) {
				int temp = 0;
				for (int k = 0; k < A->column; k++) {
					temp =temp+(A->mat[i].vec[k]* B->mat[k].vec[j]);
				}
				ans->mat[i].vec[j] = temp;
			}
		}
	}
	else{
		printf("shape error\n");
		ans->row = 0;
		ans->column = 0;
	}
	return ans;
}