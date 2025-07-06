#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "vector.h"
#include "matrix.h"


int main(void) {
	while (1) {
		int A_row, A_column, B_row, B_column, A_number, B_number;
		scanf("%d%d%d%d%d%d", &A_row, &A_column, &B_row, &B_column, &A_number, &B_number);
		matrix2D A;
		matrix2D B;
		A = *matrix2D_construct(A_row, A_column);
		B = *matrix2D_construct(B_row, B_column);
		matrix2D_fillwith(&A, A_number);
		matrix2D_fillwith(&B, B_number);
		matrix2D answer = *matrix2D_multiple(&A, &B);
		if (answer.row != 0) {
			for (int i = 0; i < answer.row; i++) {
				for (int j = 0; j < answer.column; j++) {
					printf("%d ", answer.mat[i].vec[j]);
				}
				printf("\n");
			}
		}
	}
}