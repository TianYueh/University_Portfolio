#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "vector.h"
#include "matrix.h"

vector* vector_construct(int length){
	vector* a=(vector*)calloc(1,sizeof(vector));
	a->length = length;
	a->vec = (int*)calloc(length, sizeof(int));
	return a;
}