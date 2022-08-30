#include <stdlib.h>
#include <algorithm>
#include "include/spmm_utils.h"

void generate_random_dense_matrix(int M, int N, float **out_mat, float upper_bound)
{
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);

    // For each column
    for (int j = 0; j < N; j++)
    {
        // For each row
        for (int i = 0; i < M; i++)
        {
            A[j * M + i] = ((double)rand() / rMax) * upper_bound;
        }
    }

    *out_mat = A;
}

void generate_random_csr_matrix(int M, int N, 
    int *out_nnz, int **out_csrOffsets, int **out_columns, float **out_values, 
    float upper_bound, float sparse_radio) 
{   
    double rMax = (double)RAND_MAX;
    int nnz = int((1.0 - sparse_radio) * M * N);
    int *csrOffsets = (int *)malloc(sizeof(int) * (M + 1));
    int *columns = (int *)malloc(sizeof(int) * nnz);
    float *values = (float *)malloc(sizeof(float) * nnz);

    csrOffsets[0] = 0;
    csrOffsets[M] = nnz;
    for (int i = 1; i < M; ++i) {
        csrOffsets[i] = rand() % (nnz + 1);
    }
    std::sort(csrOffsets + 1, csrOffsets + M - 1);

    for (int i = 0; i < M; ++i) {
        int offset = csrOffsets[i];
        int len = csrOffsets[i+1] - csrOffsets[i];
        for (int j = offset; j < offset + len; ++j) {
            columns[j] = rand() % N;
            std::sort(columns + j, columns + j + len - 1);
            values[j] = ((double)rand() / rMax) * upper_bound;
        }
    }

    *out_nnz = nnz;
    *out_csrOffsets = csrOffsets;
    *out_columns = columns;
    *out_values = values;
}