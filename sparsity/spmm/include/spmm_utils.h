#ifndef _SPMM_UTILS_H
#define _SPMM_UTILS_H

/*
 * Generate a dense matrix with M rows and N columns in column-major order. The matrix
 * will be filled with random single-precision floating-point values between 0
 * and upper_bound(default is 100.0).
 */
void generate_random_dense_matrix(int M, int N, float **out_mat, float upper_bound=100.0);

/*
 * Generate a sparse matrix with M rows and N columns in CSR format with 
 * sparse_radio(default=0.9). The matrix will be filled with random 
 * single-precision floating-point values between 0
 * and upper_bound(default=100.0).
 */
void generate_random_csr_matrix(int M, int N, 
    int *out_nnz, int **out_csrOffsets, int **out_columns, float **out_values, 
    float upper_bound=100.0, float sparse_radio=0.9);

#endif // _SPMM_UTILS_H