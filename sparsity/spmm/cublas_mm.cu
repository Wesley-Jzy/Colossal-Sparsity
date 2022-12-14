#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "include/common.h"
#include "include/spmm_utils.h"

/*
 * A simple example of performing matrix-vector multiplication using the cuBLAS
 * library and some randomly generated inputs.
 */

/*
 * M = # of rows
 * N = # of columns
 */
int M = 1024;
int N = 1024;

int main(int argc, char **argv)
{
    int i, j;
    float *A, *dA;
    float *B, *dB;
    float *C, *dC;
    float beta;
    float alpha;
    cublasHandle_t handle = 0;

    alpha = 3.0f;
    beta = 4.0f;

    // Generate inputs
    srand(10086);
    generate_random_dense_matrix(M, N, &A);
    generate_random_dense_matrix(N, M, &B);
    C = (float *)malloc(sizeof(float) * M * M);
    memset(C, 0x00, sizeof(float) * M * M);

    // Create the cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&handle));

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK_CUDA(cudaMalloc((void **)&dB, sizeof(float) * N * M));
    CHECK_CUDA(cudaMalloc((void **)&dC, sizeof(float) * M * M));

    // Transfer inputs to the device
    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(float), A, M, dA, M));
    CHECK_CUBLAS(cublasSetMatrix(N, M, sizeof(float), B, N, dB, N));
    CHECK_CUBLAS(cublasSetMatrix(M, M, sizeof(float), C, M, dC, M));

    // Execute the matrix-vector multiplication
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, N, &alpha,
                dA, M, dB, N, &beta, dC, M));

    // Retrieve the output vector from the device
    CHECK_CUBLAS(cublasGetMatrix(M, M, sizeof(float), dC, M, C, M));

    for (j = 0; j < 10; j++)
    {
        for (i = 0; i < 10; i++)
        {
            printf("%2.2f ", C[j * M + i]);
        }
        printf("...\n");
    }

    printf("...\n");

    free(A);
    free(B);
    free(C);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}