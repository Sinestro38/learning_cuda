#include <iostream>
#define N 512

//
__global__ void mat_mul (double* res, double* mat1, double* mat2, int n)
{
    int bid_x = blockIdx.x;
    int tid_x = threadIdx.x;

    for (int row_index{0}; row_index < n; row_index++) {
        for (int col_index{0}; col_index < n; col_index++) {
            if (bid_x == row_index && tid_x == col_index) {
                int index = row_index*n + col_index;
                res[index] = 0;
                // compute dot product
                for (int k{0}; k < n; k++) {
                    int dot_index_1 = row_index*n + k;
                    int dot_index_2 = k*n + col_index;
                    res[index] += mat1[dot_index_1] * mat2[dot_index_2];
                }
            }
        }
    }    
}

int main()
{
    double* arr;
    double* brr;
    double* res;
    double* correct_res;

    cudaMallocManaged( &arr, (N*N) * sizeof(double) );
    cudaMallocManaged( &brr, (N*N) * sizeof(double) );
    cudaMallocManaged( &res, (N*N) * sizeof(double) );
    cudaMallocManaged( &correct_res, (N*N) * sizeof(double) );
    
    // -- INITIALIZE --
    // Note that arr[i][j] is the same as *(*(arr+i)+j)
    for (int i{}; i<N; i++)
    {
        arr[i*N +(N-1-i)] = 1;
        brr[i*N +(N-1-i)] = 1;
        correct_res[i*N +(i)] = 1;
    }
    // -- END INITIALIZE --
            
    // Execute kernel
    mat_mul<<<N,N>>>(res, arr, brr, N);
    cudaDeviceSynchronize();
    
    // -- BEGIN TEST -- 
    bool correct{true};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int index = i*N +j;
            if (res[index] != correct_res[index]) {
                correct = false;
                break;
            }
        }
    }
    if (correct)
    printf("example PASSED\n");
    else
        printf("example FAILED: wrong result\n"); 
    // -- END TEST -- 
        
    // Deallocate unified memory so must free each double pointer array in each matrix
    cudaFree( arr );
    cudaFree( brr );
    cudaFree( res );
    cudaFree( correct_res );
    return 0;
}