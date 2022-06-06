#include <iostream>
#define N 2 

//
__global__ void mat_mul (double** res, double** mat1, double** mat2, int n)
{
    // int index = threadIdx.x;
    // int index = threadIdx.y;

    for (int row_index{0}; row_index < n; row_index++) {
        for (int col_index{0}; col_index < n; col_index++) {
            res[row_index][col_index] = 0;
            // compute dot product
            for (int k{0}; k < n; k++) {
                res[row_index][col_index] += mat1[row_index][k] * mat2[k][col_index];
            }
        }
    }    
}

int main()
{

    double* arr[N]; // an array containing double pointers
    double* brr[N];
    double* res[N];
    double* correct_res[N];
    // allocate memory for each row
    for (int row_num{}; row_num < N; row_num++) {
        arr[row_num] = (double*) malloc( sizeof(double) * N );
        brr[row_num] = (double*) malloc( sizeof(double) * N );
        res[row_num] = (double*) malloc( sizeof(double) * N );
        correct_res[row_num] = (double*) malloc( sizeof(double) * N );
    }
    
    // -- INITIALIZE --
    // Note that arr[i][j] is the same as *(*(arr+i)+j)
    for (int i{}; i<N; i++)
    {
        arr[i][N-1-i] = 1;
        brr[i][N-1-i] = 1;
        correct_res[i][i] = 1;
    }
    // -- END INITIALIZE --
    
    
    
    // -- TRANSFER MATRIX FROM HOST TO DEVICE --
    
    // used to store cuda allocated memory addresses on host
    double** d_ah = (double**) malloc( sizeof(double*) *N);
    double** d_bh = (double**) malloc( sizeof(double*) *N);
    double** d_resh = (double**) malloc( sizeof(double*) *N);
    
    // Transfer array memory from host to device
    for ( int i{0}; i<N; i++ ) {
        // for every double pointer in d_a, allocate N [double] bytes
        // of contiguous memory to store the row of doubles, here we
        // pass the memory location where the memory address is stored
        // since we wish to alter the state of which memory address is 
        // being stored
        cudaMalloc( (void**) &(d_ah[i]), sizeof(double)*N );
        cudaMalloc( (void**)&d_bh[i], sizeof(double)*N );
        cudaMalloc( (void**)&d_resh[i], sizeof(double)*N );
        cudaMemcpy( d_ah[i] /* memory address on gpu */, 
            arr[i] /* memory address on cpu */,  
            sizeof(double)*N /* num bytes to copy from start */, 
            cudaMemcpyHostToDevice /* direction of copy */
        );
        cudaMemcpy( d_bh[i], brr[i], sizeof(double)*N, cudaMemcpyHostToDevice );
    }
    // Transfer cuda memory addresses stored in host to device 
    double** d_a; // contains memory addresses 
    double** d_b;
    double** d_res;
    cudaMalloc( (void***)&d_a, sizeof(double*)*N );
    cudaMalloc( (void***)&d_b, sizeof(double*)*N );
    cudaMalloc( (void***)&d_res, sizeof(double*)*N );
    
    cudaMemcpy( d_a, d_ah, sizeof(double*)*N, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, d_bh, sizeof(double*)*N, cudaMemcpyHostToDevice);
    cudaMemcpy( d_res, d_resh, sizeof(double*)*N, cudaMemcpyHostToDevice);
    
    // -- END TRANSFER MATRIX FROM HOST TO DEVICE --
    
    mat_mul<<<1,1>>>(d_res, d_a, d_b, N);
    
    // Transfer array result from device to host
    cudaMemcpy(d_resh, d_res, sizeof(double*)*N, cudaMemcpyDeviceToHost);
    for ( int i{0}; i<N; i++ ) {
        cudaMemcpy(res[i], d_resh[i], sizeof(double) *N, cudaMemcpyDeviceToHost);
    }
    
    
    // --- PRINT MATRIX ON HOST ---
    // for (int i=0; i < N; i++)
    // {
    //         for (int j{0}; j<N; j++) std::cout << arr[i][j];
        
    //         std::cout << std::endl;
    //     }
    // --- END PRINT MATRIX ON HOST ---
    
    
    // -- BEGIN TEST -- 
    bool correct{true};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << res[i][j];
            if (res[i][j] != correct_res[i][j]) {
                correct = false;
                // break;
            }
        }
        std::cout << std::endl;
    }
    if (correct)
    printf("example PASSED\n");
    else
        printf("example FAILED: wrong result\n"); 
    // -- END TEST -- 
        
    // Deallocate cuda memory
    cudaFree( d_a );
    cudaFree( d_b );
    cudaFree( d_res );
    // Deallocate host memory so must free each double pointer array in each matrix
    for (int row_i=0; row_i < N; row_i++)
    {   
        free( arr[row_i] );
        free( brr[row_i] );
        free( res[row_i] );
        free( correct_res[row_i] );
    }
    return 0;
}