#include <stdio.h>
#define N 10000000

// NEW MULTI-THREADED VERSION
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Handling case where there are leftover threads above num of elements
    if (tid < n){
        out[tid] = a[tid] + b[tid];
    }
}

// __global__ void get_num_blocks(int* num_blocks) {
//     *num_blocks = gridDim.x;
// }

int main(){
    // get num blocks in GPU
    // int* d_numblocks = 0;
    // int* num_blocks = 0;
    // cudaMalloc( (void**)&d_numblocks, sizeof(int) );
    // get_num_blocks<<<1, 1>>>(d_numblocks);
    // cudaMemcpy(num_blocks, d_numblocks, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Number of blocks on GPU: %.3f \n", *num_blocks);    

    float *a, *b, *out; 
    // printf("Total num of blocks on GPU: %.3f, ", gridDim.x);

    // Allocate memory
    a   = (float*) malloc(sizeof(float) * N);
    b   = (float*) malloc(sizeof(float) * N);
    out = (float*) malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Print first 10 elements of out
    for (int i = 0; i < 10; i++){
        if (i == 9){
            printf("%.3f\n", out[i]);
            continue;
        }
        printf("%.3f, ", out[i]);
    }

    // Allocate device memory
    float* d_a;
    float* d_b;
    float* d_out;
    cudaMalloc( (void**)&d_a, sizeof(float)*N );
    cudaMalloc( (void**)&d_b, sizeof(float)*N );
    cudaMalloc( (void**)&d_out, sizeof(float)*N );
    
    // Transfer array memory from host to device
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
        
    // Executing kernel
    int block_size = 256;
    int grid_size = ((N + block_size) / block_size); // ensuring enough threads
    vector_add<<<grid_size, block_size>>>(d_out, d_a, d_b, N);
    
    // Transfer array memory from device to host
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    // Print first 10 elements of out
    for (int i = 0; i < 10; i++){
        if (i == 9){
            printf("%.3f\n", out[i]);
            continue;
        }
        printf("%.3f, ", out[i]);
    }
    
    // Deallocate device memory
    cudaFree( d_a );
    cudaFree( d_b );
    cudaFree( d_out );
    
    // Deallocate host memory
    free( a );
    free( b );
    free( out );
}