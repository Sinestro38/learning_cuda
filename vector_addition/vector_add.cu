#include <stdio.h>
#define N 10000000

// OLD SINGLE-THREADED VERSION
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = 0;
    int stride = 1;
    for(int i = index; i < n; i += stride){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; 

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
        
    // Main function
    vector_add<<<1, 512>>>(d_out, d_a, d_b, N);

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