# QtWithCuda
#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 
#include "curand_kernel.h" 

#include <stdio.h> 
#include <time.h> 

#define N (1024*1024) 
#define M (200000) 
#define THREADS_PER_BLOCK 1024 

__global__ void vector_add(double *a, double *b, double *c) 
{ 
	int index = blockIdx.x * blockDim.x + threadIdx.x; 
	if (index == 0) 
	{ 
		printf("In vector_add.\n"); 
	} 
	for(int j=0;j<M;j++) 
	{ 
		c[index] = a[index]*a[index] + b[index]*b[index]; 
	} 
} 

int main() 
{  
 	double *a, *b, *c; 
 	int size = N * sizeof( double ); 
 
 	a = (double *)malloc( size ); 
 	b = (double *)malloc( size ); 
 	c = (double *)malloc( size ); 
 
 	for( int i = 0; i < N; i++ ) 
 	{ 
 		a[i] = b[i] = i; 
 		c[i] = 0; 
 	} 

 	double *d_a, *d_b, *d_c;  
 	cudaMalloc( (void **) &d_a, size ); 
 	cudaMalloc( (void **) &d_b, size ); 
 	cudaMalloc( (void **) &d_c, size );  
 
 	cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice ); 
 	cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice ); 
 
 	for(int i = 0; i < 10; i++) 
 	{
 		clock_t start = clock();
 		vector_add<<< (N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c ); 
 		cudaDeviceSynchronize(); 	
 		clock_t end = clock(); 
 		float time = ((float)(end-start))/CLOCKS_PER_SEC; 	
 		printf("%f\n", time);
 	}
 
 	cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost ); 
 
 	printf( "c[%d] = %f\n",0,c[0] ); 
 	printf( "c[%d] = %f\n",N-1, c[N-1] );  
 
 	free(a); 
 	free(b); 
 	free(c); 
 	cudaFree( d_a ); 
 	cudaFree( d_b ); 
 	cudaFree( d_c ); 

 	system("pause"); 

	return 0; 
} 
