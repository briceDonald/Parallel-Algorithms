#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "stdlib.h"
#include <math.h>

using namespace std;
#define THREADS_PER_BLOCK (1 << 6)
#define B_ARRAY_SIZE 10

/************************************** DEVICE SIDE ****************************************/
__global__ void  par_global_compute_count( int *d_in, int *d_out, int sz )
{
	// get thread index
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if( tid < sz )
	{
		int targetIdx = d_in[tid] / 100;
		atomicAdd( &d_out[targetIdx], 1 );
	}
}

__global__ void  par_shared_compute_count( int *d_in, int *d_out, int sz )
{
	// get thread index
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int bTid = threadIdx.x;
	__shared__ int tpB[B_ARRAY_SIZE];

	// Each thread in range [0, 9 within a block will zero the tpB array
	if( bTid < B_ARRAY_SIZE )
		tpB[bTid] = 0;

	if( tid < sz )
	{
		// Each thread within a block atomically increments it's target slot locally.
		// low contention
		int targetIdx = d_in[tid] / 100;
		atomicAdd( &tpB[targetIdx], 1 );
		__syncthreads();

		// The first ten threads within each block atomically updates the global out array. 
		// low contention.
		if( bTid < B_ARRAY_SIZE )
			atomicAdd( &d_out[bTid], tpB[bTid] ); 
	}
}

__global__ void par_hillis_steele_sum( int *d_in, int *d_out, int sz  )
{
	// get thread index
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int h = (int) ceil( log2( float(sz) ) );
	__shared__ int tpOut[B_ARRAY_SIZE];

	if( tid < sz )
	{
		int val;
		tpOut[tid] = d_in[tid];

		for(int i = 0; i < h; i++)
		{
			int step = (int) pow(2.0f, i);
			if( tid-step < 0 )
				break;

			val = tpOut[tid-step];
			__syncthreads();
			tpOut[tid] += val;
			__syncthreads();
		}

		d_out[tid] = tpOut[tid];
	}
}


/********************************* HOST CODE ************************************/
void printProperties( int device )
{
	struct cudaDeviceProp prop;

	cout << "\nBRICE DONALD NGNIGHA" << endl;
	cout << "PROB 2.\n" << endl;

	if( cudaGetDeviceProperties( &prop, device ) )
	{
		cout << "error: Failed to get device properties." << endl;
		return;
	}
	
	cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
	cout << "Max Threads per dims x: " << prop.maxThreadsDim[0] << " y: "<< prop.maxThreadsDim[1] << " z: " << prop.maxThreadsDim[2] << endl;
	cout << "Shared memory per block " << prop.sharedMemPerBlock << endl;
	cout << "Registers per block " << prop.regsPerBlock << endl;
	cout << "Max grid size x: " << prop.maxGridSize[0] << " y: "<< prop.maxGridSize[1] << " z: " << prop.maxGridSize[2] << endl;
	cout << "\n\n" << endl;
}

int read_csv( vector<int> &dataVec, const char *filename )
{
    int length = 0;
    string value;
    ifstream datafile;

    cout << "Opening file : " << filename << endl;

    try
    {
		datafile.open( filename );

		while( datafile.good() )
		{
			getline( datafile, value, ',' );
			dataVec.push_back(  atoi( value.c_str() ) );
			length++;
		}

		datafile.close();
	}
	catch(...)
	{
		cout << "Error opnening file: " << value << endl;
	}

    return length;
}

void sequential_compute_count( int *A, int *B, int len )
{
	if( !B || !A )
		return;

	memset( B, 0, B_ARRAY_SIZE*sizeof(int) );

	for(int i = 0; i < len; i++)
		B[ (A[i]/100) ] += 1;
}

void parallel_compute_count_global(int * A, int *B, int len )
{
	int *d_in, *d_out; 
	int *h_in = A, *h_out = B;

	cudaMalloc( (void**) &d_in, len*sizeof(int) ); 
	cudaMalloc( (void**) &d_out, B_ARRAY_SIZE*sizeof(int) );
	
	cudaMemcpy( d_in, h_in, len*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset( d_out, 0, B_ARRAY_SIZE*sizeof(int) );

	int numBlocks = (int) ceil( (1.0f * len)/THREADS_PER_BLOCK );
	par_global_compute_count<<<numBlocks, THREADS_PER_BLOCK>>>( d_in, d_out, len );

	cudaMemcpy( h_out, d_out, B_ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree( d_in );
	cudaFree( d_out );
}

void parallel_compute_count_shared(int * A, int *B, int len )
{
	int *d_in, *d_out; 
	int *h_in = A, *h_out = B;

	cudaMalloc( (void**) &d_in, len*sizeof(int) ); 
	cudaMalloc( (void**) &d_out, B_ARRAY_SIZE*sizeof(int) );
	
	cudaMemcpy( d_in, h_in, len*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset( d_out, 0, B_ARRAY_SIZE*sizeof(int) );

	int numBlocks = (int) ceil( (1.0f * len)/THREADS_PER_BLOCK );
	par_shared_compute_count<<<numBlocks, THREADS_PER_BLOCK>>>( d_in, d_out, len );

	cudaMemcpy( h_out, d_out, B_ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree( d_in );
	cudaFree( d_out );
}

void sequential_hellis_steele_sum( int *A, int *B, int len)
{
	for(int i = 0; i < len; i++)
		if(i == 0)
			B[i] = A[i];
		else
			B[i] = B[i-1] + A[i];
}

void parallel_hillis_steele_sum(int *A, int *B, int len)
{
	int *d_in, *d_out; 
	int *h_in = A, *h_out = B;

	cudaMalloc( (void**) &d_in, len*sizeof(int) ); 
	cudaMalloc( (void**) &d_out,len* sizeof(int) );
	
	cudaMemcpy( d_in, h_in, len*sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = (int) ceil( (1.0f * len)/THREADS_PER_BLOCK );
	par_hillis_steele_sum<<<numBlocks, THREADS_PER_BLOCK>>>( d_in, d_out, len );

	cudaMemcpy( h_out, d_out, len*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree( d_in );
	cudaFree( d_out );
}

int main( int argc, char **argv )
{
	int count = 0;
	cudaGetDeviceCount(&count);
	
	int devId = -1;
	cudaGetDevice(&devId);
	printProperties(devId);

	bool err1=false, err2=false, err3=false;
	vector<int> dataVec;
	int len = read_csv( dataVec, "inp.txt" );
	int *A = &dataVec[0];

    int seqCounts[B_ARRAY_SIZE];
	int parGlobalCounts[B_ARRAY_SIZE];
	int parSharedCounts[B_ARRAY_SIZE];

	sequential_compute_count( A, seqCounts, len );

	cout << "Prob. 2-a -- USING GLOBAL ARRAY IN DEVICE FOR COMPUTING COUNTS --" << endl;
	cout << "2-a => \t" << "Par" << "\t-vs-\t" << "Seq" << endl;   
	parallel_compute_count_global( A, parGlobalCounts, len );
	for(int i = 0; i < B_ARRAY_SIZE; i ++)
	{
		cout << "2-a => \t" << parGlobalCounts[i] << "\t-vs-\t" << seqCounts[i] << endl;
		if(parGlobalCounts[i] != seqCounts[i])
		{
			cout <<"Error 2-a: " << "Output mismatch on count."<<endl;
			err1 = true;
			break;
		}
	}

	cout << "\nProb. 2-b -- USING SHARED ARRAY IN DEVICE FOR COMPUTING COUNTS --" << endl;
	cout << "2-b => \t" << "Par" << "\t-vs-\t" << "Seq" << endl;   
	parallel_compute_count_shared( A, parSharedCounts, len );
	for(int i = 0; i < B_ARRAY_SIZE; i ++)
	{
		cout << "2-b => \t" << parSharedCounts[i] << "\t-vs-\t" << seqCounts[i] << endl;
		if(parSharedCounts[i] != seqCounts[i])
		{
			cout <<"Error 2-b: " << "Output mismatch on count."<<endl;
			err2 = true;
			break;
		}
	}
	
	int parB[B_ARRAY_SIZE];
	int seqB[B_ARRAY_SIZE];
	sequential_hellis_steele_sum( seqCounts, seqB, B_ARRAY_SIZE );
	parallel_hillis_steele_sum( seqCounts, parB, B_ARRAY_SIZE );

	cout << "\nProb. 2-c -- COMPUTING THE EXCLUSIVE SUM- HILLIS/STEELE --" << endl;
	cout << "2-c => \t" << "Par" << "\t-vs-\t" << "Seq" << endl;   
	for(int i = 0; i < B_ARRAY_SIZE; i ++)
	{
		cout << "2-c => \t" << parB[i] << "\t-vs-\t" << seqB[i] << endl;   
		if(parB[i] != seqB[i])
		{
			cout <<"Error 2-c: " << "Output mismatch on hillis/steele sum."<<endl;
			err3 = true;
			break;
		}
	}
	
	if( !err1 && !err2 && !err3 )
		cout << "Success" << endl;

	return 0;
}
