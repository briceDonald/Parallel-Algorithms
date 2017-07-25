#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "stdlib.h"
#include <math.h>

using namespace std;
#define THREADS_DIM_X (1 << 5)
#define B_ARRAY_SIZE 10

/************************************** DEVICE SIDE ****************************************/

__global__ void compute_predicate( const int *d_in, int *d_swap, int sz, int bit )
{
	int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if( tid < sz )
	{
		unsigned int mask = 1 << bit;
		d_swap[tid]  = d_in[tid] & mask ? 0 : 1;
	}
}

__global__ void compute_exclusive_sum( const int *d_swap, int *d_pos, int sz )
{
	int val;
	int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	int h = (int) ceil( log2( float(sz) ) );


	if( tid < sz )
	{
		d_pos[tid] = d_swap[tid];
		__syncthreads();
		
		// Compute the exclusive sum using Hillis/steele and removing self
		for(int i = 0; i < h; i++)
		{
			int step = (int) pow(2.0f, i);
			if( tid-step < 0 )
				val = 0;
			else
				val = d_pos[tid-step];
			
			__syncthreads();
			
			d_pos[tid] += val;
			__syncthreads();
		}

		d_pos[tid] -= d_swap[tid];
	}
}

__global__ void parallel_swap( int *d_in, int *d_swap, int *d_pos, int sz )
{
	// get thread index
	int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if( tid < sz )
	{
		// perform the swaps
		int rank;
		if( d_swap[tid] )
			rank = d_pos[tid];
		else
		{
			int nextRank = d_pos[sz-1] + d_swap[sz-1];
			rank = nextRank + tid - d_pos[tid];
		}

		int tp = d_in[tid];
		__syncthreads();

		d_in[rank] = tp;
		__syncthreads();

	}
}


/********************************* HOST CODE ************************************/
void printProperties( int device )
{
	struct cudaDeviceProp prop;

	cout << "BRICE DONALD NGNIGHA" << endl;
	cout << "PROB 3" << endl;

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

void quickSort( int *A, int left, int right )
{
	int i = left, j = right;
	int tmp;
	int pivot = A[ (left + right)/2 ];
 
	// partition
	while(i <= j) {
		while(A[i] < pivot) i++;
		while(A[j] > pivot) j--;
		if (i <= j) {
			tmp  = A[i];
			A[i] = A[j];
			A[j] = tmp;
			i++;
			j--;
		}
	};
 
	/* recursion */
	if (left < j)
		quickSort(A, left, j);
	if (i < right)
		quickSort(A, i, right);
}

void sequential_radix_sort( int *A, int *B, int len)
{
	memcpy( B, A, len*sizeof(int) );
	quickSort( B, 0, len-1 );
}

void parallel_radix_sort(int *A, int *B, int len)
{
	int *d_in, *d_out, *d_swap, *d_pos;
	int *h_in = A, *h_out = B;


	cudaMalloc( (void**) &d_in, len * sizeof(int) ); 
	cudaMalloc( (void**) &d_out, len * sizeof(int) );
	cudaMalloc( (void**) &d_pos, len * sizeof(int) );
	cudaMalloc( (void**) &d_swap, len * sizeof(int) );

	cudaMemcpy( d_in, h_in, len*sizeof(int), cudaMemcpyHostToDevice);

	int x = THREADS_DIM_X;
	int y = (int) ceil( float(len)/THREADS_DIM_X );
	int numBlocks = 1;
	cout << "numblocks: " << numBlocks << " threads per block " << x << " x " << y << endl;

	for(int i = 0; i < 32; i++)
	{
		compute_predicate<<<numBlocks, dim3(x, y)>>>( d_in, d_swap, len, i );
		cudaThreadSynchronize();

		compute_exclusive_sum<<<numBlocks, dim3(x,y)>>>( d_swap, d_pos, len );
		cudaThreadSynchronize();

		parallel_swap<<<numBlocks, dim3(x,y)>>>( d_in, d_swap, d_pos, len );
		cudaThreadSynchronize();
	}

	cudaMemcpy( h_out, d_in, len*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree( d_in );
	cudaFree( d_out );
	cudaFree( d_pos );
	cudaFree( d_swap );
}

int main( int argc, char **argv )
{
	int count = 0;
	cudaGetDeviceCount(&count);
	
	int devId = -1;
	cudaGetDevice(&devId);
	printProperties(devId);

	bool err = false;
	vector<int> dataVec;
	int len = read_csv( dataVec, "inp.txt" );
	int *A = &dataVec[0];
	
	int parB[len];
	int seqB[len];
	sequential_radix_sort( A, seqB, len );
	parallel_radix_sort( A, parB, len );

	cout << "\nProb. 3 -- SORTING USING RADIX SORT ALGORITHM --" << endl;
	cout << "3-- => \t" << "Par" << "\t-vs-\t" << "Seq" << endl;   
	for(int i = 0; i < len; i ++)
	{
		cout << "3-- => \t" << parB[i] << "\t-vs-\t" << seqB[i] << endl;   
		if(parB[i] != seqB[i])
		{
			cout <<"Error 3-- " << "Output mismatch on radix sorting."<<endl;
			err = true;
			break;
		}
	}
	
	if( !err )
		cout << "Success" << endl;

	return 0;
}
