#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "stdlib.h"
#include <math.h>

using namespace std;
#define THREADS_PER_BLOCK (1 << 9)



__global__ void  par_compute_min( int *d_in, int *d_out, int sz )
{
	// get thread index
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int h = (int) ceil( log2(sz*1.0f) );

	if( tid < sz )
	{
		int tpMin;
		for(int i = 1; i <= h; i++)
		{
			int base = (int) ceil( sz/pow(2.0f, i) );
			if( tid < base )
			{
				// compute the left and right indeces
				int L = 2 * tid;
				int R = 2 * tid + 1;

				// Assign temp max to left child
				tpMin = d_in[L];

				if( R < (int)ceil(sz/pow(2.0f, i-1)) )
					tpMin = ( d_in[L] <= d_in[R] ) ? d_in[L] : d_in[R];
				
				__syncthreads();
				d_in[tid] = tpMin;
				__syncthreads();
			}
		}

		if( tid == 0 )
			*d_out = d_in[tid];
	}
}

__global__ void par_compute_last_digit( int *d_in, int *d_out, int sz  )
{
	// get thread index
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if( tid < sz )
	{
		d_out[tid] = d_in[tid] % 10;
	}
}


/*************************************** SERIAL CODE ****************************************/
void printProperties( int device )
{
	struct cudaDeviceProp prop;
	
	if( cudaGetDeviceProperties( &prop, device ) )
	{
		cout << "error: Failed to get device properties." << endl;
		return;
	}

	cout << "BRICE NGNIGHA" << endl;
	cout << "Prob1.\n" << endl;
	
	cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
	cout << "Max Threads per dims x: " << prop.maxThreadsDim[0] << " y: "<< prop.maxThreadsDim[1] << " z: " << prop.maxThreadsDim[2] << endl;
	cout << "Shared memory per block " << prop.sharedMemPerBlock << endl;
	cout << "Registers per block " << prop.regsPerBlock << endl;
	cout << "Max grid size x: " << prop.maxGridSize[0] << " y: "<< prop.maxGridSize[1] << " z: " << prop.maxGridSize[2] << endl;
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

int sequential_compute_min( int *A, int len )
{
	if( !A || len == 0 )
		return -1;
	
	int min = A[0];
	
	for(int i = 0; i < len; i++)
		if(min > A[i])
			min = A[i];
	return min;
}

int parallel_compute_min(int * A, int len )
{
	int min;
	int *d_in, *d_out; 
	int *h_in = A;

	cudaMalloc( (void**) &d_in, len*sizeof(int) ); 
	cudaMalloc( (void**) &d_out, sizeof(int) );
	
	cudaMemcpy( d_in, h_in, len*sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = (int) ceil( (1.0f * len)/THREADS_PER_BLOCK );
	par_compute_min<<<numBlocks, THREADS_PER_BLOCK>>>( d_in, d_out, len );

	cudaMemcpy( &min, d_out, sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree( d_in );
	cudaFree( d_out );
	return min;	
}

void sequential_compute_last_digit( int *A, int *B, int len)
{
	for(int i = 0; i < len; i++)
		B[i] = A[i] % 10;
}

void parallel_compute_last_digit(int *A, int *B, int len)
{
	int *d_in, *d_out; 
	int *h_in = A, *h_out = B;

	cudaMalloc( (void**) &d_in, len*sizeof(int) ); 
	cudaMalloc( (void**) &d_out,len* sizeof(int) );
	
	cudaMemcpy( d_in, h_in, len*sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = (int) ceil( (1.0f * len)/THREADS_PER_BLOCK );
	par_compute_last_digit<<<numBlocks, THREADS_PER_BLOCK>>>( d_in, d_out, len );

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

	bool err1=false, err2=false;
	vector<int> dataVec;
	int len = read_csv( dataVec, "inp.txt" );
	int *A = &dataVec[0];

	int parMin, seqMin;
	parMin = parallel_compute_min( A, len );
	seqMin = sequential_compute_min( A, len );

	cout << "\nProb. 1-a -- COMPUTING THE MIN IN LOG N TIME --" << endl;
	cout << "1-b => \t" << "Par" << "\t-vs-\t" << "Seq" << endl;   
	if(parMin != seqMin)
	{
		cout <<"Error 1-a: " << "Output mismatch on array min."<<endl;
		err1  = true;
	}
	else
		cout << "1-b => \t" << parMin << "\t-vs-\t" << seqMin << endl;


	int parB[len], seqB[len];
	parallel_compute_last_digit( A, parB, len );
	sequential_compute_last_digit( A, seqB, len );

	cout << "\n\nProb. 1-b -- COMPUTING THE LAST DIGIT OF EACH ARRAY NUMBERS --" << endl;
	cout << "1-b => \t" << "Par" << "\t-vs-\t" << "Seq\tVal" << endl;   
	for(int i = 0; i < len; i ++)
	{
		cout << "1-b => \t" << parB[i] << "\t-vs-\t" << seqB[i] << "\t" << A[i] << endl;
		if(parB[i] != seqB[i])
		{
			cout <<"Error 1-b: " << "Output mismatch on last digit."<<endl;
			err2 = true;
			break;
		}
	}

	if( !err1 && !err2 )
		cout << "\nSuccess" << endl;

	return 0;
}
