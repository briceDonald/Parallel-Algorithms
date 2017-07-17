#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "stdlib.h"
#include <string.h>
#include <math.h>
#include <mpi.h>

// USES C++ 98

using namespace std;

int* load_matrix( const char *filename, int &rows, int &cols )
{
    int idx = 0;
    string value;
    ifstream datafile;

    vector< vector<int> > dataMatrix;
    int *matrix;

    cout << "-> Opening file : " << filename << endl;

    try
    {
		datafile.open( filename );

		while( datafile.good() )
		{
            getline( datafile, value, ' ' );

            if( idx == 0 )
            {
                rows = atoi( value.c_str() );
            }
            else if( idx == 1 )
            {
                cols = atoi( value.c_str() );
                matrix = (int*) malloc( rows * cols * sizeof(int) );
            }
            else if( idx > 1 )
            {
                matrix[idx-2] = atoi( value.c_str() );
            }

            idx++;
		}

        if( rows * cols != idx-2 )
            throw string("Matrix missing data: ");

		datafile.close();
	}
    catch(string s)
    {
        cout << "Error: " << s << endl;
        throw s;
    }
	catch(...)
	{
		cout << "Error opening file: " << value << endl;
	}

    cout <<matrix[0]<<"---"<<endl;
    return matrix;
}

void load_vector( vector<int> &data, const char *filename, int &len )
{
    len = 0;
    string value;
    ifstream datafile;

    cout << "Opening file : " << filename << endl;

    try
    {
		datafile.open( filename );

		while( datafile.good() )
		{
			getline( datafile, value, ' ');
			data.push_back(  atoi( value.c_str() ) );
			len++;
		}

		datafile.close();
	}
	catch(...)
	{
		cout << "Error opnening file: " << filename << endl;
	}

    return;
}

void sequential_mattrix_multiply( char* processors )
{
    int vLen;
    vector<int> V;
    int rows, cols, *matrix = NULL;

    load_vector(V, "vector.txt", vLen);
    matrix = load_matrix( "matrix.txt", rows, cols );

    if( vLen != rows )
    {
        throw "Unmaching sizes";
    }

    if( matrix == NULL || V.empty() )
    {
        throw "Unable to read matrix.";
    }

    int localResult[cols];
    memset( localResult, 0, cols*sizeof(int) );


    for(int i = 0; i < rows*cols; i++)
    {
        int vecIdx = (i / cols) % vLen;
        int idx  = i % cols;

        localResult[idx] += matrix[i] * V[vecIdx];
        // cout << i << " v: " << idx << " : " << vecIdx << " : "<< matrix[i] << endl;
    }

    for(int i = 0; i < cols; i++)
    {
        cout << i << " SRes: " << localResult[i] << endl;
    }

    free(matrix);
}

void mpi_matrix_multiply( char* processors )
{
    int vLen;
    vector<int> V;

    int rows, cols, *matrix = NULL;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Find out rank, size
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // We are assuming at least 2 processes for this task
    if( world_size < 1 )
    {
        fprintf(stderr, "World size must be greater than 1 for %s\n", processors);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if( world_rank == 0 )
    {
        // Root process reads in the necessary files
        load_vector(V, "vector.txt", vLen);
        matrix = load_matrix( "matrix.txt", rows, cols );

        if( vLen != rows )
        {
            throw "Unmaching sizes";
        }

        if( matrix == NULL || V.empty() )
        {
            throw "Unable to read matrix.";
        }
    }

    // MPI WORK
    // Broadcast matrix size and vector
//    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    int numChunks = rows / world_size;
    int remChunks = rows % world_size;
    if( world_rank < remChunks ) numChunks++;

    int vecMat[vLen];
    int vecIdx;
    int rxSize = numChunks * cols;
    int rxChunks[numChunks*cols];

    if (world_rank == 0)
    {
        // Copy the vector matrix
        memcpy( vecMat, &V[0], vLen*sizeof(int) );

        // Compute the number of elements to send to each processes
        int chunkSize[world_size];
        int chunkOfst[world_size];
        int vecMatIdx[vLen];
        int offset = 0;

        // each i here is actually representative of a world_rank
        for(int i = 0; i < world_size; i++)
        {
            int chunks = rows / world_size;
            if( i < remChunks )
            {
                chunks++;
            }

            chunkOfst[i] = cols * offset;
            chunkSize[i] = cols * chunks;
            vecMatIdx[i] = offset % vLen;
            offset += chunks;
            cout<<vecMatIdx[i] << " <==" << endl;
        }

        // Scatter each process their chunks
        MPI_Scatterv(matrix, chunkSize, chunkOfst, MPI_INT, rxChunks, rxSize, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // Communicate where each node should get their value from the vector
        MPI_Scatter(vecMatIdx, 1, MPI_INT, &vecIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else
    {
        // Scatter each process their chunks
        MPI_Scatterv(matrix, NULL, NULL, MPI_INT, rxChunks, rxSize, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // Communicate where each node should get their value from the vector
        MPI_Scatter(NULL, 1, MPI_INT, &vecIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Bcast(vecMat, vLen, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Every nodes to use vecIdx, their unique index in the vector matrix vecmat
    // to compute and reduce all their chunks to one single chunk localResult
    int localResult[cols];
    memset( localResult, 0, cols*sizeof(int) );

    for(int i = 0; i < rxSize; i++)
    {
        int ofst = i / cols;
        int idx  = i % cols;

        localResult[idx] += rxChunks[i] * vecMat[(vecIdx+ofst) % vLen];
        // cout << world_rank << " v: " << idx << " : " << vecIdx+ofst << " : "<< rxChunks[i] << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for(int i = 0; i < cols; i++)
        cout << world_rank << i << " v: " << " : "<< localResult[i] << endl;

    if( world_rank == 0 )
    {
        // Write the result
        // for(int i = 0; i < cols; i++)
        // {
        //     cout << i << " v: " << " : "<< localResult[i] << endl;
        // }

        // Clean exit
        free(matrix);
    }

    MPI_Finalize();
    // cout << "done " << endl;
}

int main( int argc, char **argv )
{
    sequential_mattrix_multiply(argv[0]);
    mpi_matrix_multiply(argv[0]);
	return 0;
}
