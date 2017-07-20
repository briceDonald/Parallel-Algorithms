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

void load_matrix( vector<int> &matrix, const char *filename, int &rows, int &cols )
{
    cout << "Opening file : " << filename << endl;
    try
    {
        char c;
        string num;
        int idx = 0;
        ifstream datafile;

        datafile.open( filename );

		while( datafile.good() )
		{
            c  = datafile.get();

            if( c == ' ' || c == '\t' || c == '\r' || c == '\n' )
            {
                if( num.size() )
                {
                    if( idx == 0 )
                    {
                        rows = atoi( num.c_str() );
                    }
                    else if( idx == 1 )
                    {
                        cols = atoi( num.c_str() );
                    }
                    else
                    {
                        matrix.push_back( atoi( num.c_str() ) );
                    }

                    idx++;
                }

                num = "";
                continue;
            }

            num.append(1, c);
		}

		datafile.close();
	}
	catch(...)
	{
		cout << "Error opening file: " << filename << endl;
        throw string( "Error opening input file.");
	}
}

void load_vector( vector<int> &data, const char *filename, int &len )
{
    cout << "Opening file : " << filename << endl;
    try
    {
        char c;
        string num;
        len = 0;
        ifstream datafile;

		datafile.open( filename );

        while( datafile.good() )
		{
            c  = datafile.get();

            if( c == ' ' || c == '\t' || c == '\r' || c == '\n' )
            {
                if( num.size() )
                {
                    data.push_back( atoi( num.c_str() ) );
                    len++;
                }

                num = "";
                continue;
            }

            num.append(1, c);
		}

		datafile.close();
	}
	catch(...)
	{
		cout << "Error opening file: " << filename << endl;
        throw string( "Error opening input file.");
	}
}

void mpi_matrix_multiply( char* processors )
{
    int vLen;
    vector<int> V;

    int rows, cols, *Mat = NULL;
    vector<int> matrix;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Find out rank, size
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if( world_rank == 0 )
    {
        // Root process reads in the necessary files
        load_vector( V, "vector.txt", vLen );
        load_matrix( matrix,  "matrix.txt", rows, cols );
        Mat=&matrix[0];

        if( vLen != rows )
        {
            throw "Unmaching sizes";
        }

        if( matrix.empty() || V.empty() )
        {
            throw "Unable to read matrix.";
        }
    }

    // Broadcast matrix size and multiplicator vector to matrix
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

        // Compute the number of elements to ''scatterv' to each processes
        int chunkSize[world_size];
        int chunkOfst[world_size];
        int vecMatIdx[world_size];
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
        }

        // Scatter each process their chunks
        MPI_Scatterv(Mat, chunkSize, chunkOfst, MPI_INT, rxChunks, rxSize, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // Communicate where each node should get their value from the vector
        MPI_Scatter(vecMatIdx, 1, MPI_INT, &vecIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else
    {
        // Scatter each process their chunks
        MPI_Scatterv(Mat, NULL, NULL, MPI_INT, rxChunks, rxSize, MPI_INT, 0, MPI_COMM_WORLD);
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
        int chunkId = i / cols;
        int idx  = i % cols;
        localResult[idx] += rxChunks[i] * vecMat[(vecIdx+chunkId) % vLen];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Do map operation in log n time
    int curSize = world_size;
    int h = int( ceil( log2f(world_size) ) );
    for( int i = 1; i <= h; i++ )
    {
        int half = int( ceil( curSize / 2.0) );

        if( world_rank >= half )
        {
            int rankDest = world_rank - half;
            if( rankDest >= 0 && rankDest < half )
            {
                // Send local results to correspondant
                MPI_Send(localResult, cols, MPI_INT, rankDest, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            int rankSrce = world_rank  + half;
            if( rankSrce >= half && rankSrce < curSize )
            {
                // Receive local results from correspondant
                int rxData[cols];
                MPI_Recv(rxData, cols, MPI_INT, rankSrce, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for(int i = 0; i < cols; i++)
                {
                    localResult[i] += rxData[i];
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        curSize = half;
    }

    if( world_rank == 0 )
    {
        // Write the result
        ofstream result;

        result.open("result.txt", ios_base::out);
        for(int i = 0; i < cols; i++) result << localResult[i] << " ";
        result << endl;
        result.close();
    }

    MPI_Finalize();
}

int main( int argc, char **argv )
{
    mpi_matrix_multiply(argv[0]);
	return 0;
}
