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

void sequential_quickSort( int *A, int start, int end )
{
    int tmp;
    int i = start;
    int j = end;

    if( i > j || A == NULL )
        return;

    int pivot = A[ (start + end)/2 ];

    // partition
    while(i <= j)
    {
        while(A[i] < pivot) i++;
        while(A[j] > pivot) j--;
        if (i <= j)
        {
            tmp  = A[i];
            A[i] = A[j];
            A[j] = tmp;
            i++;
            j--;
        }
    };

    // recursive part
    if (start < j)
        sequential_quickSort(A, start, j);
    if (i < end)
        sequential_quickSort(A, i, end);
}

void mpi_hypersort( void )
{
    int len;
    vector<int> data;
    const int ROOT = 0;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Find out rank, size
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int size;
    int *partitionsSize;
    int *partitionsOfst;

    if( world_rank == ROOT )
    {
        // Root node reads in the necessary files
        load_vector(data, "input.txt", len);

        int minSize = len / world_size;
        int remElts = len % world_size;

        // Partition the data
        int offset = 0;
        partitionsSize = (int*)malloc( world_size * sizeof(int) );
        partitionsOfst = (int*)malloc( world_size * sizeof(int) );

        // Determine the size of each partition and where each node starts
        for(int nodeId = 0; nodeId < world_size; nodeId++)
        {
            int sz = minSize;
            if( nodeId < remElts )
            {
                sz++;
            }

            partitionsSize[nodeId] = sz;
            partitionsOfst[nodeId] = offset;
            offset += sz;
        }
    }

    // Communicate overall length
    MPI_Bcast(&len, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Scatter each node their partition size Node 0 is root
    MPI_Scatter( partitionsSize, 1, MPI_INT, &size, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Barrier( MPI_COMM_WORLD);

    // Scatter each node their chunks using scatterv: Node 0 is root
    vector<int> partition(size);
    MPI_Scatterv( &data[0], partitionsSize, partitionsOfst, MPI_INT, \
                  &partition[0], size, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Barrier( MPI_COMM_WORLD);

    if( world_rank == ROOT )
    {
        free(partitionsSize);
        free(partitionsOfst);
    }

    // Now do seqential sort
    sequential_quickSort( &partition[0], 0, size-1 );

    // Lower and Higher halves swaps
    const int MAX_RX_SIZE = len;
    MPI_Comm SUB_COMM = MPI_COMM_WORLD;
    int h = int( ceil( log2f(world_size) ) );

    for( int i = 1; i <= h; i++ )
    {
        int subRank, subSize;
        MPI_Comm_rank(SUB_COMM, &subRank);
        MPI_Comm_size(SUB_COMM, &subSize);

        int half = (subSize + 1) / 2;

        // Compute the median and determine wich median to broadcast
        int median = (1<<31), *medians;
        char valid = 0, *validMedian;

        if( !partition.empty() )
        {
            valid = 1;
            median = partition[(size + 1)/2 - 1];
        }

        if( subRank == ROOT )
        {
            validMedian = (char*) malloc( sizeof(char) * subSize );
            medians     = (int*)  malloc( sizeof(int)  * subSize );
        }

        // Each node to communicate their medians, and if the median is valid to the root
        MPI_Gather( &valid,  1, MPI_CHAR, validMedian, 1, MPI_CHAR, ROOT, SUB_COMM );
        MPI_Gather( &median, 1, MPI_INT,  medians,     1, MPI_INT,  ROOT, SUB_COMM );

        if( subRank == ROOT )
        {
            // Let the root search to find the first valid median, the broadcast it
            for( int i = 0; i < subSize; i++ )
            {
                if( validMedian[i] )
                {
                    median = medians[i];
                    break;
                }
            }

            free(medians);
            free(validMedian);
        }

        // Broadcast the median to all nodes within the comm
        MPI_Bcast(&median, 1, MPI_INT, ROOT, SUB_COMM);

        // Each node to separate their partition into two
        // k, items greather than or equal to median
        int k = 0;
        while( k < size && partition[k] <= median )
        {
            k++;
        }

        // Get partner and swap: first half [0, k-1], second half [k, size]
        int rxCount;
        int partner = -1;
        MPI_Status rxStatus;
        int partnerRxBuffer[MAX_RX_SIZE];

        if(subRank < half )
        {
            // Low ranks
            partner = subRank + half;

            // Check if the destination is valid
            if( partner < subSize )
            {
                // Receive low list from partner, then send high list
                MPI_Recv(partnerRxBuffer, MAX_RX_SIZE, MPI_INT, partner, 0, SUB_COMM, &rxStatus);
                MPI_Send(&partition[k], size - k, MPI_INT, partner, 0, SUB_COMM);

                // clear space in the partition and add the new data
                MPI_Get_count(&rxStatus, MPI_INT, &rxCount);
                partition.erase( partition.begin() + k, partition.end() );
                partition.insert( partition.begin(), partnerRxBuffer, partnerRxBuffer + rxCount);
            }
        }
        else
        {
            // High ranks
            partner = subRank - half;

            // Check if the destination is valid
            if( partner < half )
            {
                // Send low list to partner then receive high list from partner
                MPI_Send(&partition[0], k, MPI_INT, partner, 0, SUB_COMM);
                MPI_Recv(partnerRxBuffer, MAX_RX_SIZE, MPI_INT, partner, 0, SUB_COMM, &rxStatus);

                // clear space in the partition and add the new data
                MPI_Get_count(&rxStatus, MPI_INT, &rxCount);
                partition.erase( partition.begin(), partition.begin() + k );
                partition.insert( partition.begin(), partnerRxBuffer, partnerRxBuffer + rxCount);
            }
        }

        if( subSize % 2 && subSize > 1)
        {
            // This node is ranked right in the middle of odd number of nodes
            size = partition.size();
            MPI_Barrier( SUB_COMM );

            int middleNode = half - 1;
            if( subRank == middleNode )
            {
                // Send high list to my right and the low list to my left, but receive nothing
                MPI_Send(&partition[k], size - k, MPI_INT, middleNode + 1, ROOT, SUB_COMM);
                MPI_Send(&partition[0], k, MPI_INT, middleNode - 1, ROOT, SUB_COMM);
                partition.clear();
            }
            else if( subRank == middleNode - 1 || subRank == middleNode + 1 )
            {
                MPI_Recv(partnerRxBuffer, MAX_RX_SIZE, MPI_INT, middleNode, 0, SUB_COMM, &rxStatus);
                MPI_Get_count(&rxStatus, MPI_INT, &rxCount);
                partition.insert( partition.begin(), partnerRxBuffer, partnerRxBuffer + rxCount);
            }
        }

        size = partition.size();
        sequential_quickSort( &partition[0], 0, size-1 );

        MPI_Barrier( SUB_COMM );

        // Split the subworld
        int color = subRank / half;
        MPI_Comm_split(SUB_COMM, color, subRank, &SUB_COMM);
    }

    // Each process to tell how many elements the root gathers from it.
    int *rxSizes, *rxOfsts;

    if( world_rank == ROOT )
    {
        rxSizes = (int*)  malloc( sizeof(int)  * world_size );
        rxOfsts = (int*)  malloc( sizeof(int)  * world_size );
    }

    // Each node to communicate their partition size to the root
    MPI_Gather( &size, 1, MPI_INT, rxSizes, 1, MPI_INT, ROOT, MPI_COMM_WORLD );

    if( world_rank == ROOT )
    {
        // compute the offset array
        int offset = 0;
        for( int i = 0; i < world_size; i++ )
        {
            rxOfsts[i] = offset;
            offset += rxSizes[i];
        }
    }

    MPI_Gatherv( &partition[0], size, MPI_INT, &data[0], rxSizes, rxOfsts, MPI_INT, ROOT, MPI_COMM_WORLD );

    if( world_rank == ROOT )
    {
        // Write the result
        ofstream result;

        result.open("output.txt", ios_base::out);
        for(int i = 0; i < len; i++)
        {
            result << data[i] << "\n";
        }

        result << endl;
        result.close();

        free(rxSizes);
        free(rxOfsts);
    }

    MPI_Finalize();
}

int main( int argc, char **argv )
{
    mpi_hypersort();
	return 0;
}
