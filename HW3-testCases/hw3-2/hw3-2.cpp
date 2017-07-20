#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "stdlib.h"
#include <math.h>
#include <mpi.h>

// USES C++ 98

using namespace std;

void load_data( vector<int> &data, const char *filename, int &len )
{
    len = 0;
    ifstream datafile;

    try
    {
        cout << "Opening file : " << filename << endl;
		datafile.open( filename );

        char c;
        string num;
		while( datafile.good() )
		{
            c  = datafile.get();
            if( c == ' ' || c == '\r' || c == '\n' )
            {
                if(num.size() )
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
        load_data(data, "input.txt", len);

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
    MPI_Scatterv( &data[0], partitionsSize, partitionsOfst, MPI_INT, &partition[0], size, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Barrier( MPI_COMM_WORLD);

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

        // Compute the median
        int median;
        if( subRank == ROOT )
        {
            median = partition[( (size + 1)/2 - 1 )];
        }

        // Broadcast the median to all nodes within the comm
        MPI_Bcast(&median, 1, MPI_INT, ROOT, SUB_COMM);
        MPI_Barrier(SUB_COMM);

        // Each node to separate their partition into two
        // k, items greather than or equal to median
        int k = 0;
        while( k < size && partition[k] <= median ) k++;

        // first half [0, k-1], second half [k, size]
        // Get partner and swap
        int partner = -1;
        MPI_Status rxStatus;
        int half = (subSize + 1) / 2;
        int partnerRxBuffer[MAX_RX_SIZE];

        if(subRank < half )
        {
            // Low ranks
            partner = subRank + half;

            // Check if the destination is valid
            if( partner < subSize )
            {
                // Receive low list from partner
                MPI_Recv(partnerRxBuffer, MAX_RX_SIZE, MPI_INT, partner, 0, SUB_COMM, &rxStatus);

                // Send high list to partner
                MPI_Send(&partition[k], size - k, MPI_INT, partner, 0, SUB_COMM);

                // clear space in the partition and add the new data
                int rxCount;
                MPI_Get_count(&rxStatus, MPI_INT, &rxCount);
                //                cout << "Low rxCount: " << rxCount << " k " << k << endl;
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
                // Send low list to partner
                MPI_Send(&partition[0], k, MPI_INT, partner, 0, SUB_COMM);

                // Receive high list from partner
                MPI_Recv(partnerRxBuffer, MAX_RX_SIZE, MPI_INT, partner, 0, SUB_COMM, &rxStatus);

                // clear space in the partition and add the new data
                int rxCount;
                MPI_Get_count(&rxStatus, MPI_INT, &rxCount);
                //                cout << "Hgh rxCount: " << rxCount << " k " << k << endl;
                partition.erase( partition.begin(), partition.begin() + k );
                partition.insert( partition.begin(), partnerRxBuffer, partnerRxBuffer + rxCount);
            }
        }

        size = partition.size();
        cout << "WLD: " << world_rank << " sub " << subRank << " => " << partner << " size " << size << endl;

        //        cout << subRank << " B median: " << median << " half " << half << endl;
        MPI_Barrier( MPI_COMM_WORLD );

        sequential_quickSort( &partition[0], 0, size-1 );


        // divide the subworld by 2
        int color = subRank / 2;//half;
        MPI_Comm_split(SUB_COMM, color, subRank, &SUB_COMM);

        if( subRank == ROOT )
        {
          cout << "........................................................." << endl;
            // split the context
        }
    }

    {
        char t[30];
        sprintf(t, "Rank: %d Size: %d\n", world_rank, size);
        string str(t);

        for(int i = 0; i < size; i++)
        {
            sprintf(t, "%d\t", partition[i]);
            str += string(t);
        }

        MPI_Barrier( MPI_COMM_WORLD);
        if(size)
            cout << str << endl;
    }

    // // Get the median
    // if( world_rank == ROOT )
    // {
    //     int median = partition[( (size+1)/2 - 1 )];

    //     // Broadcast the median
    //     cout << median << endl;
    //     cout << str << endl;
    // }


    if( world_rank == 0 )
    {
        // Write the result
        // ofstream result;

        // result.open("result.txt", ios_base::out);
        //        for(int i = 0; i < cols; i++) result << localResult[i] << " ";
        // result << endl;
        // result.close();

        // Clean up
        free(partitionsOfst);
        free(partitionsSize);
    }

    MPI_Finalize();
}

int main( int argc, char **argv )
{
    // vector<int> vec;
    // int len;
    // load_data(vec, "input.txt", len);
    // sequential_quickSort(&vec[0], 0, len-1);
    // for(int i = 0; i < len; i++)
    //     cout << "-> " << vec[i] << endl;

    mpi_hypersort();
    return 0;
}
