#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "stdlib.h"
#include "string.h"
#include <math.h>

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


/////////////////////////// PROBLEM 1 //////////////////////////////////////////////

void sequential_matrix_multiply( void )
{
    int vLen;
    vector<int> V, Mat;
    int rows, cols, *matrix = NULL;

    load_vector( V, "vector.txt", vLen );
    load_matrix( Mat, "matrix.txt", rows, cols );
    matrix = &Mat[0];

    if( vLen != rows )
    {
        throw "Unmaching sizes";
    }

    if( Mat.empty() || V.empty() )
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
    }

    // Write the result
    ofstream result;

    result.open("seq_result.txt", ios_base::out|ios_base::trunc);
    for(int i = 0; i < cols; i++)
    {
        result << localResult[i] << "\n";
    }

    result << endl;
    result.close();
}

/////////////////////////// PROBLEM 2 //////////////////////////////////////////////

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


void run_seq_quicksort()
{
    vector<int> vec;
    int len;
    load_vector(vec, "input.txt", len);
    sequential_quickSort(&vec[0], 0, len-1);

    // Write the result
    ofstream result;

    result.open("seq_output.txt", ios_base::out|ios_base::trunc);
    for(int i = 0; i < len; i++)
    {
        result << vec[i] << "\n";
    }

    result << endl;
    result.close();

}

///////////////////////////   MAIN  //////////////////////////////////////////////

int main( int argc, char **argv )
{
    sequential_matrix_multiply();
    run_seq_quicksort();
    return 0;
}
