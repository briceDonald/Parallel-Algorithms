//
// Created by van quach on 7/30/17.
//initrd.img.old

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sstream>
#include <chrono>

#include "./lib/matrixUtils.h"

#define CUBLAS_CHECK(stat, msg) if(stat != CUBLAS_STATUS_SUCCESS) { cout << msg << endl; return EXIT_FAILURE; }
#define IDX2C(row, col, ld) (((col)*(ld))+(row))

int decomposeLU(cublasHandle_t handle, int nRows, int nCols, float *matrix, int lda, int *permuteRows, float eps);
int decomposeBlockedLU(int nRows, int nCols, float *matrix, int blockSize, float eps);

using namespace std;
using namespace std::chrono;

int main() {

//
//	float *matrix;
//	float esp = 1e-6;




//	matrix = (float *) malloc(16 * sizeof(*matrix));
//	matrix[IDX2C(0, 0, 4)] = 2;
//	matrix[IDX2C(0, 1, 4)] = -3;
//	matrix[IDX2C(0, 2, 4)] = 1;
//	matrix[IDX2C(0, 3, 4)] = 2;
//
//	matrix[IDX2C(1, 0, 4)] = 5;
//	matrix[IDX2C(1, 1, 4)] = -1;
//	matrix[IDX2C(1, 2, 4)] = 2;
//	matrix[IDX2C(1, 3, 4)] = 1;
//
//	matrix[IDX2C(2, 0, 4)] = 3;
//	matrix[IDX2C(2, 1, 4)] = 2;
//	matrix[IDX2C(2, 2, 4)] = 6;
//	matrix[IDX2C(2, 3, 4)] = -5;
//
//	matrix[IDX2C(3, 0, 4)] = -1;
//	matrix[IDX2C(3, 1, 4)] = 1;
//	matrix[IDX2C(3, 2, 4)] = 3;
//	matrix[IDX2C(3, 3, 4)] = 2;

/*
 *
 2    -3    1     2
 2.5  6.5   -0.5  -4
 1.5  1     1     -4
 -0.5 -1/13 9/13  71/13



 */

//	cout << "Matrix A " << endl;
//	printMatrix(4, 4, matrix);
//	//decomposeLU(handle, 3, 3, d_matrix, permute, esp);
//	decomposeBlockedLU(4, 4, matrix, 3, esp);
//
//	printMatrix(4, 4, matrix);
//
//	free(matrix);
/*
	generateMatrix(10, "./data");
	generateMatrix(100, "./data");
	generateMatrix(1000, "./data");
	generateMatrix(10000, "./data");
*/
	vector<float> v_matrix;
	float *matrix;
	stringstream ssFileName, ssLogFilename;
	float eps = 1e-6;
	int numOfRun = 10;

	high_resolution_clock::time_point t1, t2;
	ssLogFilename << "./log/log.txt";
	for(int size = 10; size < 20000; size *= 10) {
		ssFileName.str("");
		ssFileName.clear();
		ssFileName << "./data" << "/matrix_" << size << ".txt";
		readInputData(ssFileName.str(), v_matrix);
		matrix = (float *)malloc(size * size * sizeof(float));
		copy(v_matrix.begin(), v_matrix.end(), matrix);

		cout << "Start LU decompose for matrix size " << size << endl;
		for(int runIndex = 0; runIndex < numOfRun; runIndex++) {
			t1 = high_resolution_clock::now();
			decomposeBlockedLU(size, size, matrix, size / 5, eps);
			t2 = high_resolution_clock::now();
			auto duration = duration_cast<microseconds>( t2 - t1 ).count();
			saveToLogFile(ssLogFilename.str(), size, duration);
			cout << " Run " << runIndex << " takes " << duration << " in microseconds." << endl;
		}
		cout << "End LU decompose for matrix size " << size << endl;

		free(matrix);
	}


	return 0;
}

int decomposeLU(cublasHandle_t handle, int nRows, int nCols, float *matrix, int lda, int *permuteRows, float eps) {
	cublasStatus_t stat;
	int minDim = min(nRows, nCols);
	int pivotRow;
	float alpha;

	for (int rowIndex = 0; rowIndex < minDim - 1; rowIndex++) {
		stat = cublasIsamax(handle, nRows - rowIndex, &matrix[IDX2C(rowIndex, rowIndex, lda)], 1, &pivotRow);
		CUBLAS_CHECK(stat, "CUBLAS samax failed")
		pivotRow += rowIndex - 1;
		int rowIndexP1 = rowIndex + 1;
		permuteRows[rowIndex] = pivotRow;
		if (pivotRow != rowIndex) {
			stat = cublasSswap(handle, nCols, &matrix[IDX2C(pivotRow, 0, lda)], lda, &matrix[IDX2C(rowIndex, 0, lda)], lda);
			CUBLAS_CHECK(stat, "CUBLAS swap failed")
		}
		float valcheck;
		stat = cublasGetVector(1, sizeof(float), &matrix[IDX2C(rowIndex, rowIndex, lda)], 1, &valcheck, 1);
		CUBLAS_CHECK(stat, "CUBLAS get vector failed")

		if (fabs(valcheck) < eps) {
			return EXIT_FAILURE;
		}
		if (rowIndexP1 < nRows) {
			alpha = 1.0f / valcheck;
			stat = cublasSscal(handle, nRows - rowIndexP1, &alpha, &matrix[IDX2C(rowIndexP1, rowIndex, lda)], 1);
			CUBLAS_CHECK(stat, "CUBLAS scal failed")
		}

		if (rowIndexP1 < minDim) {
			alpha = -1.0f;
			stat = cublasSger(handle, nRows - rowIndexP1, nCols - rowIndexP1, &alpha, &matrix[IDX2C(rowIndexP1, rowIndex, lda)], 1,
			                  &matrix[IDX2C(rowIndex, rowIndexP1, lda)], lda, &matrix[IDX2C(rowIndexP1, rowIndexP1, lda)], lda);
			CUBLAS_CHECK(stat, "CUBLAS ger failed")
		}

	}
	return EXIT_SUCCESS;
}

int decomposeBlockedLU(int nRows, int nCols, float *matrix, int blockSize, float eps) {
	float *d_matrix;
	int permute[nRows];
	int minSize, realBlockSize, lda, returnVal;
	float alpha, beta;

	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	lda = nRows;
	for(int i = 0; i < nRows; i++) {
		permute[i] = -1;
	}

	stat = cublasCreate(&handle);
	CUBLAS_CHECK(stat, "CUBLAS initialization failed")
	cudaStat = cudaMalloc((void **) &d_matrix, nRows * nCols * sizeof(*matrix));
	if (cudaStat != cudaSuccess) {
		cout << "Error while allocating device memory" << endl;
		return EXIT_FAILURE;
	}

	stat = cublasSetMatrix(nRows, nCols, sizeof(*matrix), matrix, nRows, d_matrix, nRows);
	CUBLAS_CHECK(stat, "CUBLAS set matrix failed")

	minSize = min(nRows, nCols);
	if (blockSize > minSize || blockSize == 1) {
		returnVal = decomposeLU(handle, nRows, nCols, d_matrix, lda, permute, eps);
	} else {
		for(int i = 0; i < minSize; i += blockSize) {
			realBlockSize = min(minSize - i, blockSize);
			returnVal = decomposeLU(handle, realBlockSize, realBlockSize, &d_matrix[IDX2C(i, i, lda)], lda, &permute[i], eps);
			for(int p = i; p < min(nRows, i + realBlockSize); p++) {
				if (permute[p] > 0) {
					permute[p] += i;

					if (permute[p] != p) {
						stat = cublasSswap(handle, i, &d_matrix[IDX2C(p, 0, lda)], lda, &d_matrix[IDX2C(permute[p], 0, lda)], lda);
						CUBLAS_CHECK(stat, "CUBLAS 1st swap failed")
						stat = cublasSswap(handle, nCols - i - realBlockSize, &d_matrix[IDX2C(p, i + realBlockSize, lda)], lda,
						                   &d_matrix[IDX2C(permute[p], i + realBlockSize, lda)], lda);
						CUBLAS_CHECK(stat, "CUBLAS 2nd swap failed")
					}
				}
			}

			alpha = 1;
			stat = cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, realBlockSize,
			                   nCols - i - realBlockSize, &alpha, &d_matrix[IDX2C(i, i, lda)], lda,
			                   &d_matrix[IDX2C(i, i+realBlockSize, lda)], lda);
			CUBLAS_CHECK(stat, "CUBLAS trsm failed for U")

			stat = cublasStrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, nRows - i - realBlockSize,
			                   realBlockSize, &alpha, &d_matrix[IDX2C(i, i, lda)], lda,
			                   &d_matrix[IDX2C(i+realBlockSize, i, lda)], lda);
			CUBLAS_CHECK(stat, "CUBLAS trsm failed for U")
			if (i+realBlockSize < nRows)
			{
				alpha = -1;
				beta = 1;
				stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,   nRows - i - realBlockSize, nCols - i - realBlockSize,
				                   realBlockSize, &alpha, &d_matrix[IDX2C(i+realBlockSize, i, lda)], lda,
				                   &d_matrix[IDX2C(i, (realBlockSize+i), lda)],lda, &beta,
				                   &d_matrix[IDX2C(i+realBlockSize, realBlockSize+i, lda)],lda );
				CUBLAS_CHECK(stat, "CUBLAS trsm failed")
			}
		}

	}

	stat = cublasGetMatrix(nRows, nCols, sizeof(*matrix), d_matrix, nRows, matrix, nRows);
	CUBLAS_CHECK(stat, "CUBLAS get matrix failed")

	cudaFree(d_matrix);
	cublasDestroy(handle);
	return returnVal;
}
