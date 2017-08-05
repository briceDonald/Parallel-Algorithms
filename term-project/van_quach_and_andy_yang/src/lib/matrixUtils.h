//
// Created by van quach on 7/29/17.
//

#include <iostream>
#include <string>
#include <vector>

using namespace std;

#ifndef SRC_MATRIXUTILS_H
#define SRC_MATRIXUTILS_H
void printMatrix(int, int, float*);
void generateMatrix(int, string);
void readInputData(string, vector<float>& out);
void saveMatrixResult(vector<float>&, string);
void saveToLogFile(string, int, long);
#endif //SRC_MATRIXUTILS_H
