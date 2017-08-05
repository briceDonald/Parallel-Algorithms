//
// Created by van quach on 7/29/17.
//
#include <random>
#include <sstream>
#include <fstream>
#include "matrixUtils.h"

void printMatrix(int nRows, int nCols, float *matrix) {
	for(int rowIndex = 0; rowIndex < nRows; rowIndex++) {
		for(int colIndex = 0; colIndex < nCols; colIndex++) {
			cout << matrix[rowIndex + nRows * colIndex] << " ";
		}
		cout << endl;
	}
}

void generateMatrix(int size, string path) {
	stringstream ssFileName;
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-999, 999);

	ssFileName << path << "/matrix_" << size << ".txt";
	ofstream inputFile;
	inputFile.open(ssFileName.str());
	for(int i= 0; i < size; i++) {
		inputFile << distribution(generator);
		inputFile << endl;
	}
	inputFile.close();
}

void readInputData(string fileName, vector<float>& out) {
	ifstream inFile(fileName);
	string line;
	while(getline(inFile, line)) {
		istringstream lineStream(line);
		out.push_back(stof(line));
	}
}

void saveMatrixResult(vector<float>& result, string path) {
	int size = result.size();
	stringstream ssFileName;
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-999, 999);

	ssFileName << path << "/result_" << size << ".txt";
	ofstream inputFile;
	inputFile.open(ssFileName.str());
	for(int i= 0; i < result.size(); i++) {
		inputFile << distribution(generator);
		inputFile << endl;
	}
	inputFile.close();
}

void saveToLogFile(string logFileName, int matrixSize, long durationInmicrosecs) {
	ofstream inputFile;
	inputFile.open(logFileName, fstream::app);
	inputFile << endl <<  matrixSize << "|" << durationInmicrosecs;
}