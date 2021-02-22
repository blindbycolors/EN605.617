//
// Created by nou on 2/21/21.
//
#include <vector>

#ifndef MODULE4_ASSIGNMENT_CUH
#define MODULE4_ASSIGNMENT_CUH

std::vector<std::pair<int,int>> parseOperations(int, char **);
void printWarning(int , const int, const int);
void parseMathArgv(const int , char **, std::vector<int>&, std::vector<int>&);
void doMathOperations(int, char **);
void doCombos(const std::vector<std::pair<int, int>>);
void doCipher(int, char **);
void fillArrays(int * a, int *b, int totalThreads);

#endif //MODULE4_ASSIGNMENT_CUH
