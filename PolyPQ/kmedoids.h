#ifndef KMEDOIDS_H_
#define KMEDOIDS_H_

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


double getDistance(const double *x1, const double *x2, int m);
void autoscaling(double* const x, const int n, const int m);
int getCluster(const double* const x, const double* const c, const int m, int k);
char constr(const int *y, const int val, int s);
int* startCoreNums(const int k, const int n);
void detCores(const double* const x, double* const c, const int* const sn, int k, const int m);
void detStartPartition(const double* const x, const double* const c, int* const y, int* const nums, int n, const int m, const int k);
void calcCores(const double* const x, double* const c, const int* const y, const int* const nums, const int n, const int m, const int k);
int getMedoidNum(const double* const x, const double* const c, const int* const y, const int n, const int m, const int id);
void setMedoids(const double *x, double * const c, const int* const y, const int n, const int m, int k);
char checkPartition(const double* const x, const double* const c, int* const y, int* const nums, int n, const int m, const int k);
char cyclicRecalc(const double* const x, double* const c, int* const y, int* const nums, const int n, const int m, const int k);
void kmedoids(const double* const X, int* const y, const int n, const int m, const int k);
extern double (*kmedoids_distance)(const double*, const double*, int);

#endif