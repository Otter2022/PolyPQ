/* kmedoids_parallel.c */
#include "kmedoids.h"
#include <pthread.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>  // For printf

/* Global pointer for custom distance function */
double (*kmedoids_distance)(const double*, const double*, int) = NULL;

/* Use 16 threads by default (adjust as needed) */
#define NUM_THREADS 16

/* --- getDistance remains the same (with custom function support) --- */
double getDistance(const double *x1, const double *x2, int m) {
    if (kmedoids_distance) {
        return kmedoids_distance(x1, x2, m);
    }
    double d, r = 0.0;
    while (m--) {
        d = *(x1++) - *(x2++);
        r += d * d;
    }
    return sqrt(r);
}

/* --- autoscaling remains unchanged --- */
void autoscaling(double* const x, const int n, const int m) {
    const int s = n * m;
    int j;
    for (j = 0; j < m; j++) {
        double sd, Ex = 0.0, Exx = 0.0, *ptr;
        for (ptr = x + j; ptr < x + s; ptr += m) {
            sd = *ptr;
            Ex += sd;
            Exx += sd * sd;
        }
        Exx /= n;
        Ex /= n;
        sd = sqrt(Exx - Ex * Ex);
        for (ptr = x + j; ptr < x + s; ptr += m) {
            *ptr = (*ptr - Ex) / sd;
        }
    }
}

/* --- Serial helper functions --- */

/* Returns the cluster (medoid) index for point x given medoid set c */
int getCluster(const double* const x, const double* const c, const int m, int k) {
    int res = --k;
    double minD = getDistance(x, c + k * m, m);
    while (k--) {
        const double curD = getDistance(x, c + k * m, m);
        if (curD < minD) {
            minD = curD;
            res = k;
        }
    }
    return res;
}

/* Returns a random set of k unique indices from 0 to n-1 */
char constr(const int *y, const int val, int s) {
    while (s--) if (*(y++) == val) return 1;
    return 0;
}
int* startCoreNums(const int k, const int n) {
    srand((unsigned)clock());
    int *y = (int*)malloc(k * sizeof(int));
    int i = 0, val;    
    while (i < k) {
        do val = rand() % n;
        while (constr(y, val, i));
        y[i] = val;
        i++;
    }
    return y;
}

/* Copy rows of x indexed by sn into c */
void detCores(const double* const x, double* const c, const int* const sn, int k, const int m) {
    while (k--) memcpy(c + k * m, x + sn[k] * m, m * sizeof(double));
}

/* --- Parallelized functions using pthreads --- */

/* 1. Parallel Partitioning (used in detStartPartition)
   Each thread computes cluster assignments for a slice of points
   and accumulates local counts for each cluster. */
struct PartitionArgs {
    const double *x;
    const double *c;
    int *y;
    int start;
    int end;
    int m;
    int k;
    int *local_counts; /* Array of length k allocated per thread */
};

void* partition_worker(void *arg) {
    struct PartitionArgs *args = (struct PartitionArgs*) arg;
    for (int i = args->start; i < args->end; i++) {
        int cluster = getCluster(args->x + i * args->m, args->c, args->m, args->k);
        args->y[i] = cluster;
        args->local_counts[cluster]++;
    }
    return NULL;
}

void parallel_detStartPartition(const double* x, const double* c, int* y, int* nums, int n, int m, int k) {
    pthread_t threads[NUM_THREADS];
    struct PartitionArgs args[NUM_THREADS];
    int local_counts[NUM_THREADS][k];
    for (int t = 0; t < NUM_THREADS; t++) {
        for (int j = 0; j < k; j++) {
            local_counts[t][j] = 0;
        }
    }
    int chunk = n / NUM_THREADS;
    int remainder = n % NUM_THREADS;
    int start = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        int end = start + chunk + (t < remainder ? 1 : 0);
        args[t].x = x;
        args[t].c = c;
        args[t].y = y;
        args[t].start = start;
        args[t].end = end;
        args[t].m = m;
        args[t].k = k;
        args[t].local_counts = local_counts[t];
        pthread_create(&threads[t], NULL, partition_worker, &args[t]);
        start = end;
    }
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
    /* Combine the per-thread counts */
    for (int j = 0; j < k; j++) {
        nums[j] = 0;
        for (int t = 0; t < NUM_THREADS; t++) {
            nums[j] += local_counts[t][j];
        }
    }
}

/* 2. Parallel CheckPartition
   Each thread reassigns points and sets a flag if any assignment changes.
   Local counts for each cluster are also computed. */
struct CheckArgs {
    const double *x;
    const double *c;
    int *y;
    int start;
    int end;
    int m;
    int k;
    int *local_counts; /* Array of length k */
    int local_flag;    /* Set to 1 if any change is detected */
};

void* check_worker(void *arg) {
    struct CheckArgs *args = (struct CheckArgs*) arg;
    args->local_flag = 0;
    for (int i = args->start; i < args->end; i++) {
        int new_cluster = getCluster(args->x + i * args->m, args->c, args->m, args->k);
        if (args->y[i] != new_cluster)
            args->local_flag = 1;
        args->y[i] = new_cluster;
        args->local_counts[new_cluster]++;
    }
    return NULL;
}

int parallel_checkPartition(const double* x, const double* c, int* y, int* nums, int n, int m, int k) {
    pthread_t threads[NUM_THREADS];
    struct CheckArgs args[NUM_THREADS];
    int local_counts[NUM_THREADS][k];
    for (int t = 0; t < NUM_THREADS; t++) {
        for (int j = 0; j < k; j++) {
            local_counts[t][j] = 0;
        }
        args[t].local_flag = 0;
    }
    int chunk = n / NUM_THREADS;
    int remainder = n % NUM_THREADS;
    int start = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        int end = start + chunk + (t < remainder ? 1 : 0);
        args[t].x = x;
        args[t].c = c;
        args[t].y = y;
        args[t].start = start;
        args[t].end = end;
        args[t].m = m;
        args[t].k = k;
        args[t].local_counts = local_counts[t];
        pthread_create(&threads[t], NULL, check_worker, &args[t]);
        start = end;
    }
    int flag = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
        if (args[t].local_flag)
            flag = 1;
    }
    for (int j = 0; j < k; j++) {
        nums[j] = 0;
        for (int t = 0; t < NUM_THREADS; t++) {
            nums[j] += local_counts[t][j];
        }
    }
    return flag;
}

/* 3. Parallel calcCores
   Each thread sums contributions from a subset of points into a local sum array.
   Then the local arrays are combined into the global centroids.
   Finally, each cluster sum is divided by its count.
*/
struct CalcArgs {
    const double *x;
    const int *y;
    int start;
    int end;
    int m;
    int k;
    double *local_sum; /* Array of size k*m (initialized to 0) */
};

void* calc_worker(void *arg) {
    struct CalcArgs *args = (struct CalcArgs*) arg;
    for (int i = args->start; i < args->end; i++) {
        int cluster = args->y[i];
        int offset = cluster * args->m;
        int point_offset = i * args->m;
        for (int j = 0; j < args->m; j++) {
            args->local_sum[offset + j] += args->x[point_offset + j];
        }
    }
    return NULL;
}

void parallel_calcCores(const double* x, double* c, const int* y, const int* nums, int n, int m, int k) {
    /* Zero out global centroids */
    memset(c, 0, k * m * sizeof(double));
    pthread_t threads[NUM_THREADS];
    struct CalcArgs args[NUM_THREADS];
    double* local_sums[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        local_sums[t] = (double*) calloc(k * m, sizeof(double));
    }
    int chunk = n / NUM_THREADS;
    int remainder = n % NUM_THREADS;
    int start = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        int end = start + chunk + (t < remainder ? 1 : 0);
        args[t].x = x;
        args[t].y = y;
        args[t].start = start;
        args[t].end = end;
        args[t].m = m;
        args[t].k = k;
        args[t].local_sum = local_sums[t];
        pthread_create(&threads[t], NULL, calc_worker, &args[t]);
        start = end;
    }
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
    /* Combine local sums into global centroids */
    for (int t = 0; t < NUM_THREADS; t++) {
        for (int i = 0; i < k * m; i++) {
            c[i] += local_sums[t][i];
        }
        free(local_sums[t]);
    }
    /* Divide each cluster sum by its count */
    for (int i = 0; i < k; i++) {
        int count = nums[i];
        int offset = i * m;
        for (int j = 0; j < m; j++) {
            c[offset + j] /= count;
        }
    }
}

/* 4. Parallel setMedoids
   Each thread processes a subset of the clusters and, for each, finds the medoid.
*/
struct SetMedoidsArgs {
    const double *x;
    int n;
    int m;
    const int *y;
    int start_cluster;
    int end_cluster;
    double *c;
};

void* setMedoids_worker(void *arg) {
    struct SetMedoidsArgs *args = (struct SetMedoidsArgs*) arg;
    for (int cl = args->start_cluster; cl < args->end_cluster; cl++) {
        int medoid = getMedoidNum(args->x, args->c + cl * args->m, args->y, args->n, args->m, cl);
        memcpy(args->c + cl * args->m, args->x + medoid * args->m, args->m * sizeof(double));
    }
    return NULL;
}

void parallel_setMedoids(const double *x, double* c, const int* y, int n, int m, int k) {
    pthread_t threads[NUM_THREADS];
    struct SetMedoidsArgs args[NUM_THREADS];
    int chunk = k / NUM_THREADS;
    int remainder = k % NUM_THREADS;
    int start_cluster = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        int end_cluster = start_cluster + chunk + (t < remainder ? 1 : 0);
        args[t].x = x;
        args[t].n = n;
        args[t].m = m;
        args[t].y = y;
        args[t].start_cluster = start_cluster;
        args[t].end_cluster = end_cluster;
        args[t].c = c;
        pthread_create(&threads[t], NULL, setMedoids_worker, &args[t]);
        start_cluster = end_cluster;
    }
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
}

/* --- cyclicRecalc: uses the parallel functions above ---
     It recalculates the cores, resets the medoids,
     and then checks the partition in parallel. */
char cyclicRecalc(const double* x, double* c, int* y, int* nums, int n, int m, int k) {
    parallel_calcCores(x, c, y, nums, n, m, k);
    parallel_setMedoids(x, c, y, n, m, k);
    return parallel_checkPartition(x, c, y, nums, n, m, k);
}

/* --- The remaining (serial) functions remain unchanged --- */

/* Copy medoid from the cluster with minimum distance */
int getMedoidNum(const double* const x, const double* const c, const int* const y, const int n, const int m, const int id) {
    int i, res = 0;
    while (res < n && y[res] != id) res++;
    if (res == n) {
        /* No point found for this cluster, fallback to index 0 */
        return 0;
    }
    double minD = getDistance(x + res * m, c, m);
    for (i = res + 1; i < n; i++) {
        if (y[i] == id) {
            const double curD = getDistance(x + i * m, c, m);
            if (curD < minD) {
                minD = curD;
                res = i;
            }
        }
    }
    return res;
}

/* --- The main kmedoids function ---
     It copies the input data, autos-scales, initializes cores,
     uses the parallel partitioning, and then iterates until convergence.
     A checkpoint is printed every 10 iterations similar to your kmeans implementation.
*/
void kmedoids(const double* const X, int* const y, const int n, const int m, const int k) {
    double *x = (double*)malloc(n * m * sizeof(double));
    memcpy(x, X, n * m * sizeof(double));
    autoscaling(x, n, m);
    int *nums = startCoreNums(k, n);
    double *c = (double*)malloc(k * m * sizeof(double));
    detCores(x, c, nums, k, m);
    parallel_detStartPartition(x, c, y, nums, n, m, k);

    int *y_last = (int*)malloc(n * sizeof(int));
    memcpy(y_last, y, n * sizeof(int));
    int iter = 0;
    int same_change_count = 0;
    int prev_changes = -1;

    while (cyclicRecalc(x, c, y, nums, n, m, k)) {
        iter++;
        int changes = 0;
        for (int i = 0; i < n; i++) {
            if (y_last[i] != y[i])
                changes++;
        }
        if (iter % 10 == 0) {
            printf("Iteration %d completed: %d changes\n", iter, changes);
            fflush(stdout);
        }
        if (changes == prev_changes) {
            same_change_count++;
        } else {
            same_change_count = 0;
            prev_changes = changes;
        }
        if (same_change_count >= 20) {
            printf("No improvement over 20 iterations. Breaking at iteration %d with %d changes.\n", iter, changes);
            fflush(stdout);
            break;
        }
        if (iter > 1000) {
            printf("Reached maximum iterations. Breaking at iteration %d with %d changes.\n", iter, changes);
            fflush(stdout);
            break;
        }
        memcpy(y_last, y, n * sizeof(int));
    }
    free(x);
    free(c);
    free(nums);
    free(y_last);
}

// sort vectors by area, then give number of centroids choose top k for that area space
// like psrs
// the data has three main categories of  data very large, large, and small 
