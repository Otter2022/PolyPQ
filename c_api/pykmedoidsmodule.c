/* pykmedoidsmodule.c */
#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kmedoids.h"

/* --- Alternative distance functions --- */

double weighted_jaccard_distance(const double *x1, const double *x2, int m) {
    double min_sum = 0.0;
    double max_sum = 0.0;
    for (int i = 0; i < m; i++) {
        double a = x1[i];
        double b = x2[i];
        min_sum += (a < b) ? a : b;
        max_sum += (a > b) ? a : b;
    }
    if (max_sum == 0.0)
        return 0.0;
    double similarity = min_sum / max_sum;
    return 1.0 - similarity;
}

double jaccard_distance(const double *x1, const double *x2, int m) {
    int intersection = 0;
    int union_count = 0;
    for (int i = 0; i < m; i++) {
        int v1 = (x1[i] != 0.0);
        int v2 = (x2[i] != 0.0);
        if (v1 || v2) {
            union_count++;
            if (v1 && v2)
                intersection++;
        }
    }
    if (union_count == 0)
        return 0.0;
    double similarity = (double)intersection / union_count;
    return 1.0 - similarity;
}

/*
 * Python function: kmedoids(points, k, metric="euclidean")
 *
 *   points : A list of points, where each point is a sequence of numbers.
 *   k      : The desired number of clusters.
 *   metric : Optional string ("euclidean", "jaccard", or "weighted_jaccard")
 *            to choose the distance function (defaults to "euclidean").
 *
 * Returns:
 *   A Python list of integers representing the cluster assignment for each point.
 */
static PyObject * py_kmedoids(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *points_list;
    int k;
    const char *metric = "euclidean";

    static char *kwlist[] = {"points", "k", "metric", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|s", kwlist,
                                     &points_list, &k, &metric)) {
        return NULL;
    }
    if (!PyList_Check(points_list)) {
        PyErr_SetString(PyExc_TypeError, "Points must be provided as a list of sequences.");
        return NULL;
    }
    Py_ssize_t n = PyList_Size(points_list);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "Points list must not be empty.");
        return NULL;
    }
    
    /* Determine the dimensionality from the first point */
    PyObject *first_point = PyList_GetItem(points_list, 0);
    if (!PySequence_Check(first_point)) {
        PyErr_SetString(PyExc_TypeError, "Each point must be a sequence (list, tuple, etc.).");
        return NULL;
    }
    Py_ssize_t m = PySequence_Size(first_point);
    
    /* Allocate a contiguous C array for points (row-major order) */
    double *X = (double*) malloc(n * m * sizeof(double));
    if (!X) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for points array.");
        return NULL;
    }
    
    /* Convert the Python list of points into the C array */
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *pt_obj = PyList_GetItem(points_list, i);
        if (!PySequence_Check(pt_obj) || PySequence_Size(pt_obj) != m) {
            PyErr_SetString(PyExc_ValueError, "Each point must be a sequence of numbers with the same dimension.");
            free(X);
            return NULL;
        }
        for (Py_ssize_t j = 0; j < m; j++) {
            PyObject *num = PySequence_GetItem(pt_obj, j);
            if (!PyNumber_Check(num)) {
                PyErr_SetString(PyExc_TypeError, "Point coordinates must be numbers.");
                Py_DECREF(num);
                free(X);
                return NULL;
            }
            double value = PyFloat_AsDouble(num);
            Py_DECREF(num);
            X[i * m + j] = value;
        }
    }
    
    /* Validate k (must be between 1 and the number of points) */
    if (k <= 0 || k > n) {
        PyErr_SetString(PyExc_ValueError, "k must be between 1 and the number of points.");
        free(X);
        return NULL;
    }
    
    /* Set the global distance function pointer based on the metric */
    if (strcmp(metric, "jaccard") == 0) {
        kmedoids_distance = jaccard_distance;
    } else if (strcmp(metric, "weighted_jaccard") == 0) {
        kmedoids_distance = weighted_jaccard_distance;
    } else {
        kmedoids_distance = NULL; // defaults to Euclidean distance
    }
    
    /* Allocate an array for the resulting cluster assignments */
    int *y = (int*) malloc(n * sizeof(int));
    if (!y) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for cluster assignments.");
        free(X);
        return NULL;
    }
    
    /* Run the k-medoids algorithm */
    kmedoids(X, y, (int)n, (int)m, k);
    
    /* Reset the distance function pointer to default */
    kmedoids_distance = NULL;
    
    /* Build a Python list to hold the result */
    PyObject *result = PyList_New(n);
    if (!result) {
        free(X);
        free(y);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *cluster_num = PyLong_FromLong(y[i]);
        PyList_SetItem(result, i, cluster_num);  // Steals the reference
    }
    
    free(X);
    free(y);
    return result;
}

static PyMethodDef pykmedoids_methods[] = {
    {"kmedoids", (PyCFunction) py_kmedoids, METH_VARARGS | METH_KEYWORDS,
     "Perform k-medoids clustering on a list of points with selectable distance metric.\n\n"
     "Parameters:\n"
     "  points (list of sequences): Data points, where each point is a sequence of numbers.\n"
     "  k (int): The number of clusters to form.\n"
     "  metric (str, optional): 'euclidean', 'jaccard', or 'weighted_jaccard'. Defaults to 'euclidean'.\n\n"
     "Returns:\n"
     "  list: Cluster assignments (an integer per point)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pykmedoids_module = {
    PyModuleDef_HEAD_INIT,
    "pykmedoidsmodule",  /* Module name */
    "A Python interface for the k-medoids clustering algorithm with selectable distance metrics.",
    -1,
    pykmedoids_methods
};

PyMODINIT_FUNC PyInit_pykmedoidsmodule(void) {
    return PyModule_Create(&pykmedoids_module);
}
