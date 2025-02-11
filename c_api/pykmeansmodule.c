/* c_api/pykmeansmodule.c */
#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kmeans.h"

/* Global variable for the dimensionality.
   (All points and centers are assumed to have the same dimension.) */
static int dim = 0;

/* --- Distance functions --- */

/* Euclidean distance function.
   Assumes that each Pointer points to an array of doubles of length 'dim'. */
double euclidean_distance(const Pointer a, const Pointer b) {
    const double *p1 = (const double *) a;
    const double *p2 = (const double *) b;
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double d = p1[i] - p2[i];
        sum += d * d;
    }
    return sqrt(sum);
}

/* Jaccard distance function.
   Here we assume the points are represented as binary vectors (0/1 values)
   or that nonzero values indicate presence. The function computes the
   Jaccard similarity (intersection over union) and returns 1 - similarity. */
double jaccard_distance(const Pointer a, const Pointer b) {
    const double *p1 = (const double *) a;
    const double *p2 = (const double *) b;
    int intersection = 0;
    int union_count = 0;
    for (int i = 0; i < dim; i++) {
        int v1 = (p1[i] != 0.0);
        int v2 = (p2[i] != 0.0);
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

/* --- Centroid (mean) calculation function --- */

/* This function computes the arithmetic mean of all points assigned to a cluster.
   It assumes each 'centroid' is a pre-allocated array of 'dim' doubles. */
void compute_centroid(const Pointer *objs, const int *clusters, size_t num_objs,
                      int cluster, Pointer centroid) {
    double *cent = (double *) centroid;
    int count = 0;
    /* Zero out the centroid */
    for (int i = 0; i < dim; i++) {
        cent[i] = 0.0;
    }
    for (size_t i = 0; i < num_objs; i++) {
        if (clusters[i] == cluster) {
            double *pt = (double *) objs[i];
            for (int j = 0; j < dim; j++) {
                cent[j] += pt[j];
            }
            count++;
        }
    }
    if (count > 0) {
        for (int j = 0; j < dim; j++) {
            cent[j] /= count;
        }
    }
}

/* --- Python wrapper for the kmeans() function --- */

/*
 * Python function: kmeans(points, centers, max_iterations=0, metric="euclidean")
 *
 *   points  : A list of points, where each point is a sequence of numbers.
 *   centers : A list of initial centers (each a sequence of numbers).
 *   max_iterations : Optional maximum number of iterations (default uses KMEANS_MAX_ITERATIONS).
 *   metric  : Optional string ("euclidean" or "jaccard") to choose the distance function.
 *
 * Returns a list of integers representing the cluster assignment for each point.
 */
static PyObject * py_kmeans(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *points_list;
    PyObject *centers_list;
    int max_iterations = 0;  /* if zero, kmeans() will use its default */
    const char *metric = "euclidean";

    static char *kwlist[] = {"points", "centers", "max_iterations", "metric", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|is", kwlist,
                                     &points_list, &centers_list, &max_iterations, &metric)) {
        return NULL;
    }
    
    if (!PyList_Check(points_list) || !PyList_Check(centers_list)) {
        PyErr_SetString(PyExc_TypeError, "Both points and centers must be Python lists.");
        return NULL;
    }
    
    Py_ssize_t num_points = PyList_Size(points_list);
    Py_ssize_t k = PyList_Size(centers_list);
    if (num_points == 0 || k == 0) {
        PyErr_SetString(PyExc_ValueError, "Points and centers lists must not be empty.");
        return NULL;
    }
    
    /* Determine the dimension from the first point */
    PyObject *first_point = PyList_GetItem(points_list, 0);
    if (!PySequence_Check(first_point)) {
        PyErr_SetString(PyExc_TypeError, "Each point must be a sequence (e.g., list or tuple).");
        return NULL;
    }
    dim = (int) PySequence_Size(first_point);
    
    /* Allocate arrays for the points, centers, and cluster assignments */
    Pointer *objs = (Pointer*) malloc(num_points * sizeof(Pointer));
    Pointer *centers = (Pointer*) malloc(k * sizeof(Pointer));
    int *clusters = (int*) malloc(num_points * sizeof(int));
    if (!objs || !centers || !clusters) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for kmeans arrays.");
        goto cleanup_fail;
    }
    
    /* Convert the points list into C arrays (each point is allocated separately) */
    for (Py_ssize_t i = 0; i < num_points; i++) {
        PyObject *pt_obj = PyList_GetItem(points_list, i);
        if (!PySequence_Check(pt_obj) || PySequence_Size(pt_obj) != dim) {
            PyErr_SetString(PyExc_ValueError, "Each point must be a sequence of numbers with the same dimension.");
            goto cleanup_fail;
        }
        double *pt = (double*) malloc(dim * sizeof(double));
        if (!pt) {
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for a point.");
            goto cleanup_fail;
        }
        for (int j = 0; j < dim; j++) {
            PyObject *num = PySequence_GetItem(pt_obj, j);
            if (!PyNumber_Check(num)) {
                PyErr_SetString(PyExc_TypeError, "Point coordinates must be numbers.");
                Py_DECREF(num);
                free(pt);
                goto cleanup_fail;
            }
            pt[j] = PyFloat_AsDouble(num);
            Py_DECREF(num);
        }
        objs[i] = pt;
    }
    
    /* Convert the centers list into C arrays */
    for (Py_ssize_t i = 0; i < k; i++) {
        PyObject *ct_obj = PyList_GetItem(centers_list, i);
        if (!PySequence_Check(ct_obj) || PySequence_Size(ct_obj) != dim) {
            PyErr_SetString(PyExc_ValueError, "Each center must be a sequence of numbers with the same dimension.");
            goto cleanup_fail;
        }
        double *ct = (double*) malloc(dim * sizeof(double));
        if (!ct) {
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for a center.");
            goto cleanup_fail;
        }
        for (int j = 0; j < dim; j++) {
            PyObject *num = PySequence_GetItem(ct_obj, j);
            if (!PyNumber_Check(num)) {
                PyErr_SetString(PyExc_TypeError, "Center coordinates must be numbers.");
                Py_DECREF(num);
                free(ct);
                goto cleanup_fail;
            }
            ct[j] = PyFloat_AsDouble(num);
            Py_DECREF(num);
        }
        centers[i] = ct;
    }
    
    /* Set up the kmeans_config structure and choose the distance method based on the metric argument */
    kmeans_config config;
    if (strcmp(metric, "jaccard") == 0) {
        config.distance_method = jaccard_distance;
    } else {
        config.distance_method = euclidean_distance;
    }
    config.centroid_method = compute_centroid;
    config.objs = objs;
    config.num_objs = num_points;
    config.centers = centers;
    config.k = (unsigned int) k;
    config.max_iterations = (max_iterations > 0) ? max_iterations : KMEANS_MAX_ITERATIONS;
    config.clusters = clusters;
    
    /* Run the kmeans algorithm */
    kmeans_result res = kmeans(&config);
    
    /* Convert the resulting cluster assignments to a Python list */
    PyObject *result_list = PyList_New(num_points);
    for (Py_ssize_t i = 0; i < num_points; i++) {
        PyObject *cluster_num = PyLong_FromLong(clusters[i]);
        PyList_SetItem(result_list, i, cluster_num);
    }
    
    /* Free the allocated memory */
    for (Py_ssize_t i = 0; i < num_points; i++) {
        free(objs[i]);
    }
    for (Py_ssize_t i = 0; i < k; i++) {
        free(centers[i]);
    }
    free(objs);
    free(centers);
    free(clusters);
    
    if (res == KMEANS_OK)
        return result_list;
    else if (res == KMEANS_EXCEEDED_MAX_ITERATIONS) {
        PyErr_SetString(PyExc_RuntimeError, "K-means exceeded maximum iterations.");
        Py_DECREF(result_list);
        return NULL;
    } else {
        PyErr_SetString(PyExc_RuntimeError, "K-means encountered an error.");
        Py_DECREF(result_list);
        return NULL;
    }
    
cleanup_fail:
    if (objs) {
        for (Py_ssize_t i = 0; i < num_points; i++) {
            if (objs[i])
                free(objs[i]);
        }
        free(objs);
    }
    if (centers) {
        for (Py_ssize_t i = 0; i < k; i++) {
            if (centers[i])
                free(centers[i]);
        }
        free(centers);
    }
    if (clusters)
        free(clusters);
    return NULL;
}

/* Module method table */
static PyMethodDef polyPQ_methods[] = {
    {"kmeans", (PyCFunction)py_kmeans, METH_VARARGS | METH_KEYWORDS,
     "Perform k-means clustering with selectable distance metric.\n\n"
     "Parameters:\n"
     "  points (list of sequences): Data points (each a sequence of numbers).\n"
     "  centers (list of sequences): Initial centers.\n"
     "  max_iterations (int, optional): Maximum iterations (default is library default).\n"
     "  metric (str, optional): 'euclidean' or 'jaccard'. Defaults to 'euclidean'.\n\n"
     "Returns:\n"
     "  list: Cluster assignments (an integer per point)."},
    {NULL, NULL, 0, NULL}
};

/* Module definition structure with the new module name "PolyPQ" */
static struct PyModuleDef polyPQ_module = {
    PyModuleDef_HEAD_INIT,
    "PolyPQ",  /* New module name */
    "A Python interface to the product quantization library including k-means clustering.",
    -1,
    polyPQ_methods
};

/* Module initialization function renamed accordingly */
PyMODINIT_FUNC PyInit_PolyPQ(void) {
    return PyModule_Create(&polyPQ_module);
}