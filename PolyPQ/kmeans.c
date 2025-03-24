/*-------------------------------------------------------------------------
*
* kmeans.c
*    Generic k-means implementation
*
* Copyright (c) 2016, Paul Ramsey <pramsey@cleverelephant.ca>
*
*------------------------------------------------------------------------*/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "kmeans.h"

#ifdef KMEANS_THREADED
#include <pthread.h>
#endif

static void
update_r(kmeans_config *config)
{
	int i;

	for (i = 0; i < config->num_objs; i++)
	{
		double distance, curr_distance;
		int cluster, curr_cluster;
		Pointer obj;

		assert(config->objs != NULL);
		assert(config->num_objs > 0);
		assert(config->centers);
		assert(config->clusters);

		obj = config->objs[i];

		/*
		* Don't try to cluster NULL objects, just add them
		* to the "unclusterable cluster"
		*/
		if (!obj)
		{
			config->clusters[i] = KMEANS_NULL_CLUSTER;
			continue;
		}

		/* Initialize with distance to first cluster */
		curr_distance = (config->distance_method)(obj, config->centers[0]);
		curr_cluster = 0;

		/* Check all other cluster centers and find the nearest */
		for (cluster = 1; cluster < config->k; cluster++)
		{
			distance = (config->distance_method)(obj, config->centers[cluster]);
			if (distance < curr_distance)
			{
				curr_distance = distance;
				curr_cluster = cluster;
			}
		}

		/* Store the nearest cluster this object is in */
		config->clusters[i] = curr_cluster;
	}
}

static void
update_means(kmeans_config *config)
{
	int i;

	for (i = 0; i < config->k; i++)
	{
		/* Update the centroid for this cluster */
		(config->centroid_method)(config->objs, config->clusters, config->num_objs, i, config->centers[i]);
	}
}

#ifdef KMEANS_THREADED

static void * update_r_threaded_main(void *args)
{
	kmeans_config *config = (kmeans_config*)args;
	update_r(config);
	pthread_exit(args);
}

static void update_r_threaded(kmeans_config *config)
{
	/* Computational complexity is function of objs/clusters */
	/* We only spin up threading infra if we need more than one core */
	/* running. We keep the threshold high so the overhead of */
	/* thread management is small compared to thread compute time */
	int num_threads = config->num_objs * config->k / KMEANS_THR_THRESHOLD;

	/* Can't run more threads than the maximum */
	num_threads = (num_threads > KMEANS_THR_MAX ? KMEANS_THR_MAX : num_threads);

	/* If the problem size is small, don't bother w/ threading */
	if (num_threads < 1)
	{
		update_r(config);
	}
	else
	{
		pthread_t thread[KMEANS_THR_MAX];
		pthread_attr_t thread_attr;
		kmeans_config thread_config[KMEANS_THR_MAX];
		int obs_per_thread = config->num_objs / num_threads;
		int i, rc;

		for (i = 0; i < num_threads; i++)
		{
			/*
			* Each thread gets a copy of the config, but with the list pointers
			* offest to the start of the batch the thread is responsible for, and the
			* object count number adjusted similarly.
			*/
			memcpy(&(thread_config[i]), config, sizeof(kmeans_config));
			thread_config[i].objs += i*obs_per_thread;
			thread_config[i].clusters += i*obs_per_thread;
			thread_config[i].num_objs = obs_per_thread;
			if (i == num_threads-1)
			{
				thread_config[i].num_objs += config->num_objs - num_threads*obs_per_thread;
			}

			/* Initialize and set thread detached attribute */
			pthread_attr_init(&thread_attr);
			pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

			/* Now we just run the thread, on its subset of the data */
			rc = pthread_create(&thread[i], &thread_attr, update_r_threaded_main, (void *) &thread_config[i]);
			if (rc)
			{
				printf("ERROR: return code from pthread_create() is %d\n", rc);
				exit(-1);
			}
		}

		/* Free attribute and wait for the other threads */
		pthread_attr_destroy(&thread_attr);

		/* Wait for all calculations to complete */
		for (i = 0; i < num_threads; i++)
		{
		    void *status;
			rc = pthread_join(thread[i], &status);
			if (rc)
			{
				printf("ERROR: return code from pthread_join() is %d\n", rc);
				exit(-1);
			}
		}
	}
}

int update_means_k;
pthread_mutex_t update_means_k_mutex;

static void *
update_means_threaded_main(void *arg)
{
	kmeans_config *config = (kmeans_config*)arg;
	int i = 0;

	do
	{
		pthread_mutex_lock (&update_means_k_mutex);
		i = update_means_k;
		update_means_k++;
		pthread_mutex_unlock (&update_means_k_mutex);

		if (i < config->k)
			(config->centroid_method)(config->objs, config->clusters, config->num_objs, i, config->centers[i]);
	}
	while (i < config->k);

	pthread_exit(arg);
}

static void
update_means_threaded(kmeans_config *config)
{
	/* We only spin up threading infra if we need more than one core */
	/* running. We keep the threshold high so the overhead of */
	/* thread management is small compared to thread compute time */
	int num_threads = config->num_objs / KMEANS_THR_THRESHOLD;

	/* Can't run more threads than the maximum */
	num_threads = (num_threads > KMEANS_THR_MAX ? KMEANS_THR_MAX : num_threads);

	/* If the problem size is small, don't bother w/ threading */
	if (num_threads < 1)
	{
		update_means(config);
	}
	else
	{
		/* Mutex protected counter to drive threads */
		pthread_t thread[KMEANS_THR_MAX];
		pthread_attr_t thread_attr;
		int i, rc;

		pthread_mutex_init(&update_means_k_mutex, NULL);
		update_means_k = 0;

		pthread_attr_init(&thread_attr);
		pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

		/* Create threads to perform computation  */
		for (i = 0; i < num_threads; i++)
		{

			/* Now we just run the thread, on its subset of the data */
			rc = pthread_create(&thread[i], &thread_attr, update_means_threaded_main, (void *) config);
			if (rc)
			{
				printf("ERROR: return code from pthread_create() is %d\n", rc);
				exit(-1);
			}
		}

		pthread_attr_destroy(&thread_attr);

		/* Watch until completion  */
		for (i = 0; i < num_threads; i++)
		{
		    void *status;
			rc = pthread_join(thread[i], &status);
			if (rc)
			{
				printf("ERROR: return code from pthread_join() is %d\n", rc);
				exit(-1);
			}
		}

		pthread_mutex_destroy(&update_means_k_mutex);
	}
}

#endif /* KMEANS_THREADED */

kmeans_result
kmeans(kmeans_config *config)
{
    int iterations = 0;
    int *clusters_last;
    size_t clusters_sz = sizeof(int) * config->num_objs;
    int i;
    const double CHANGE_THRESHOLD_PERCENT = 0;  /* 10% of points */
    int change_threshold = (int)(CHANGE_THRESHOLD_PERCENT * config->num_objs);
    int same_change_count = 0;    /* Count how many iterations have the same number of changes */
    int prev_changes = -1;        /* Store number of changes from previous iteration */

    assert(config);
    assert(config->objs);
    assert(config->num_objs);
    assert(config->distance_method);
    assert(config->centroid_method);
    assert(config->centers);
    assert(config->clusters);
    assert(config->k);
    assert(config->k <= config->num_objs);

    /* Zero out cluster numbers */
    memset(config->clusters, 0, clusters_sz);

    if (!config->max_iterations)
        config->max_iterations = KMEANS_MAX_ITERATIONS;

    clusters_last = kmeans_malloc(clusters_sz);
    if (!clusters_last) {
        return KMEANS_ERROR;
    }

    while (1)
    {
        memcpy(clusters_last, config->clusters, clusters_sz);

#ifdef KMEANS_THREADED
        update_r_threaded(config);
        update_means_threaded(config);
#else
        update_r(config);
        update_means(config);
#endif

        iterations++;

        /* Count how many assignments changed */
        int changes = 0;
        for (i = 0; i < config->num_objs; i++) {
            if (clusters_last[i] != config->clusters[i])
                changes++;
        }

        /* Print progress every 10 iterations */
        if (iterations % 10 == 0) {
            printf("Iteration %d completed: %d changes\n", iterations, changes);
            fflush(stdout);
        }

        /* Check for standard convergence if few changes */
        if (changes <= change_threshold) {
            kmeans_free(clusters_last);
            config->total_iterations = iterations;
            return KMEANS_OK;
        }

        /* Check if the number of changes has stayed the same over multiple iterations */
        if (changes == prev_changes) {
            same_change_count++;
        } else {
            same_change_count = 0;
            prev_changes = changes;
        }

        /* If the changes are the same for, say, 5 consecutive iterations, break out */
        if (same_change_count >= 20) {
            printf("No improvement in changes over 20 iterations. Breaking out at iteration %d with %d changes.\n",
                   iterations, changes);
            fflush(stdout);
            kmeans_free(clusters_last);
            config->total_iterations = iterations;
            return KMEANS_OK;
        }

        if (iterations > config->max_iterations) {
            kmeans_free(clusters_last);
            config->total_iterations = iterations;
            return KMEANS_OK;
        }
    }

    kmeans_free(clusters_last);
    config->total_iterations = iterations;
    return KMEANS_ERROR;
}
