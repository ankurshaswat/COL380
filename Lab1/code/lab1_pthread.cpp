#include "helper.h"
#include "point_with_cluster.h"
#include <iostream>
#include <pthread.h>
#include <vector>

using namespace std;

int count = 0;
int work_type = -1;
int NUM_THREADS;
int num_points;
int num_changes = 0;
int num_iters = 0;
bool work = true;
int num_centroids;
int **points_cluster;
int *data_points_loc;
float **centroids_loc;
vector<vector<point_with_cluster *>> thread_aggregates;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *aggregrate_points(void *in) {
  int tid = (long)in;

for (int i = 0; i < num_centroids; i++) {
        thread_aggregates[tid][i]->reset();
      }

      int length = num_points;
      int num_of_items_to_process = ceil((1.0 * length) / NUM_THREADS);
      for (int i = tid * num_of_items_to_process;
           i < (tid + 1) * num_of_items_to_process && i < length; i++) {
        thread_aggregates[tid][(*points_cluster)[4 * i + 3]]->add_to_point(
            (*points_cluster)[4 * i], (*points_cluster)[4 * i + 1],
            (*points_cluster)[4 * i + 2]);
      }
  return NULL;
}

void *find_centroid(void *in) {
  int tid = (long)in;

  int centroid_number = tid;
      while (centroid_number < num_centroids) {

        for (int i = 1; i < NUM_THREADS; i++) {
          thread_aggregates[0][centroid_number]->accumulate_values(
              thread_aggregates[i][centroid_number]);
        }
        thread_aggregates[0][centroid_number]->average_out_point();

        centroid_number += NUM_THREADS;
  }

  return NULL;
}

void *assign_centroids(void *in) {
  int tid = (long)in;

  int local_changes = 0;

      int length = num_points;
      int num_of_items_to_process = ceil((1.0 * length) / NUM_THREADS);
      for (int i = tid * num_of_items_to_process;
           i < (tid + 1) * num_of_items_to_process && i < length; i++) {

        double min_dist = __INT_MAX__;
        int closest_centroid = -1;
        int offset = (num_iters - 1) * num_centroids * 3;
        for (int j = 0; j < num_centroids; j++) {
          float dist = distance(data_points_loc, i, *centroids_loc, j, offset);

          if (dist < min_dist) {
            min_dist = dist;
            closest_centroid = j;
          }
        }

        if (closest_centroid != (*points_cluster)[4 * i + 3]) {
          (*points_cluster)[4 * i + 3] = closest_centroid;
          local_changes++;
        }
      }

  pthread_mutex_lock(&mutex);
  num_changes += local_changes;
  pthread_mutex_unlock(&mutex);

  return NULL;
}

void *initialize(void *in) {
  int tid = (long)in;

    int length = num_points;
      int num_of_items_to_process = ceil((1.0 * length) / NUM_THREADS);
      for (int i = tid * num_of_items_to_process;
           i < (tid + 1) * num_of_items_to_process && i < length; i++) {
        (*points_cluster)[4 * i] = data_points_loc[3 * i];
        (*points_cluster)[4 * i + 1] = data_points_loc[3 * i + 1];
        (*points_cluster)[4 * i + 2] = data_points_loc[3 * i + 2];
        (*points_cluster)[4 * i + 3] = rand() % num_centroids;
      }

  return NULL;
}


void kmeans_pthread(int num_threads, int N, int K, int *data_points,
                    int **data_point_cluster, float **centroids,
                    int *num_iterations) {
  int MAX_ITERS = 100;
  NUM_THREADS = num_threads;
  num_centroids = K;
  num_points = N;
  data_points_loc = data_points;

  for (int i = 0; i < NUM_THREADS; i++) {
    vector<point_with_cluster *> thread_aggr;
    for (int j = 0; j < K; j++) {
      point_with_cluster *point = new point_with_cluster;
      thread_aggr.push_back(point);
    }
    thread_aggregates.push_back(thread_aggr);
  }

  *data_point_cluster = (int *)malloc(sizeof(int) * (num_points)*4);
  *centroids = (float *)malloc(sizeof(float) * ((MAX_ITERS + 1) * K) * 3);
  points_cluster = data_point_cluster;
  centroids_loc = centroids;

  int return_code;
  pthread_t threads[NUM_THREADS];

  srand(1);
    // Accumulating points values
    for (int i = 0; i < NUM_THREADS; i++) {
      return_code =
          pthread_create(&threads[i], NULL, initialize, (void *)i);
    }

    // Wait for computations to complete
    for (int i = 0; i < NUM_THREADS; i++) {
      pthread_join(threads[i], NULL);
    }

    // Accumulating points values
    for (int i = 0; i < NUM_THREADS; i++) {
      return_code =
          pthread_create(&threads[i], NULL, aggregrate_points, (void *)i);
    }

    // Wait for computations to complete
    for (int i = 0; i < NUM_THREADS; i++) {
      pthread_join(threads[i], NULL);
    }

    // Find new Centroids by averaging points values
    for (int i = 0; i < NUM_THREADS && i < K; i++) {
      return_code = pthread_create(&threads[i], NULL, find_centroid, (void *)i);
    }

    // Wait for computations to complete
    for (int i = 0; i < NUM_THREADS && i < K; i++) {
      pthread_join(threads[i], NULL);
    }

    

  int num_iters = 0;

  do {
    num_iters++;
    num_changes = 0;

    // Assign new closest centrois
    for (int i = 0; i < NUM_THREADS; i++) {
      return_code =
          pthread_create(&threads[i], NULL, assign_centroids, (void *)i);
    }

    // Wait for computations to complete
    for (int i = 0; i < NUM_THREADS; i++) {
      pthread_join(threads[i], NULL);
    }

    // Accumulating points values
    for (int i = 0; i < NUM_THREADS; i++) {
      return_code =
          pthread_create(&threads[i], NULL, aggregrate_points, (void *)i);
    }

    // Wait for computations to complete
    for (int i = 0; i < NUM_THREADS; i++) {
      pthread_join(threads[i], NULL);
    }

    // Find new Centroids by averaging points values
    for (int i = 0; i < NUM_THREADS && i < K; i++) {
      return_code = pthread_create(&threads[i], NULL, find_centroid, (void *)i);
    }

    // Wait for computations to complete
    for (int i = 0; i < NUM_THREADS && i < K; i++) {
      pthread_join(threads[i], NULL);
    }
    int offset = num_iters * K * 3;

      for (int j = 0; j < K; j++) {
          (*centroids)[offset + 3 * j] = thread_aggregates[0][j]->x;
          (*centroids)[offset + 3 * j + 1] = thread_aggregates[0][j]->y;
          (*centroids)[offset + 3 * j + 2] = thread_aggregates[0][j]->z;
        }



  } while (num_changes > 0.01 * num_points && num_iters < MAX_ITERS);

  *num_iterations = num_iters;
}