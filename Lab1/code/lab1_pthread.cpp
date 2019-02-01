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

void *thread_work(void *in) {
  int tid = (long)in;
  int prev_work_type = -1;

  while (work) {
    if (work_type == 0) {

      prev_work_type = 0;

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
      count++;
      pthread_mutex_unlock(&mutex);
    } else if (work_type == 1) {
      prev_work_type = 1;

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

      pthread_mutex_lock(&mutex);
      count++;
      pthread_mutex_unlock(&mutex);
    } else if (work_type == 2) {

      prev_work_type = 2;

      int centroid_number = tid;
      while (centroid_number < num_centroids) {

        for (int i = 1; i < NUM_THREADS; i++) {
          thread_aggregates[0][centroid_number]->accumulate_values(
              thread_aggregates[i][centroid_number]);
        }
        thread_aggregates[0][centroid_number]->average_out_point();
        // centroids[centroid_number]->copy(thread_aggregates[0][centroid_number]);

        centroid_number += NUM_THREADS;
      }

      pthread_mutex_lock(&mutex);
      count++;
      pthread_mutex_unlock(&mutex);
    } else if (work_type == 3) {
      prev_work_type = 3;
      int length = num_points;
      int num_of_items_to_process = ceil((1.0 * length) / NUM_THREADS);
      for (int i = tid * num_of_items_to_process;
           i < (tid + 1) * num_of_items_to_process && i < length; i++) {
        (*points_cluster)[4 * i] = data_points_loc[3 * i];
        (*points_cluster)[4 * i + 1] = data_points_loc[3 * i + 1];
        (*points_cluster)[4 * i + 2] = data_points_loc[3 * i + 2];
        (*points_cluster)[4 * i + 3] = rand() % num_centroids;
      }

      pthread_mutex_lock(&mutex);
      count++;
      pthread_mutex_unlock(&mutex);
    }

    while (work_type == prev_work_type) {
    }
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

  // Creating Threads
  for (int i = 0; i < NUM_THREADS; i++) {
    return_code = pthread_create(&threads[i], NULL, thread_work, (void *)i);
  }

  work_type = 3;
  work = true;
  while (count < NUM_THREADS) {
  }

  count = 0;
  work_type = 1;
  while (count < NUM_THREADS) {
  }
  count = 0;
  work_type = 2;
  while (count < NUM_THREADS) {
  }

  for (int j = 0; j < K; j++) {
    (*centroids)[3 * j] = thread_aggregates[0][j]->x;
    (*centroids)[3 * j + 1] = thread_aggregates[0][j]->y;
    (*centroids)[3 * j + 2] = thread_aggregates[0][j]->z;
  }

  do {

    num_iters++;
    num_changes = 0;

    int offset = num_iters * K * 3;

    for (int i = 0; i < 3; i++) {
      count = 0;
      work_type = i;
      while (count < NUM_THREADS) {
      }

      if (work_type == 2) {
        for (int j = 0; j < K; j++) {
          (*centroids)[offset + 3 * j] = thread_aggregates[0][j]->x;
          (*centroids)[offset + 3 * j + 1] = thread_aggregates[0][j]->y;
          (*centroids)[offset + 3 * j + 2] = thread_aggregates[0][j]->z;
        }
      }
    }

  } while (num_changes > 0.01 * num_points && num_iters < MAX_ITERS);

  // Waiting for threads to close
  work_type = -1;
  work = false;
  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  *num_iterations = num_iters;
  pthread_exit(NULL);
}