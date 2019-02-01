#include "helper.h"
#include "point_with_cluster.h"
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

void kmeans_omp(int num_threads, int N, int K, int *data_points,
                int **data_point_cluster, float **centroids,
                int *num_iterations) {

  omp_set_num_threads(num_threads);

  int MAX_ITERS = 100;

  vector<vector<point_with_cluster *>> thread_aggregates;

  for (int i = 0; i < num_threads; i++) {
    vector<point_with_cluster *> thread_aggr;
    for (int j = 0; j < K; j++) {
      point_with_cluster *point = new point_with_cluster;
      thread_aggr.push_back(point);
    }
    thread_aggregates.push_back(thread_aggr);
  }

  *data_point_cluster = (int *)malloc(sizeof(int) * (N)*4);
  *centroids = (float *)malloc(sizeof(float) * ((MAX_ITERS + 1) * K) * 3);

  srand(1);

#pragma omp parallel
  {

    int tid = omp_get_thread_num();
    int num_of_items_to_process = ceil((1.0 * N) / num_threads);
    for (int i = tid * num_of_items_to_process;
         i < (tid + 1) * num_of_items_to_process && i < N; i++) {
      (*data_point_cluster)[4 * i] = data_points[3 * i];
      (*data_point_cluster)[4 * i + 1] = data_points[3 * i + 1];
      (*data_point_cluster)[4 * i + 2] = data_points[3 * i + 2];
      (*data_point_cluster)[4 * i + 3] = rand() % K;
    }

#pragma omp barrier

    for (int i = 0; i < K; i++) {
      thread_aggregates[tid][i]->reset();
    }

    for (int i = tid * num_of_items_to_process;
         i < (tid + 1) * num_of_items_to_process && i < N; i++) {
      thread_aggregates[tid][(*data_point_cluster)[4 * i + 3]]->add_to_point(
          (*data_point_cluster)[4 * i], (*data_point_cluster)[4 * i + 1],
          (*data_point_cluster)[4 * i + 2]);
    }

#pragma omp barrier

    int centroid_number = tid;
    while (centroid_number < K) {

      for (int i = 1; i < num_threads; i++) {
        thread_aggregates[0][centroid_number]->accumulate_values(
            thread_aggregates[i][centroid_number]);
      }
      thread_aggregates[0][centroid_number]->average_out_point();

      centroid_number += num_threads;
    }
  }

  for (int j = 0; j < K; j++) {
    (*centroids)[3 * j] = thread_aggregates[0][j]->x;
    (*centroids)[3 * j + 1] = thread_aggregates[0][j]->y;
    (*centroids)[3 * j + 2] = thread_aggregates[0][j]->z;
  }

  int num_changes = 0;
  int num_iters = 0;
  do {

    num_iters++;
    num_changes = 0;

    int offset = num_iters * K * 3;

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      int local_changes = 0;

      int num_of_items_to_process = ceil((1.0 * N) / num_threads);
      for (int i = tid * num_of_items_to_process;
           i < (tid + 1) * num_of_items_to_process && i < N; i++) {

        double min_dist = __INT_MAX__;
        int closest_centroid = -1;
        int offset = (num_iters - 1) * K * 3;
        for (int j = 0; j < K; j++) {
          float dist = distance(data_points, i, *centroids, j, offset);

          if (dist < min_dist) {
            min_dist = dist;
            closest_centroid = j;
          }
        }

        if (closest_centroid != (*data_point_cluster)[4 * i + 3]) {
          (*data_point_cluster)[4 * i + 3] = closest_centroid;
          local_changes++;
        }
      }

#pragma omp atomic
      num_changes += local_changes;

#pragma omp barrier

      for (int i = 0; i < K; i++) {
        thread_aggregates[tid][i]->reset();
      }

      for (int i = tid * num_of_items_to_process;
           i < (tid + 1) * num_of_items_to_process && i < N; i++) {
        thread_aggregates[tid][(*data_point_cluster)[4 * i + 3]]->add_to_point(
            (*data_point_cluster)[4 * i], (*data_point_cluster)[4 * i + 1],
            (*data_point_cluster)[4 * i + 2]);
      }

#pragma omp barrier

      int centroid_number = tid;
      while (centroid_number < K) {

        for (int i = 1; i < num_threads; i++) {
          thread_aggregates[0][centroid_number]->accumulate_values(
              thread_aggregates[i][centroid_number]);
        }
        thread_aggregates[0][centroid_number]->average_out_point();

        centroid_number += num_threads;
      }
    }

    for (int j = 0; j < K; j++) {
      (*centroids)[offset + 3 * j] = thread_aggregates[0][j]->x;
      (*centroids)[offset + 3 * j + 1] = thread_aggregates[0][j]->y;
      (*centroids)[offset + 3 * j + 2] = thread_aggregates[0][j]->z;
    }

  } while (num_changes > 0.01 * N && num_iters < MAX_ITERS);

  *num_iterations = num_iters;
}