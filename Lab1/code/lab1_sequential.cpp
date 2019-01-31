#include "lab1_sequential.h"
#include "helper.h"
#include <cmath>
#include <iostream>
#include <stdlib.h>
using namespace std;

void kmeans_sequential(int N, int K, int *data_points, int **data_point_cluster,
                       float **centroids, int *num_iterations) {

  int MAX_ITERS = 150;
  int num_iters = 0;
  int num_points = N;

  *data_point_cluster = (int *)malloc(sizeof(int) * (num_points)*4);
  *centroids = (float *)malloc(sizeof(float) * ((MAX_ITERS + 1) * K) * 3);

  srand(1);

  for (int i = 0; i < num_points; i++) {
    (*data_point_cluster)[4 * i] = data_points[3 * i];
    (*data_point_cluster)[4 * i + 1] = data_points[3 * i + 1];
    (*data_point_cluster)[4 * i + 2] = data_points[3 * i + 2];
    (*data_point_cluster)[4 * i + 3] = rand() % K;
  }

  int cluster_counts[K];

  for (int i = 0; i < K; i++) {
    cluster_counts[i] = 0;
  }

  for (int i = 0; i < num_points; i++) {
    (*centroids)[(*data_point_cluster)[4 * i + 3] * 3] +=
        (*data_point_cluster)[4 * i];
    (*centroids)[(*data_point_cluster)[4 * i + 3] * 3 + 1] +=
        (*data_point_cluster)[4 * i + 1];
    (*centroids)[(*data_point_cluster)[4 * i + 3] * 3 + 2] +=
        (*data_point_cluster)[4 * i + 2];
    cluster_counts[(*data_point_cluster)[4 * i + 3]]++;
  }

  for (int i = 0; i < K; i++) {
    if (cluster_counts[i] == 0) {
      cluster_counts[i] = 1;
    }
    (*centroids)[3 * i] /= cluster_counts[i];
    (*centroids)[3 * i + 1] /= cluster_counts[i];
    (*centroids)[3 * i + 2] /= cluster_counts[i];
    cluster_counts[i] = 0;
  }

  int num_changes = 0;

  do {
    num_iters++;

    num_changes = 0;
    // Assign new closest centrois
    for (int i = 0; i < num_points; i++) {

      float min_dist = __INT_MAX__;
      int closest_centroid = -1;

      for (int j = 0; j < K; j++) {
        float dist =
            distance(data_points, i, *centroids, j, (num_iters - 1) * K * 3);
        if (dist < min_dist) {
          min_dist = dist;
          closest_centroid = j;
        }
      }

      if (closest_centroid != (*data_point_cluster)[4 * i + 3]) {
        (*data_point_cluster)[4 * i + 3] = closest_centroid;
        num_changes++;
      }
    }

    int offset_centroid = num_iters * K * 3;

    // Find new Centroids by averaging points values
    for (int i = 0; i < K; i++) {
      (*centroids)[offset_centroid + 3 * i] = 0;
      (*centroids)[offset_centroid + 3 * i + 1] = 0;
      (*centroids)[offset_centroid + 3 * i + 2] = 0;
    }

    for (int i = 0; i < num_points; i++) {
      (*centroids)[offset_centroid + (*data_point_cluster)[4 * i + 3] * 3] +=
          (*data_point_cluster)[4 * i];
      (*centroids)[offset_centroid + (*data_point_cluster)[4 * i + 3] * 3 +
                   1] += (*data_point_cluster)[4 * i + 1];
      (*centroids)[offset_centroid + (*data_point_cluster)[4 * i + 3] * 3 +
                   2] += (*data_point_cluster)[4 * i + 2];
      cluster_counts[(*data_point_cluster)[4 * i + 3]]++;
    }

    for (int i = 0; i < K; i++) {
      if (cluster_counts[i] == 0) {
        cluster_counts[i] = 1;
      }
      (*centroids)[offset_centroid + 3 * i] /= cluster_counts[i];
      (*centroids)[offset_centroid + 3 * i + 1] /= cluster_counts[i];
      (*centroids)[offset_centroid + 3 * i + 2] /= cluster_counts[i];
      cluster_counts[i] = 0;
    }
    cout << num_changes << endl << endl;

    printAverageDistance(num_points, *data_point_cluster, *centroids,
                         offset_centroid);
  } while (num_changes > 0.01 * num_points && num_iters < MAX_ITERS);

  *num_iterations = num_iters;
  cout << "Num Iters= " << num_iters << endl;
}