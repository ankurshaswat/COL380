#include "helper.h"
#include "point_with_cluster.h"
#include <omp.h>
#include <vector>

using namespace std;

int NUM_THREADS = 8;
int K = 100;
int NUM_POINTS = 10000;

int num_changes = 0;

vector<point_with_cluster *> points;
vector<point_with_cluster *> centroids;
vector<vector<point_with_cluster *>> thread_aggregates;

int main() {
  omp_set_num_threads(NUM_THREADS);

  srand(1);

  generate_random_points(NUM_POINTS, K, points);

  int return_code;

  for (int i = 0; i < NUM_THREADS; i++) {
    vector<point_with_cluster *> thread_aggr;
    for (int j = 0; j < K; j++) {
      point_with_cluster *point = new point_with_cluster;
      thread_aggr.push_back(point);
    }
    thread_aggregates.push_back(thread_aggr);
  }

  for (int i = 0; i < K; i++) {
    point_with_cluster *new_point = new point_with_cluster;
    centroids.push_back(new_point);
  }

  int num_iters = 0;

  do {
    cout << num_changes << endl << endl;

    num_iters++;
    num_changes = 0;

// Accumulating points values
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      for (int i = 0; i < K; i++) {
        thread_aggregates[tid][i]->reset();
      }

      int length = points.size();
      int num_of_items_to_process = ceil((1.0 * length) / NUM_THREADS);
      for (int i = tid * num_of_items_to_process;
           i < (tid + 1) * num_of_items_to_process && i < length; i++) {
        thread_aggregates[tid][points[i]->cluster]->add_to_point(points[i]);
      }
    }

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      int centroid_number = tid;
      while (centroid_number < K) {

        for (int i = 1; i < NUM_THREADS; i++) {
          thread_aggregates[0][centroid_number]->accumulate_values(
              thread_aggregates[i][centroid_number]);
        }
        thread_aggregates[0][centroid_number]->average_out_point();
        centroids[centroid_number]->copy(thread_aggregates[0][centroid_number]);

        centroid_number += NUM_THREADS;
      }
    }

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      int local_changes = 0;

      int length = points.size();
      int num_of_items_to_process = ceil((1.0 * length) / NUM_THREADS);
      for (int i = tid * num_of_items_to_process;
           i < (tid + 1) * num_of_items_to_process && i < length; i++) {

        double min_dist = __INT_MAX__;
        int closest_centroid = -1;

        for (int j = 0; j < K; j++) {
          int dist = distance(points[i], centroids[j]);
          if (dist < min_dist) {
            min_dist = dist;
            closest_centroid = j;
          }
        }

        if (closest_centroid != points[i]->cluster) {
          points[i]->cluster = closest_centroid;
          local_changes++;
        }
      }

#pragma omp atomic
      num_changes += local_changes;
    }

    printAverageDistance(points, centroids);
  } while (num_changes > 0.01 * NUM_POINTS && num_iters < 1000);

  cout << "Num Iters= " << num_iters << endl;

  pthread_exit(NULL);

  return 0;
}