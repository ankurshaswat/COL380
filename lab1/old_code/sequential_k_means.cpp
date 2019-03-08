#include "helper.h"
#include "point_with_cluster.h"
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

int K = 100;
int NUM_POINTS = 10000;

int main() {

  srand(1);

  vector<point_with_cluster *> points;
  generate_random_points(NUM_POINTS, K, points);

  vector<point_with_cluster *> centroids;
  for (int i = 0; i < K; i++) {
    point_with_cluster *new_point = new point_with_cluster;
    centroids.push_back(new_point);
  }

  int num_changes = 0;
  int num_iters = 0;

  do {
    cout << num_changes << endl << endl;

    num_iters++;
    num_changes = 0;

    // Find new Centroids by averaging points values
    for (int i = 0; i < K; i++) {
      centroids[i]->reset();
    }
    for (int i = 0; i < NUM_POINTS; i++) {
      centroids[points[i]->cluster]->add_to_point(points[i]);
    }
    for (int i = 0; i < K; i++) {
      centroids[i]->average_out_point();
    }

    // Assign new closest centrois
    for (int i = 0; i < NUM_POINTS; i++) {

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
        num_changes++;
      }
    }

    printAverageDistance(points, centroids);
  } while (num_changes > 0.01 * NUM_POINTS && num_iters < 1000);

  cout << "Num Iters= " << num_iters << endl;

  return 0;
}