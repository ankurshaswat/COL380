#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "point_with_cluster.h"
using namespace std;

int NUM_THREADS = 4;
int k = 3;
int num_points = 10;

struct thread_args {
  int tid;
  vector<point_with_cluster *> points;

  thread_args(int tid, vector<point_with_cluster *> &points) {
    this->tid = tid;
    this->points = points;
  }
};

vector<vector<point_with_cluster *>> thread_aggregates;

void generate_random_points(int number, int number_of_clusters,
                            vector<point_with_cluster *> &points) {

  for (int i = 0; i < number; i++) {
    point_with_cluster *new_point = new point_with_cluster;

    new_point->x = rand() % 200 - 100;
    new_point->y = rand() % 200 - 100;
    new_point->z = rand() % 200 - 100;
    new_point->cluster = rand() % number_of_clusters;

    points.push_back(new_point);
  }
}

double distance(point_with_cluster *point1, point_with_cluster *point2) {
  return sqrt(pow(point1->x - point2->x, 2) + pow(point1->y - point2->y, 2) +
              pow(point1->z - point2->z, 2));
}

void printPoints(vector<point_with_cluster *> &points) {
  cout << "Cycle Start" << endl;
  for (int i = 0; i < points.size(); i++) {
    cout << "(" << points[i]->x << "," << points[i]->y << "," << points[i]->z
         << ") " << points[i]->cluster << endl;
  }
}

void printAverageDistance(vector<point_with_cluster *> &points,
                          vector<point_with_cluster *> &cluster) {
  double totalDist = 0;
  for (int i = 0; i < points.size(); i++) {
    totalDist += distance(points[i], cluster[points[i]->cluster]);
  };
  cout << totalDist << endl;
}

void *aggregrate_points(void *in) {
  thread_args *args = (thread_args *)in;
  vector<point_with_cluster *> points = args->points;
  int tid = args->tid;
  //   cout << "Thread Started = " << args->tid << endl;
  for (int i = 0; i < k; i++) {
    thread_aggregates[tid][i]->reset();
  }

  int length = points.size();
  int num_of_items_to_process = length / NUM_THREADS;
  for (int i = tid * num_of_items_to_process;
       i < (tid + 1) * num_of_items_to_process && i < length; i++) {
    thread_aggregates[tid][points[i]->cluster]->add_to_point(points[i]);
  }
}

int main() {

  srand(1);

  vector<point_with_cluster *> points;
  generate_random_points(num_points, k, points);

  int return_code;
  pthread_t threads[NUM_THREADS];

  for (int i = 0; i < NUM_THREADS; i++) {

    vector<point_with_cluster *> thread_aggr;
    for (int j = 0; j < k; j++) {
      point_with_cluster *point = new point_with_cluster;
      thread_aggr.push_back(point);
    }
    thread_aggregates.push_back(thread_aggr);

    // thread_args *t_args = new thread_args(i, points);

    // return_code =
    // pthread_create(&threads[i], NULL, aggregrate_points, (void *)t_args);

    // if (return_code) {
    //   cout << "Error:unable to create thread," << return_code << endl;
    //   exit(-1);
    // }
  }
  //   pthread_exit(NULL);

  vector<point_with_cluster *> centroids;
  for (int i = 0; i < k; i++) {
    point_with_cluster *new_point = new point_with_cluster;
    centroids.push_back(new_point);
  }

  int num_changes = 0;
  int num_iters = 0;

  do {
    printPoints(points);
    cout << num_changes << endl;
    num_iters++;
    num_changes = 0;

    // Find new Centroids by averaging points values
    for (int i = 0; i < NUM_THREADS; i++) {
      thread_args *t_args = new thread_args(i, points);

      return_code =
          pthread_create(&threads[i], NULL, aggregrate_points, (void *)t_args);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
      pthread_join(threads[i], NULL);
    }

    for (int j = 0; j < k; j++) {
      for (int i = 1; i < NUM_THREADS; i++) {
        thread_aggregates[0][j]->add_to_point(thread_aggregates[i][j]);
      }
      thread_aggregates[0][j]->average_out_point();
      centroids[j]->copy(thread_aggregates[0][j]);
    }

    //     for (int i = 0; i < num_points; i++) {
    //       centroids[points[i]->cluster]->add_to_point(points[i]);
    //     }
    //     for (int i = 0; i < k; i++) {
    //       centroids[i]->average_out_point();
    //     }

    // Assign new closest centrois
    for (int i = 0; i < num_points; i++) {

      double min_dist = __INT_MAX__;
      int closest_centroid = -1;

      for (int j = 0; j < k; j++) {
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
  } while (num_changes > 0.01 * num_points && num_iters < 1000);

  cout << "Num Iters= " << num_iters << endl;

  return 0;
}