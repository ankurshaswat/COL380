#include "point_with_cluster.h"
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

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

void printPoint(point_with_cluster *p) {
  cout << "Point" << endl;
  cout << "(" << p->x << "," << p->y << "," << p->z << ") " << p->cluster
       << endl
       << endl;
}

void printPoints(vector<point_with_cluster *> &points, int type) {
  if (type == 0) {
    cout << "Points" << endl;
  } else if (type == 1) {
    cout << "Centroids" << endl;
  } else {
    cout << "Thread Aggr" << endl;
  }
  for (int i = 0; i < points.size(); i++) {
    cout << "(" << points[i]->x << "," << points[i]->y << "," << points[i]->z
         << ") " << points[i]->cluster << endl;
  }
  cout << endl;
}

void printAverageDistance(vector<point_with_cluster *> &points,
                          vector<point_with_cluster *> &cluster) {
  double totalDist = 0;
  for (int i = 0; i < points.size(); i++) {
    totalDist += distance(points[i], cluster[points[i]->cluster]);
  };
  cout << totalDist << endl;
}