#include <cmath>
#include <iostream>
using namespace std;

float distance(int *data_points, int point_num, float *centroids,
               int centroid_num, int offset) {

  return sqrt(
      pow(data_points[3 * point_num] - centroids[offset + 4 * centroid_num],
          2) +
      pow(data_points[3 * point_num + 1] -
              centroids[offset + 4 * centroid_num + 1],
          2) +
      pow(data_points[3 * point_num + 2] -
              centroids[offset + 4 * centroid_num + 2],
          2));
}

void printAverageDistance(int N, int *data_points, float *centroids,
                          int offset) {
  float totalDist = 0;
  for (int i = 0; i < N; i++) {
    totalDist +=
        distance(data_points, i, centroids, data_points[4 * i + 3], offset);
  };
  cout << totalDist << endl;
}

void printPoint(int *data_points, int n) {
  cout << data_points[3 * n] << ' ' << data_points[3 * n + 1] << ' '
       << data_points[3 * n + 2] << endl;
}

void printCentroid(float *data_points, int n, int offset) {
  cout << data_points[offset + 3 * n] << ' ' << data_points[offset + 3 * n + 1]
       << ' ' << data_points[offset + 3 * n + 2] << endl;
}