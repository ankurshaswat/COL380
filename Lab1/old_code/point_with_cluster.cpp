#include "point_with_cluster.h"

void point_with_cluster::add_to_point(point_with_cluster *point) {
  this->x += point->x;
  this->y += point->y;
  this->z += point->z;

  this->points_in_cluster++;
}

void point_with_cluster::accumulate_values(point_with_cluster *point) {
  this->x += point->x;
  this->y += point->y;
  this->z += point->z;
  this->points_in_cluster += point->points_in_cluster;
}

void point_with_cluster::copy(point_with_cluster *point) {
  this->x = point->x;
  this->y = point->y;
  this->z = point->z;
  this->points_in_cluster = point->points_in_cluster;
}

void point_with_cluster::average_out_point() {
  int n = this->points_in_cluster;

  if (n == 0) {
    this->x = 0;
    this->y = 0;
    this->z = 0;
  } else {
    this->x /= n;
    this->y /= n;
    this->z /= n;
  }

  this->points_in_cluster = 0;
}

void point_with_cluster::reset() {
  this->x = 0;
  this->y = 0;
  this->z = 0;
  this->points_in_cluster = 0;
}