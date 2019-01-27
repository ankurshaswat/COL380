#ifndef POINTCLUST
#define POINTCLUST

struct point_with_cluster {
  /* data */
  double x = 0, y = 0, z = 0;
  int cluster;
  int points_in_cluster = 0;
  void add_to_point(point_with_cluster *point);
  void copy(point_with_cluster *point);
  void average_out_point();
  void reset();
};

#endif