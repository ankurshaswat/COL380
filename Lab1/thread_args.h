

#ifndef THREADARGS
#define THREADARGS

#include <vector>
#include "point_with_cluster.h"
using namespace std;

struct thread_args {
  int tid;
  vector<point_with_cluster *> points;

  thread_args(int tid, vector<point_with_cluster *> &points);
};

#endif