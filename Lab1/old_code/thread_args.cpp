#include "thread_args.h"

thread_args::thread_args(int tid, vector<point_with_cluster *> &points) {
    this->tid = tid;
    this->points = points;
  }