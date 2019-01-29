all: point_with_cluster.cpp thread_args.cpp sequential_k_means.cpp parallel_pthreads.cpp parallel_pthreadsv2.cpp
		g++ -g -o sequential.out point_with_cluster.cpp sequential_k_means.cpp 
		g++ -g -o parallel_pthreads.out point_with_cluster.cpp thread_args.cpp parallel_pthreads.cpp -lpthread
		g++ -g -o parallel_pthreadsv2.out point_with_cluster.cpp thread_args.cpp parallel_pthreadsv2.cpp -lpthread

clean:
	rm -rf *.o *.out