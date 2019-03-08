all: point_with_cluster.cpp sequential_k_means.cpp parallel_pthreads.cpp parallel_pthreadsv2.cpp parallel_omp.cpp
		g++ -g -o sequential.out point_with_cluster.cpp sequential_k_means.cpp 
		g++ -g -o parallel_pthreads.out point_with_cluster.cpp parallel_pthreads.cpp -lpthread
		g++ -g -o parallel_pthreadsv2.out point_with_cluster.cpp parallel_pthreadsv2.cpp -lpthread
		g++ -g -o parallel_omp.out point_with_cluster.cpp parallel_omp.cpp -fopenmp

clean:
	rm -rf *.o *.out