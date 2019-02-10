all: sequential pthread omp

sequential: lab1_io.cpp lab1_sequential.cpp main_sequential.cpp
		g++ -g -o ../bin/main_sequential.out lab1_io.cpp lab1_sequential.cpp main_sequential.cpp -fopenmp

pthread: point_with_cluster.cpp lab1_io.cpp lab1_pthread.cpp main_pthread.cpp
		g++ -g -o ../bin/main_pthread.out point_with_cluster.cpp lab1_io.cpp lab1_pthread.cpp main_pthread.cpp -lpthread -fopenmp

omp:  point_with_cluster.cpp lab1_io.cpp lab1_omp.cpp main_omp.cpp
		g++ -g -o ../bin/main_omp.out point_with_cluster.cpp lab1_io.cpp lab1_omp.cpp main_omp.cpp -fopenmp

clean:
	rm -rf *.o *.out ../bin/*.o ../bin/*.out