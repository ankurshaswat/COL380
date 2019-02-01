all: sequential pthread

sequential: lab1_io.cpp lab1_sequential.cpp main_sequential.cpp
		g++ -g -o main_sequential.out lab1_io.cpp lab1_sequential.cpp main_sequential.cpp

pthread: point_with_cluster.cpp lab1_io.cpp lab1_pthread.cpp main_pthread.cpp
		g++ -g -o main_pthread.out point_with_cluster.cpp lab1_io.cpp lab1_pthread.cpp main_pthread.cpp -lpthread

clean:
	rm -rf *.o *.out