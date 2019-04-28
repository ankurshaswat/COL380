mpic++ main_mpi.cpp lab4_mpi.cpp lab4_io.cpp -lm -g -o ppm
# mpirun -np 6 ./ppm ../testcases/testcase_12_1.txt
mpirun -np 6 ./ppm ../testcases/testcase_16_1.txt
mpirun -np 6 ./ppm ../testcases/testcase_10000000_10000.txt
mpirun -np 6 ./ppm ../testcases/testcase_10000_10.txt
mpirun -np 6 ./ppm ../testcases/testcase_10587_10.txt
mpirun -np 6 ./ppm ../testcases/testcase_10000000_1.txt
