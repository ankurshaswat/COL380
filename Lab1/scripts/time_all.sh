#!/bin/bash
rm ../results/time_sequential.txt
K=10

for i in `seq 10000 10000 100000`
    do
        echo -n "${i} "
        ./../bin/main_sequential.out ${K} ../dataset/dataset_${i}_${K}.txt ../results/dataset_${i}_${K}_sequential_data_points.txt ../results/dataset_${i}_${K}_sequential_centroids.txt | cut -d' ' -f 3
    done | tee -a ../results/time_sequential.txt 

for type in 'pthread' 'omp'
    do
        for threads in 2 4 8 16
            do
                rm ../results/time_${type}_${threads}.txt
                for i in `seq 10000 10000 100000`
                    do
                        echo -n "${i} "
                        ./../bin/main_${type}.out ${K} ${threads} ../dataset/dataset_${i}_${K}.txt ../results/dataset_${i}_${K}_${type}_${threads}_data_points.txt ../results/dataset_${i}_${K}_${type}_${threads}_centroids.txt | cut -d' ' -f 3
                    done | tee -a ../results/time_${type}_${threads}.txt 
            done
    done

gnuplot plot_all
