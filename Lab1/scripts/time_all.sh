#!/bin/bash
rm ../results/time_sequential.txt

for i in `seq 10000 10000 100000`
    do
        echo -n "${i} "
        ./../bin/main_sequential.out 10 ../dataset/dataset_${i}_10.txt ../results/dataset_${i}_10_sequential_data_points.txt ../results/dataset_${i}_10_sequential_centroids.txt | cut -d' ' -f 3
    done | tee -a ../results/time_sequential.txt 

for type in 'pthread' 'omp'
    do
        for threads in 2 4 8 16
            do
                rm ../results/time_${type}_${threads}.txt
                for i in `seq 10000 10000 100000`
                    do
                        echo -n "${i} "
                        ./../bin/main_${type}.out ${threads} 10 ../dataset/dataset_${i}_10.txt ../results/dataset_${i}_10_${type}_${threads}_data_points.txt ../results/dataset_${i}_10_${type}_${threads}_centroids.txt | cut -d' ' -f 3
                    done | tee -a ../results/time_${type}_${threads}.txt 
            done
    done

gnuplot plot_all