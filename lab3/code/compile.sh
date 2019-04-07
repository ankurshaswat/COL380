export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
nvcc -g -G -lm main_cuda.cu lab3_cuda.cu lab3_io.cu -o pca