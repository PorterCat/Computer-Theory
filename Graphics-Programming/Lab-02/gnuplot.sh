#!/bin/bash

gnuplot -e "set terminal png size 1280,1024; \
            set output 'plot.png'; \
            set xlabel 'Threads per Block'; \
            set ylabel 'Time (ms)'; \
            set title 'CUDA Vector Addition Performance'; \
            plot 'result.csv' using 1:2 with linespoints title 'Time vs Threads'"