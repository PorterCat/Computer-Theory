#!/bin/bash

#ncu --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum

ncu --metrics "gpu__time_duration.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
lts__t_bytes.sum,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed" \
    $1
