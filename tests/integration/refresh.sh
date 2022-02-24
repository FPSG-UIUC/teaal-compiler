#! /bin/sh

for TEST in dotprod example example2 example3 example4 example5 example6 example7 gemm gemv gram mttkrp nrm_sq outerprod sddmm spmv spmm ttm ttv
do
    echo $TEST
    ./compile.sh tests/integration/$TEST.yaml > tests/integration/$TEST.hfa
done
