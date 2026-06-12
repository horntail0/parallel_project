#!/bin/bash

: ${NODES:=1}

salloc -N $NODES --exclusive --partition=samsung --gres=gpu:4                              \
  mpirun --bind-to none -mca coll_hcoll_enable 0 -mca btl ^openib -mca pml ucx -npernode 1 \
  --oversubscribe -quiet                                                                   \
  numactl --physcpubind 0-31                                                               \
  ./main $@
