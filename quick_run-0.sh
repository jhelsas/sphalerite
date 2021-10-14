#
# SPDX-License-Identifier:  BSD-3-Clause
#
# quick_run.py :
#     Execution script for a small number of runs 
#     and default parameters
#
# (C) Copyright 2021 José Hugo Elsas
# Author: José Hugo Elsas <jhelsas@gmail.com>
#
rm AoS/data/*
mkdir -p AoS/data/
cp makefiles/Makefile_CFLAGS0 AoS/Makefile
cd AoS/
make clean
make all
./bin/example_01-ArrayOfStructs-Naive -runs 5
./bin/example_05-ArrayOfStructs-CellLinkedList-serial -runs 5
cd ..
python diff_ref.py 
python timings_statistics.py > timing_results_0.txt