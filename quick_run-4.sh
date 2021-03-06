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
mkdir -p AoS/data/
mkdir -p SoA/data/
rm AoS/data/*
rm SoA/data/*
cp makefiles/Makefile_CFLAGS4 AoS/Makefile
cp makefiles/Makefile_CFLAGS4 SoA/Makefile
cd AoS/
make clean
make all
./bin/example_01-ArrayOfStructs-Naive -runs 5
./bin/example_02-ArrayOfStructs-Naive-Omp -runs 5
./bin/example_03-ArrayOfStructs-Naive-Omp-SIMD -runs 5
./bin/example_04-ArrayOfStructs-Naive-Omp-SIMD-Tiled -runs 5
./bin/example_05-ArrayOfStructs-CellLinkedList-serial -runs 5
./bin/example_06-ArrayOfStructs-CellLinkedList-innerOmp -runs 5
./bin/example_07-ArrayOfStructs-CellLinkedList-innerOmp-SIMD -runs 5
cd ..
cd SoA/
make clean
make all
./bin/example_01-StructOfArrays-Naive -runs 5
./bin/example_02-StructOfArrays-Naive-Omp -runs 5
./bin/example_03-StructOfArrays-Naive-Omp-SIMD -runs 5
./bin/example_04-StructOfArrays-Naive-Omp-SIMD-Tiled -runs 5
./bin/example_05-StructOfArrays-CellLinkedList-serial -runs 5
./bin/example_06-StructOfArrays-CellLinkedList-InnerOmp -runs 5
./bin/example_07-StructOfArrays-CellLinkedList-InnerOmp-SIMD -runs 5
cd .. 
python diff_ref.py 
python timings_statistics.py > timing_results_4.txt