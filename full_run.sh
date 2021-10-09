#
# SPDX-License-Identifier:  BSD-3-Clause
#
# full_run.py :
#     Execution script for a large number of runs 
#     and default parameters
#
# (C) Copyright 2021 José Hugo Elsas
# Author: José Hugo Elsas <jhelsas@gmail.com>
#
cd AoS/
make clean
make all
./bin/example_01-ArrayOfStructs-Naive -runs 100
./bin/example_02-ArrayOfStructs-Naive-Omp -runs 100
./bin/example_03-ArrayOfStructs-Naive-Omp-SIMD -runs 100
./bin/example_04-ArrayOfStructs-Naive-Omp-SIMD-Tiled -runs 100
./bin/example_05-ArrayOfStructs-CellLinkedList-serial -runs 100
./bin/example_06-ArrayOfStructs-CellLinkedList-innerOmp -runs 100
./bin/example_07-ArrayOfStructs-CellLinkedList-innerOmp-SIMD -runs 100
./bin/example_08-ArrayOfStructs-CellLinkedList-outerOmp -runs 100
./bin/example_09-ArrayOfStructs-CellLinkedList-outerOmp-loadBallanced -runs 100
cd ..
cd SoA/
make clean
make all
./bin/example_01-StructOfArrays-Naive -runs 100
./bin/example_02-StructOfArrays-Naive-Omp -runs 100
./bin/example_03-StructOfArrays-Naive-Omp-SIMD -runs 100
./bin/example_04-StructOfArrays-Naive-Omp-SIMD-Tiled -runs 100
./bin/example_05-StructOfArrays-CellLinkedList-serial -runs 100
./bin/example_06-StructOfArrays-CellLinkedList-InnerOmp -runs 100
./bin/example_07-StructOfArrays-CellLinkedList-InnerOmp-SIMD -runs 100
./bin/example_08-StructOfArrays-CellLinkedList-OuterOmp -runs 100
./bin/example_09-StructOfArrays-CellLinkedList-OuterLoop-LoadBalanced -runs 100
./bin/example_10-StructOfArrays-CellLinkedList-OuterLoop-SymmetricalLoadBalancing -runs 100
cd .. 
python diff_ref.py 