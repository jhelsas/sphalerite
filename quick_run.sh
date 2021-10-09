cd AoS/
make clean
make all
./bin/example_01-ArrayOfStructs-Naive
./bin/example_02-ArrayOfStructs-Naive-Omp
./bin/example_03-ArrayOfStructs-Naive-Omp-SIMD
./bin/example_04-ArrayOfStructs-Naive-Omp-SIMD-Tiled
./bin/example_05-ArrayOfStructs-CellLinkedList-serial
./bin/example_06-ArrayOfStructs-CellLinkedList-innerOmp
./bin/example_07-ArrayOfStructs-CellLinkedList-innerOmp-SIMD
./bin/example_08-ArrayOfStructs-CellLinkedList-outerOmp
./bin/example_09-ArrayOfStructs-CellLinkedList-outerOmp-loadBallanced
cd ..
cd SoA/
make clean
make all
./bin/example_01-StructOfArrays-Naive
./bin/example_02-StructOfArrays-Naive-Omp
./bin/example_03-StructOfArrays-Naive-Omp-SIMD
./bin/example_04-StructOfArrays-Naive-Omp-SIMD-Tiled
./bin/example_05-StructOfArrays-CellLinkedList-serial
./bin/example_06-StructOfArrays-CellLinkedList-InnerOmp
./bin/example_07-StructOfArrays-CellLinkedList-InnerOmp-SIMD
./bin/example_08-StructOfArrays-CellLinkedList-OuterOmp
./bin/example_09-StructOfArrays-CellLinkedList-OuterLoop-LoadBalanced
./bin/example_10-StructOfArrays-CellLinkedList-OuterLoop-SymmetricalLoadBalancing
cd .. 
python diff_ref.py 