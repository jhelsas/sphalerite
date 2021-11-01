# Spharelite - Example Density Calculations

SPDX-License-Identifier:  BSD-3-Clause

(C) Copyright 2021 José Hugo Elsas

Author: José Hugo Elsas <jhelsas@gmail.com>

Companion repository for the series "**How to reason about numerical code optimization on modern processors**", published on medium.

The goal of this repository is to serve as a reference to the results of the articles, and also provide a playground for anyone wishing to experiment on the different modalities of computing the SPH density. 

Each example contain a version of the density calculation that is timed, which have the command line options shown below. Running without command line will execute in the default mode. Executing the examples without creating AoS/data/ or SoA/data/ with crash the executable for being unable to write the results files. 

The suggested execution path is to run `bash quick_run.sh` , which compiles and executes 5 times each example, or `bash full_run.sh` which compiles and runs each example 100 times. `quick_run.sh` can take a few minutes in a normal computer, so don't run the `full_run.sh` unless you have time to spare. 

Running `python timings_statistics.py` produces the timings statistics for the results produced in the runs. 

**Command Line Options**: 

 * -runs  <int>   : Set the number of repetitions (runs) for calculating the density. The value of the density is based on the last  iteration.

   ​							  Default value: 1

 * -run_seed <int>: Flag to set an alternative seed use for the PRNG. Instead of feeding seed to the PRNG directly, it feeds  seed + iteration, 

   ​                              as to generate different configurations for each iteration. 

   ​                              Default value: 0 - (possible 0/1)

 * -seed     <int>: Set the seed to use for the SPH particles uniform position generation in the box

   ​							  Default value: 123123123

 * -N        <int>: Set the number of SPH particles to be used

   ​						      Default value: 1e5 = 100,000

 * -h      <float>: Set the value of the smoothing kernel parameter h, which corresponds to half of the support of the kernel. 

   ​    						  Default value: 0.05

 * -Nx       <int>: Set the number of Cells in the X direction

   ​							  Default value: 10

 * -Ny       <int>: Set the number of Cells in the Y direction

   ​							  Default value: 10

 * -Nz       <int>: Set the number of Cells in the Z direction

   ​							  Default value: 10

 * -Xmin   <float>: Set the lower bound in the X direction for the Cell Linked List box 

   ​							  Default value: 0.0

 * -Ymin   <float>: Set the lower bound in the Y direction for the Cell Linked List box 

   ​							  Default value: 0.0

 * -Ymin   <float>: Set the lower bound in the Z direction for the Cell Linked List box 

   ​							  Default value: 0.0

 * -Xmax   <float>: Set the lower bound in the X direction for the Cell Linked List box 

   ​							  Default value: 1.0

 * -Ymax   <float>: Set the lower bound in the Y direction for the Cell Linked List box 

   ​							  Default value: 1.0

 * -Zmax   <float>: Set the lower bound in the Z direction for the Cell Linked List box 

   ​							  Default value: 1.0

**Dependencies**: 

- GNU C Compiler (gcc)
- Gnu Scientific Library (GSL)
- klib (as git submodule)
- mzc (as git submodule)

# File Structure

The file tree structure can be found below:

- diff_ref.py: Computes the difference between reference density and all other densities
- timings_statistics.py: Compute the timings statistics
- include/ : Common libraries for all versions of the code
  - klib/ : klib library include khash header-only hash-table library.
  - mzc/ : morton Z-Code Hash calculation and manipulation. 
- AoS/ : Array of Structs version of the SPH density calculation
  - src/ : Basic utilities, Cell Linked List manipulation
    - sph_linked_list.c
    - sph_utils.c
  - include/ : Headers associated with the src/ directory
    - sph_data_types.h
    - sph_linked_list.h
    - sph_utils.h
  - exec/ :
    - example_01-ArrayOfStructs-Naive.c
    - example_02-ArrayOfStructs-Naive-Omp.c
    - example_03-ArrayOfStructs-Naive-Omp-SIMD.c
    - example_04-ArrayOfStructs-Naive-Omp-SIMD-Tiled.c
    - example_05-ArrayOfStructs-CellLinkedList-Serial.c
    - example_06-ArrayOfStructs-CellLinkedList-innerOmp.c
    - example_07-ArrayOfStructs-CellLinkedList-innerOmp-SIMD.c
    - example_08-ArrayOfStructs-outerOmp.c
    - example_09-ArrayOfStructs-outerOmp-loadBallanced.c
- SoA/ : Struct of Arrays version of the SPH density calculation
  - src/ : Basic utilities, Cell Linked List manipulation
    - sph_linked_list.c
    - sph_utils.c
  - include/ : Headers associated with the src/ directory
    - sph_data_types.h
    - sph_linked_list.h
    - sph_utils.h
  - exec/ :
    - example_01-StructOfArrays-Naive.c
    - example_02-StructOfArrays-Naive-Omp.c
    - example_03-StructOfArrays-Naive-Omp-SIMD.c
    - example_04-StructOfArrays-Naive-Omp-SIMD-Tiled.c
    - example_05-StructOfArrays-CellLinkedList-Serial.c
    - example_06-StructOfArrays-CellLinkedList-innerOmp.c
    - example_07-StructOfArrays-CellLinkedList-innerOmp-SIMD.c
    - example_08-StructOfArrays-outerOmp.c
    - example_09-StructOfArrays-outerOmp-loadBallanced.c
    - example_10-StructOfArrays-outerOmp-SymmetricalLoadBalancing.c

