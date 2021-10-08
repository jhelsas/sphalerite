/*
 * SPDX-License-Identifier:  BSD-3-Clause
 * 
 * sph_linked_list.c : 
 *     general utility functions
 *
 * (C) Copyright 2021 José Hugo Elsas
 * Author: José Hugo Elsas <jhelsas@gmail.com>
 *
 */

#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/time.h>
#include <inttypes.h>

#include "sph_data_types.h"

/*
 *  Function arg_parse:
 *    Easily allocate the SPHparticle array
 * 
 *    Arguments:
 *      argc   <int>     : number of command line arguments
 *      argv  <char*>    : array of command line arguments
 *       N   <int64_t*>  : Number of SPH particles
 *       h    <double>   : Smoothing length
 *     seed   <long int> : Seed for the PRNG
 *     runs     <int>    : number of repetitions of the density calculations
 *     run_seed  <bool>  : boolean defining repeat or not the seed for all runs
 *  box <linkedListBox*> : Box of linked list cells
 */
int arg_parse(int argc, char **argv, int64_t *N, double *h,
              long int *seed, int *runs, bool *run_seed, linkedListBox *box){
  bool intern_h = true;

  box->Xmin = 0.0; box->Ymin = 0.0; box->Zmin = 0.0;
  box->Xmax = 1.0; box->Ymax = 1.0; box->Zmax = 1.0;

  for(int i=1;i<argc;){
    if( strcmp(argv[i],"-N") == 0 ){
      *N = (int64_t) atol(argv[i+1]);
      printf("N particles = %ld\n",*N);
      i+=2;
    }
    else if( strcmp(argv[i],"-seed") == 0 ){
      *seed = (long int) atol(argv[i+1]);
      printf("seed = %ld\n",*seed);
      i+=2;
    }
    else if( strcmp(argv[i],"-runs") == 0 ){
      *runs = (int) atoi(argv[i+1]);
      printf("runs = %d\n",*runs);
      i+=2;
    }
    else if( strcmp(argv[i],"-run_seed") == 0 ){
      /*int ran_seed = (int) atoi(argv[i+1]);
      if(ran_seed)
        *run_seed = true;
      else
        *run_seed = false;*/
      *run_seed=true;
      printf("run_seed = %d\n",true);
      i+=1;
    }
    else if( strcmp(argv[i],"-h") == 0 ){
      *h = atof(argv[i+1]);
      printf("h = %lf\n",*h);
      intern_h = false;
      i+=2;
    }
    else if( strcmp(argv[i],"-Xmin") == 0 ){
      box->Xmin = atof(argv[i+1]);
      printf("Xmin = %lf\n",box->Xmin);
      i+=2;
    }
    else if( strcmp(argv[i],"-Ymin") == 0 ){
      box->Ymin = atof(argv[i+1]);
      printf("Ymin = %lf\n",box->Ymin);
      i+=2;
    }
    else if( strcmp(argv[i],"-Zmin") == 0 ){
      box->Zmin = atof(argv[i+1]);
      printf("Zmin = %lf\n",box->Zmin);
      i+=2;
    }
    else if( strcmp(argv[i],"-Xmax") == 0 ){
      box->Xmax = atof(argv[i+1]);
      printf("Xmax = %lf\n",box->Xmax);
      i+=2;
    }
    else if( strcmp(argv[i],"-Ymax") == 0 ){
      box->Ymax = atof(argv[i+1]);
      printf("Ymax = %lf\n",box->Ymax);
      i+=2;
    }
    else if( strcmp(argv[i],"-Zmax") == 0 ){
      box->Zmax = atof(argv[i+1]);
      printf("Zmax = %lf\n",box->Zmax);
      i+=2;
    }
    else if( strcmp(argv[i],"-Nx") == 0 ){
      box->Nx   = atol(argv[i+1]);
      printf("Nx = %d\n",box->Nx);
      i+=2;
    }
    else if( strcmp(argv[i],"-Ny") == 0 ){
      box->Ny   = atol(argv[i+1]);
      printf("Ny = %d\n",box->Ny);
      i+=2;
    }
    else if( strcmp(argv[i],"-Nz") == 0 ){
      box->Nz   = atol(argv[i+1]);
      printf("Nz = %d\n",box->Nz);
      i+=2;
    }
    else{
      printf("unknown option: %s %s\n",argv[i],argv[i+1]);
      i+=2;
    }
  }

  if(intern_h){
    box->Nx = (int)( (box->Xmax-box->Xmin)/(2*(*h)) );
    box->Ny = (int)( (box->Ymax-box->Ymin)/(2*(*h)) );
    box->Nz = (int)( (box->Zmax-box->Zmin)/(2*(*h)) );
  }

  box->N  = (box->Nx)*(box->Ny)*(box->Nz);
  
  double min_val = fmin((box->Xmax-box->Xmin)/box->Nx,fmin((box->Ymax-box->Ymin)/box->Ny,(box->Zmax-box->Zmin)/box->Nz));
  box->width  = (int)( 0.5 + 2*(*h)/min_val );
  box->hbegin = kh_init(0);
  box->hend   = kh_init(1);

  return 0;
}