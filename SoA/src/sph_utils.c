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

int arg_parse(int argc, char **argv, int64_t *N, double *h,
              long int *seed, int *runs, bool *run_seed, linkedListBox *box){
  bool intern_h = true;

  box->Xmin = 0.0; box->Ymin = 0.0; box->Zmin = 0.0;
  box->Xmax = 1.0; box->Ymax = 1.0; box->Zmax = 1.0;
  
  if(argc%2==0){
    printf("wrong number of arguments!\n");
    printf("Maybe an option is missing a value?\n");
  }

  for(int i=1;i<argc;i+=2){
    if( strcmp(argv[i],"-N") == 0 ){
      *N = (int64_t) atol(argv[i+1]);
      printf("N particles = %ld\n",*N);
    }
    else if( strcmp(argv[i],"-seed") == 0 ){
      *seed = (long int) atol(argv[i+1]);
      printf("seed = %ld\n",*seed);
    }
    else if( strcmp(argv[i],"-runs") == 0 ){
      *runs = (int) atoi(argv[i+1]);
      printf("runs = %d\n",*runs);
    }
    else if( strcmp(argv[i],"-run_seed") == 0 ){
      int ran_seed = (int) atoi(argv[i+1]);
      if(ran_seed)
        *run_seed = true;
      else
        *run_seed = false;
      printf("run_seed = %d\n",ran_seed);
    }
    else if( strcmp(argv[i],"-h") == 0 ){
      *h = atof(argv[i+1]);
      printf("h = %lf\n",*h);
      intern_h = false;
    }
    else if( strcmp(argv[i],"-Xmin") == 0 ){
      box->Xmin = atof(argv[i+1]);
      printf("Xmin = %lf\n",box->Xmin);
    }
    else if( strcmp(argv[i],"-Ymin") == 0 ){
      box->Ymin = atof(argv[i+1]);
      printf("Ymin = %lf\n",box->Ymin);
    }
    else if( strcmp(argv[i],"-Zmin") == 0 ){
      box->Zmin = atof(argv[i+1]);
      printf("Zmin = %lf\n",box->Zmin);
    }
    else if( strcmp(argv[i],"-Xmax") == 0 ){
      box->Xmax = atof(argv[i+1]);
      printf("Xmax = %lf\n",box->Xmax);
    }
    else if( strcmp(argv[i],"-Ymax") == 0 ){
      box->Ymax = atof(argv[i+1]);
      printf("Ymax = %lf\n",box->Ymax);
    }
    else if( strcmp(argv[i],"-Zmax") == 0 ){
      box->Zmax = atof(argv[i+1]);
      printf("Zmax = %lf\n",box->Zmax);
    }
    else if( strcmp(argv[i],"-Nx") == 0 ){
      box->Nx   = atol(argv[i+1]);
      printf("Nx = %d\n",box->Nx);
    }
    else if( strcmp(argv[i],"-Ny") == 0 ){
      box->Ny   = atol(argv[i+1]);
      printf("Ny = %d\n",box->Ny);
    }
    else if( strcmp(argv[i],"-Nz") == 0 ){
      box->Nz   = atol(argv[i+1]);
      printf("Nz = %d\n",box->Nz);
    }
    else{
      printf("unknown option: %s %s\n",argv[i],argv[i+1]);
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