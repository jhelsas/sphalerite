#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h>
#include <inttypes.h>

#include <omp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_heapsort.h>

#include "../sph_data_types.h"
#include "../sph_linked_list.h"

int main(){

  int j=0,numThreads=6,err;
  int64_t N = 100000;
  double S=0.,S2=0.;
  linkedListBox *box;
  SPHparticle *lsph;

  lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));

  err = gen_unif_rdn_pos( N,123123123,lsph);

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox));

  box->Nx = box->Ny = box->Nz = 10;
  box->N  = (box->Nx)*(box->Ny)*(box->Nz);
  box->Xmin.x = 0.0; box->Xmin.y = 0.0; box->Xmin.z = 0.0;
  box->Xmax.x = 1.0; box->Xmax.y = 1.0; box->Xmax.z = 1.0;
  box->hbegin = kh_init(0);
  box->hend = kh_init(1);

  err = compute_hash_MC3D(N,lsph,box);

  qsort(lsph,N,sizeof(SPHparticle),compare_SPHparticle);

  err = setup_interval_hashtables(N,lsph,box);

  //print_boxes_populations(box);
  //print_neighbour_list_MC3D(N,1,N/13,lsph,box);
  print_neighbour_list_MC3D_lsph_file(N,1,1,lsph,box);

  free(lsph);
  safe_free_box(box);

  return 0;
}