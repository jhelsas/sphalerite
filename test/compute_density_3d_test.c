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

#include "sph_data_types.h"
#include "sph_linked_list.h"
#include "sph_compute.h"

int main(){

  int err,dbg=0;
  int64_t N = 100000;
  double h=0.05,t0,t1;
  linkedListBox *box;
  SPHparticle *lsph;

  if(dbg)
    printf("hello - 0\n");
  err = SPHparticle_SoA_malloc(N,&lsph);
  if(err)
    printf("error in SPHparticle_SoA_malloc\n");

  if(dbg)
    printf("hello - 1\n");
  err = gen_unif_rdn_pos( N,123123123,lsph);
  if(err)
    printf("error in gen_unif_rdn_pos\n");

  if(dbg)
    printf("hello - 2\n");
  box = (linkedListBox*)malloc(1*sizeof(linkedListBox));

  box->Xmin.x = -1.0; box->Xmin.y = -1.0; box->Xmin.z = -1.0;
  box->Xmax.x =  2.0; box->Xmax.y =  2.0; box->Xmax.z =  2.0;
  box->Nx = (int)( (box->Xmax.x-box->Xmin.x)/(2*h) );
  box->Ny = (int)( (box->Xmax.y-box->Xmin.y)/(2*h) );
  box->Nz = (int)( (box->Xmax.z-box->Xmin.z)/(2*h) );
  box->N  = (box->Nx)*(box->Ny)*(box->Nz);
  double min_val = fmin((box->Xmax.x-box->Xmin.x)/box->Nx,fmin((box->Xmax.y-box->Xmin.y)/box->Ny,(box->Xmax.z-box->Xmin.z)/box->Nz));
  box->width = (int)( 0.5 + 2*h/min_val );
  box->w = w_bspline_3d;
  box->hbegin = kh_init(0);
  box->hend = kh_init(1);

  t0 = omp_get_wtime();

  if(dbg)
    printf("hello - 3\n");
  err = compute_hash_MC3D(N,lsph,box);
  
  if(dbg)
    printf("hello - 4\n");

  qsort(lsph->hash,N,2*sizeof(int64_t),compare_int64_t);

  if(dbg)
    printf("hello - 5\n");
  void *swap_arr = malloc(N*sizeof(double));
  err = reorder_lsph_SoA(N,lsph,swap_arr);
  if(err)
    printf("error in reorder_lsph_SoA\n");

  if(dbg)
    printf("hello - 6\n");
  err = setup_interval_hashtables(N,lsph,box);
  if(err)
    printf("error in setup_interval_hashtables\n");

  if(dbg)
    printf("hello - 7\n");
  err = compute_density_3d(N,h,lsph,box);  
  if(err)
    printf("error in setup_interval_hashtables\n");

  t1 = omp_get_wtime();

  printf("Linked-List compute_density_3d calculation time : %lf\n",t1-t0);

  t0 = omp_get_wtime();
  if(dbg)
    printf("hello - 8\n");
  for(int64_t ii=0;ii<N;ii+=1){
    lsph->Fx[ii] = 0;
    for(int64_t jj=0;jj<N;jj+=1){
      double dist = 0.;

      dist += (lsph->x[ii]-lsph->x[jj])*(lsph->x[ii]-lsph->x[jj]);
      dist += (lsph->y[ii]-lsph->y[jj])*(lsph->y[ii]-lsph->y[jj]);
      dist += (lsph->z[ii]-lsph->z[jj])*(lsph->z[ii]-lsph->z[jj]);

      lsph->Fx[ii] += (lsph->nu[jj])*box->w(sqrt(dist),h);
    }
  }

  t1 = omp_get_wtime();
  printf("Reference compute_density_3d calculation time : %lf\n",t1-t0);
  
  if(dbg)
    printf("hello - 9\n");
  FILE *fp = fopen("data/sph_density_compute_ref.csv","w");
  for(int64_t i=0;i<N;i+=1)
    fprintf(fp,"%ld %.12lf %.12lf %.12lf\n",i,
                                            lsph->rho[i],
                                            lsph->Fx[i],
                                            fabs(lsph->rho[i]-lsph->Fx[i]));
  fclose(fp);

  if(dbg)
    printf("hello - 10\n");
  SPHparticleSOA_safe_free(N,&lsph);
  safe_free_box(box);
  free(swap_arr);

  return 0;
}