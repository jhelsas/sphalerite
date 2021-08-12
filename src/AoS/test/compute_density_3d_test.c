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
#include "../sph_compute.h"
#include "../sph_linked_list.h"

int gen_gaussian_pos(int64_t N, int seed, double sigma, SPHparticle *lsph){

  const gsl_rng_type *T=NULL;
  gsl_rng *r=NULL;

  if(lsph==NULL)
    return 1;

  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  gsl_rng_set(r,seed);

  for(int64_t i=0;i<N;i+=1){
    lsph[i].r.x = gsl_ran_gaussian(r,sigma); lsph[i].r.y = gsl_ran_gaussian(r,sigma);
    lsph[i].r.z = gsl_ran_gaussian(r,sigma); lsph[i].r.t = 0.0;

    lsph[i].u.x = 0.0;  lsph[i].u.y = 0.0;
    lsph[i].u.z = 0.0;  lsph[i].u.t = 0.0;

    lsph[i].F.x = 0.0;  lsph[i].F.y = 0.0;
    lsph[i].F.z = 0.0;  lsph[i].F.t = 0.0;

    lsph[i].nu = 1.0; lsph[i].rho  = 0.0;
    lsph[i].id = i;   lsph[i].hash = 0;
  }

  gsl_rng_free(r);

  return 0;
}

int main(){

  int j=0,err=0,seed=123123123;
  int64_t N = 10000;
  double sigma = 1.0,h=0.1,min_val;
  linkedListBox *box;
  SPHparticle *lsph;

  lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));

  //err = gen_gaussian_pos( N,seed,sigma,lsph);
  err = gen_unif_rdn_pos(N,seed,lsph);

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox));

  box->Nx = box->Ny = box->Nz = 100;
  box->N  = (box->Nx)*(box->Ny)*(box->Nz);
  box->Xmin.x = -5.0; box->Xmin.y = -5.0; box->Xmin.z = -5.0;
  box->Xmax.x =  5.0; box->Xmax.y =  5.0; box->Xmax.z =  5.0;
  box->hbegin = kh_init(0);
  box->hend = kh_init(1);
  min_val = fmin((box->Xmax.x-box->Xmin.x)/box->Nx,fmin((box->Xmax.y-box->Xmin.y)/box->Ny,(box->Xmax.z-box->Xmin.z)/box->Nz));
  box->width = (int)( 0.5 + 2*h/min_val );
  box->w = w_bspline_3d;

  printf("computing hashes\n");

  err = compute_hash_MC3D(N,lsph,box);

  printf("sorting the main array\n");

  qsort(lsph,N,sizeof(SPHparticle),compare_SPHparticle);

  printf("setuping hash tables\n");

  err = setup_interval_hashtables(N,lsph,box);

  printf("computing 3d density\n");

  err = compute_density_3d(N,h,lsph,box);

  
  FILE *fp = fopen("data/sph_density_compute.csv","w");
  for(int64_t i=0;i<N;i+=1){
    double r=0;

    r += (lsph[i].r.x)*(lsph[i].r.x);
    r += (lsph[i].r.y)*(lsph[i].r.y);
    r += (lsph[i].r.z)*(lsph[i].r.z);
    fprintf(fp,"%lf %.12lf\n",sqrt(r),lsph[i].rho);
  }
  fclose(fp);

  
  for(int64_t ii=0;ii<N;ii+=1){
    lsph[ii].F.x = 0;
    for(int64_t jj=0;jj<N;jj+=1){
      double dist = 0.;

      dist += (lsph[ii].r.x-lsph[jj].r.x)*(lsph[ii].r.x-lsph[jj].r.x);
      dist += (lsph[ii].r.y-lsph[jj].r.y)*(lsph[ii].r.y-lsph[jj].r.y);
      dist += (lsph[ii].r.z-lsph[jj].r.z)*(lsph[ii].r.z-lsph[jj].r.z);

      lsph[ii].F.x += (lsph[jj].nu)*box->w(sqrt(dist),h);
    }
  }

  fp = fopen("data/sph_density_compute_ref.csv","w");
  for(int64_t i=0;i<N;i+=1){
    double r = 0.;

    r += (lsph[i].r.x)*(lsph[i].r.x);
    r += (lsph[i].r.y)*(lsph[i].r.y);
    r += (lsph[i].r.z)*(lsph[i].r.z);
    fprintf(fp,"%ld %lf %.12lf %.12lf %.12lf\n",i,sqrt(r),lsph[i].rho,lsph[i].F.x,fabs(lsph[i].rho-lsph[i].F.x));
  }
  fclose(fp);

  free(lsph);
  safe_free_box(box);

  return 0;
}