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

int SPHparticle_SoA_malloc(int N,SPHparticle **lsph){
  bool success=true;
  (*lsph) = (SPHparticle*)malloc(1*sizeof(SPHparticle));

  (*lsph).x    = (double*)malloc(N*sizeof(double));
  (*lsph).y    = (double*)malloc(N*sizeof(double));
  (*lsph).z    = (double*)malloc(N*sizeof(double));

  (*lsph).ux   = (double*)malloc(N*sizeof(double));
  (*lsph).uy   = (double*)malloc(N*sizeof(double));
  (*lsph).uz   = (double*)malloc(N*sizeof(double));

  (*lsph).Fx   = (double*)malloc(N*sizeof(double));
  (*lsph).Fy   = (double*)malloc(N*sizeof(double));
  (*lsph).Fz   = (double*)malloc(N*sizeof(double));

  (*lsph).nu   = (double*)malloc(N*sizeof(double));
  (*lsph).rho  = (double*)malloc(N*sizeof(double));
  
  (*lsph).id   = (int64_t*)malloc(N*sizeof(int64_t));
  (*lsph).hash = (int64_t*)malloc(N*sizeof(int64_t));

  if(success)
    return false;
  else{
    if((*lsph).x != NULL)
      free((*lsph).x);

    if((*lsph).y != NULL)
      free((*lsph).y);

    if((*lsph).z != NULL)
      free((*lsph).z);

    if((*lsph).ux != NULL)
      free((*lsph).ux);

    if((*lsph).uy != NULL)
      free((*lsph).uy);

    if((*lsph).uz != NULL)
      free((*lsph).uz);

    if((*lsph).Fx != NULL)
      free((*lsph).Fx);

    if((*lsph).Fy != NULL)
      free((*lsph).Fz);

    if((*lsph).Fz != NULL)
      free((*lsph).Fz);

    if((*lsph).nu != NULL)
      free((*lsph).nu);

    if((*lsph).rho != NULL)
      free((*lsph).rho);

    if((*lsph).id != NULL)
      free((*lsph).id);

    if((*lsph).hash != NULL)
      free((*lsph).hash);

    return true;
  }
}


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
    lsph->x[i] = gsl_ran_gaussian(r,sigma); 
    lsph->y[i] = gsl_ran_gaussian(r,sigma);
    lsph->z[i] = gsl_ran_gaussian(r,sigma); 

    lsph->ux[i] = 0.0;
    lsph->uy[i] = 0.0;
    lsph->uz[i] = 0.0;

    lsph->Fx[i] = 0.0;
    lsph->Fy[i] = 0.0;
    lsph->Fz[i] = 0.0;

    lsph->nu[i] = 1.0; 
    lsph->rho[i]  = 0.0;
    lsph->id[i] = i;   
    lsph->hash[i] = 0;
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

  //lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));
  err = SPHparticle_SoA_malloc(&lsph);
  if(err)
    printf("error in SPHparticle_SoA_malloc\n");

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

  fp = fopen("data/sph_density_compute_ref.csv","w");
  for(int64_t i=0;i<N;i+=1)
    fprintf(fp,"%ld %.12lf %.12lf %.12lf\n",i,
                                            lsph->rho[i],
                                            lsph->Fx[i],
                                            fabs(lsph->rho[i]-lsph->Fx[i]));
  fclose(fp);

  free(lsph);
  safe_free_box(box);

  return 0;
}