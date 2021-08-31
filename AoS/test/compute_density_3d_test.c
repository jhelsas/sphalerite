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
#include "sph_compute.h"
#include "sph_linked_list.h"

int main(){

  int j=0,err=0,seed=123123123;
  int64_t N = 100000;
  double t0,t1;
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

  t0 = omp_get_wtime();

  printf("computing hashes\n");

  err = compute_hash_MC3D(N,lsph,box);

  printf("sorting the main array\n");

  qsort(lsph,N,sizeof(SPHparticle),compare_SPHparticle);

  printf("setuping hash tables\n");

  err = setup_interval_hashtables(N,lsph,box);

  printf("computing 3d density\n");

  err = compute_density_3d(N,h,lsph,box);

  t1 = omp_get_wtime();

  printf("writing 3d density\n");
  printf("computational time = %lf\n",t1-t0);
  
  FILE *fp = fopen("data/sph_density_compute.csv","w");
  for(int64_t i=0;i<N;i+=1){
    double r=0;

    r += (lsph[i].r.x)*(lsph[i].r.x);
    r += (lsph[i].r.y)*(lsph[i].r.y);
    r += (lsph[i].r.z)*(lsph[i].r.z);
    fprintf(fp,"%lf %.12lf\n",sqrt(r),lsph[i].rho);
  }
  fclose(fp);

  printf("computing 3d density reference\n");
  t0 = omp_get_wtime();
  double kernel_constant = w_bspline_3d_constant(h);
  double inv_h = 1./h;
  #pragma omp parallel for
  for(int64_t ii=0;ii<N;ii+=1){
    lsph[ii].F.x = 0;
    double xii = lsph[ii].r.x;
    double yii = lsph[ii].r.y;
    double zii = lsph[ii].r.z;
    double rhoii = 0.0;

    //#pragma omp simd 
    for(int64_t jj=0;jj<N;jj+=1){
      double q = 0.;

      q += (xii-lsph[jj].r.x)*(xii-lsph[jj].r.x);
      q += (yii-lsph[jj].r.y)*(yii-lsph[jj].r.y);
      q += (zii-lsph[jj].r.z)*(zii-lsph[jj].r.z);

      q = sqrt(q)*inv_h;

      rhoii += (lsph[jj].nu)*w_bspline_3d_simd(q);
    }
    lsph[ii].F.x = kernel_constant*rhoii;
  }
  t1 = omp_get_wtime();
  printf("writing 3d density reference and compute\n");
  printf("computational time ref = %lf\n",t1-t0);

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