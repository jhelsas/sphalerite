#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h>
#include <inttypes.h>
#include <omp.h>

#include "sph_data_types.h"
#include "sph_linked_list.h"

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

double w_bspline_3d_constant(double h){
  return 3./(2.*M_PI*h*h*h);
}

#pragma omp declare simd
double w_bspline_3d_simd(double q){
  double wq = 0.0;
  double wq1 = (0.6666666666666666 - q*q + 0.5*q*q*q);
  double wq2 = 0.16666666666666666*(2.-q)*(2.-q)*(2.-q); 
  
  if(q<2.)
    wq = wq2;

  if(q<1.)
    wq = wq1;
  
  return wq;
}

int compute_density_3d_naive_omp_simd_tiled(int N,double h,SPHparticle *lsph){
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);
  const int64_t STRIP = 500;
  const int64_t N_prime = N - N%STRIP;

  #pragma omp parallel for 
  for(int64_t ii=0;ii<N;ii+=1)
    lsph[ii].rho = 0.;

  #pragma omp parallel for 
  for(int64_t i=0;i<N_prime;i+=STRIP){
    for(int64_t j=0;j<N_prime;j+=STRIP){
      for(int64_t ii=i;ii<i+STRIP;ii+=1){
        double xii = lsph[ii].r.x;
        double yii = lsph[ii].r.y;
        double zii = lsph[ii].r.z;
        double rhoii = 0.0;

        #pragma omp simd reduction(+:rhoii) 
        for(int64_t jj=j;jj<j+STRIP;jj+=1){
          double q = 0.;

          double xij = xii-lsph[jj].r.x;
          double yij = yii-lsph[jj].r.y;
          double zij = zii-lsph[jj].r.z;  

          q += xij*xij;
          q += yij*yij;
          q += zij*zij;

          q = sqrt(q)*inv_h;

          rhoii += lsph[jj].nu*w_bspline_3d_simd(q);
        }
        lsph[ii].rho = kernel_constant*rhoii;
      }
    }
  }

  #pragma omp parallel for 
  for(int64_t j=0;j<N_prime;j+=STRIP){
    for(int64_t ii=N_prime;ii<N;ii+=1){
      double xii = lsph[ii].r.x;
      double yii = lsph[ii].r.y;
      double zii = lsph[ii].r.z;
      double rhoii = 0.0;

      #pragma omp simd reduction(+:rhoii) 
      for(int64_t jj=j;jj<j+STRIP;jj+=1){
        double q = 0.;

        double xij = xii-lsph[jj].r.x;
        double yij = yii-lsph[jj].r.y;
        double zij = zii-lsph[jj].r.z;

        q += xij*xij;
        q += yij*yij;
        q += zij*zij;

        q = sqrt(q)*inv_h;

        rhoii += lsph[jj].nu*w_bspline_3d_simd(q);
      }
      lsph[ii].rho = kernel_constant*rhoii;
    }
  }
    
  return 0;
}

int main(){

  int err;
  int64_t N = 100000;
  double h=0.05;
  linkedListBox *box;
  SPHparticle *lsph;

  lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));

  err = gen_unif_rdn_pos( N,123123123,lsph);
  if(err)
    printf("error in gen_unif_rdn_pos\n");

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox));

  box->Xmin = -1.0; box->Ymin = -1.0; box->Zmin = -1.0;
  box->Xmax =  2.0; box->Ymax =  2.0; box->Zmax =  2.0;
  box->Nx = (int)( (box->Xmax-box->Xmin)/(2*h) );
  box->Ny = (int)( (box->Ymax-box->Ymin)/(2*h) );
  box->Nz = (int)( (box->Zmax-box->Zmin)/(2*h) );
  box->N  = (box->Nx)*(box->Ny)*(box->Nz);
  double min_val = fmin((box->Xmax-box->Xmin)/box->Nx,fmin((box->Ymax-box->Ymin)/box->Ny,(box->Zmax-box->Zmin)/box->Nz));
  box->width = (int)( 0.5 + 2*h/min_val );
  box->hbegin = kh_init(0);
  box->hend = kh_init(1);

  double t0,t1;
  t0 = omp_get_wtime();
  
  compute_density_3d_naive_omp_simd_tiled(N,h,lsph);
  
  t1 = omp_get_wtime();

  printf("compute_density_3d SoA naive calc time : %lf s \n",t1-t0);
  
  free(lsph);
  safe_free_box(box);
  
  return 0;
}