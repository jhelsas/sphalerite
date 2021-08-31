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

double w_bspline_3d(double r,double h){
  const double A_d = 3./(2.*M_PI*h*h*h);
  double q=0.;
  
  if(r<0||h<=0.)
    exit(10);
  
  q = r/h;
  if(q<=1)
    return A_d*(2./3.-q*q + q*q*q/2.0);
  else if((1.<=q)&&(q<2.))
    return A_d*(1./6.)*(2.-q)*(2.-q)*(2.-q);
  else 
    return 0.;
}

int compute_density_3d_naive(int N,double h,SPHparticle *lsph){

  for(int64_t ii=0;ii<N;ii+=1){
    lsph[ii].rho = 0;
    for(int64_t jj=0;jj<N;jj+=1){
      double dist = 0.;

      dist += (lsph[ii].r.x-lsph[jj].r.x)*(lsph[ii].r.x-lsph[jj].r.x);
      dist += (lsph[ii].r.y-lsph[jj].r.y)*(lsph[ii].r.y-lsph[jj].r.y);
      dist += (lsph[ii].r.z-lsph[jj].r.z)*(lsph[ii].r.z-lsph[jj].r.z);

      dist = sqrt(dist);

      lsph[ii].rho += lsph[jj].nu*w_bspline_3d(dist,h);
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
  
  compute_density_3d_naive(N,h,lsph);
  
  t1 = omp_get_wtime();

  printf("compute_density_3d SoA naive calc time : %lf s \n",t1-t0);
  
  free(lsph);
  safe_free_box(box);
  
  return 0;
}