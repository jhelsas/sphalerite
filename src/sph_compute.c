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

#pragma omp declare simd
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

#pragma omp declare simd
double dwdq_bspline_3d_simd(double q){
  double wq = 0.0;
  double wq1 = (1.5*q*q - 2.*q);
  double wq2 = -0.5*(2.-q)*(2.-q); 
  
  if(q<2.)
    wq = wq2;

  if(q<1.)
    wq = wq1;
  
  return wq;
}

double dwdq_bspline_3d(double r,double h){
  const double A_d = 3./(2.*M_PI*h*h*h*h);
  double q=0.;
  
  if(r<0||h<=0.)
    exit(10);
  
  q = r/h;
  if(q<=1)
    return A_d*q*(-2.+1.5*q);
  else if((1.<=q)&&(q<2.))
    return -A_d*0.5*(2.-q)*(2.-q);
  else 
    return 0.;
}

double distance_2d(double xi,double yi,
                   double xj,double yj){
  double dist = 0.0;

  dist += (xi-xj)*(xi-xj);
  dist += (yi-yj)*(yi-yj);

  return sqrt(dist);
}

#pragma omp declare simd
double distance_3d(double xi,double yi,double zi,
                   double xj,double yj,double zj){
  double dist = 0.0;

  dist += (xi-xj)*(xi-xj);
  dist += (yi-yj)*(yi-yj);
  dist += (zi-zj)*(zi-zj);

  return sqrt(dist);
}


int compute_density_3d_chunk(int64_t node_begin, int64_t node_end,
                             int64_t nb_begin, int64_t nb_end,double h,
                             double* restrict x, double* restrict y,
                             double* restrict z, double* restrict nu,
                             double* restrict rho){
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);

  #pragma omp parallel for
  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
    double rhoii = 0.0;
   
    #pragma omp simd reduction(+:rhoii)
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      rhoii += nu[jj]*w_bspline_3d_simd(q);// *w_bspline_3d_simd(q); // box->w(sqrt(dist),h);
    }
    rho[ii] += rhoii*kernel_constant;
  }

  return 0;
}

/*
int compute_density_3d_strip(int64_t node_begin, int64_t node_end,
                             int64_t nb_begin, int64_t nb_end,double h,
                             double* restrict x, double* restrict y,
                             double* restrict z, double* restrict nu,
                             double* restrict rho){
  const int64_t STRIP = 8;
  const int64_t node_prime = ((node_end-node_begin) - (node_end-node_begin)%STRIP)+node_begin;
  const int64_t nb_prime = ((nb_end-nb_begin) - (nb_end-nb_begin)%STRIP)+nb_begin;
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);

  #pragma omp parallel for
  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
    double rhoii = 0.0;

    for(int64_t jj=nb_begin;jj<nb_prime;jj+=STRIP){
      #pragma omp simd reduction(+:rhoii)
      for(int64_t j=jj; j<jj+STRIP; j+=1){
        double q = 0.;

        double xij = xii-x[j];
        double yij = yii-y[j];
        double zij = zii-z[j];

        q += xij*xij;
        q += yij*yij;
        q += zij*zij;

        q = sqrt(q)*inv_h;

        rhoii += nu[j]*w_bspline_3d_simd(q);
      }
    }

    #pragma omp simd reduction(+:rhoii)
    for(int64_t j=nb_prime;j<nb_end;j+=1){
      double q = 0.;

      double xij = xii-x[j];
      double yij = yii-y[j];
      double zij = zii-z[j];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      rhoii += nu[j]*w_bspline_3d_simd(q);
    }

    rho[ii] += rhoii*kernel_constant;
  }

  return 0;
}*/

int compute_density_3d(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int res;
  double dist = 0.0;
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0, node_end=0;
  int64_t nb_begin= 0, nb_end = 0;
  int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

  for (kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    
    if (kh_exist(box->hbegin, kbegin)){
      kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));
      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      for(int64_t ii=node_begin;ii<node_end;ii+=1)// this loop inside was the problem
        lsph->rho[ii] = 0.0; 

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          //nb_hash  = nblist[j];
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          compute_density_3d_chunk(node_begin,node_end,nb_begin,nb_end,h,
                                   lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);
          //compute_density_3d_strip(node_begin,node_end,nb_begin,nb_end,h,
          //                         lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);
        }
      }
    }
  }

  return 0;
}

int compute_density_3d_fused(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int res;
  double dist = 0.0;
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0, node_end=0;
  int64_t nb_begin= 0, nb_end = 0;
  
  /*
  for (kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    
    if (kh_exist(box->hbegin, kbegin)){
      kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));
      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      for(int64_t ii=node_begin;ii<node_end;ii+=1)// this loop inside was the problem
        lsph->rho[ii] = 0.0; 

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          //nb_hash  = nblist[j];
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          compute_density_3d_chunk(node_begin,node_end,nb_begin,nb_end,h,
                                   lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);
          //compute_density_3d_strip(node_begin,node_end,nb_begin,nb_end,h,
          //                         lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);
        }
      }
    }
  }*/

  #pragma omp parallel for 
  for(int64_t ii=0;ii<N;ii+=1){
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    kbegin = kh_get(0,box->hbegin,lsph->hash[ii]);
    kend   = kh_get(1,box->hend  ,lsph->hash[ii]);
    node_begin = kh_value(box->hbegin, kbegin);
    node_end   = kh_value(box->hend,   kend);

    lsph->rho[ii] = 0.;
    res = neighbour_hash_3d(lsph->hash[ii],nblist,box->width,box);
    
    for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
      if(nblist[j]>=0){
        nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
        nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

        for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
          dist = distance_3d(lsph->x[ii],lsph->y[ii],lsph->z[ii],
                             lsph->x[jj],lsph->y[jj],lsph->z[jj]);
          lsph->rho[ii] += (lsph->nu[jj])*w_bspline_3d(dist,h);// *(box->w(dist,h));
        }
      }
    }
  }

  return 0;
}


int compute_force_3d_chunk(int64_t node_begin, int64_t node_end,
                           int64_t nb_begin, int64_t nb_end,double h,
                           double* restrict x, double* restrict y,
                           double* restrict z, double* restrict nu,
                           double* restrict rho){
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);

  //#pragma omp parallel for
  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
    double Fx = 0.0;
    double Fy = 0.0;
    double Fz = 0.0;
    double rhoii;
   
    #pragma omp simd reduction(+:rhoii)
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      rhoii += nu[jj]*w_bspline_3d_simd(q);//*w_bspline_3d_simd(q); // box->w(sqrt(dist),h);
    }
    rho[ii] += rhoii*kernel_constant;
  }

  return 0;
}

int compute_force_3d(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int res;
  double dist = 0.0;
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0, node_end=0;
  int64_t nb_begin= 0, nb_end = 0;
  int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

  for (kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    
    if (kh_exist(box->hbegin, kbegin)){
      kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));
      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      for(int64_t ii=node_begin;ii<node_end;ii+=1)// this loop inside was the problem
        lsph->rho[ii] = 0.0; 

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          //nb_hash  = nblist[j];
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          compute_density_3d_chunk(node_begin,node_end,nb_begin,nb_end,h,
                                   lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);
          //compute_density_3d_strip(node_begin,node_end,nb_begin,nb_end,h,
          //                         lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);
        }
      }
    }
  }

  return 0;
}

/*
int compute_force_3d(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int err, res;
  double dist = 0.0;
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0, node_end=0;
  int64_t nb_hash=-1  , nb_begin= 0, nb_end = 0;
  int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

  for (kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    
    if (kh_exist(box->hbegin, kbegin)){
      kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));
      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      for(int64_t ii=node_begin;ii<node_end;ii+=1)// this loop inside was the problem
        lsph->rho[ii] = 0.0; 

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          nb_hash  = nblist[j];
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          for(int64_t ii=node_begin;ii<node_end;ii+=1){ // this loop inside was the problem
            for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
              dist = distance_3d(lsph->x[i],lsph->y[i],lsph->z[i],
                                 lsph->x[j],lsph->y[j],lsph->z[j]);
              lsph->rho[ii] += (lsph->nu[jj])*(box->w(dist,h));
            }
          }
        }
      }
    }
  }

  return 0;
} */

/*
int compute_force_3d(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int err, res;
  double dist = 0.0;
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0, node_end=0;
  int64_t nb_hash=-1  , nb_begin= 0, nb_end = 0;
  int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

  for (kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    
    if (kh_exist(box->hbegin, kbegin)){
      kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));
      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      for(int64_t ii=node_begin;ii<node_end;ii+=1){
        lsph[ii].F.x = 0.0;
        lsph[ii].F.y = 0.0;
        lsph[ii].F.z = 0.0;
      }

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          nb_hash  = nblist[j];
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          for(int64_t ii=node_begin;ii<node_end;ii+=1){ // this loop inside was the problem
            for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
              dist = double4_distance_3d(&(lsph[ii].r),&(lsph[jj].r));
              = ();
              lsph[ii].F.x += ((lsph[ii].r.x-lsph[jj].r.x)/dist);
              lsph[ii].F.y += ;
              lsph[ii].F.z += ;
            }
          }
        }
      }
    }
  }

  return 0;
}
*/