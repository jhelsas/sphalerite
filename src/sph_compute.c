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

double distance_3d(double xi,double yi,double zi,
                   double xj,double yj,double zj){
  double dist = 0.0;

  dist += (xi-xj)*(xi-xj);
  dist += (yi-yj)*(yi-yj);
  dist += (zi-zj)*(zi-zj);

  return sqrt(dist);
}

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
      if(res!=0)
        return 1;
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          //nb_hash  = nblist[j];
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          for(int64_t ii=node_begin;ii<node_end;ii+=1){ // this loop inside was the problem
            for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
              dist = distance_3d(lsph->x[ii],lsph->y[ii],lsph->z[ii],
                                 lsph->x[jj],lsph->y[jj],lsph->z[jj]);
              lsph->rho[ii] += (lsph->nu[jj])*(box->w(dist,h));
            }
          }
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