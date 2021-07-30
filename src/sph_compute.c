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
  const double A_d = 1.0/(h*h*h);
  double q=0.;
  
  if(r<0||h<=0.)
    exit(10);
  
  q = r/h;
  if(q>=2.)
    return 0;
  else if((1.<=q)&&(q<2.))
    return A_d*(1./6.)*(2.-q)*(2.-q)*(2.-q) ;
  else
    return A_d*(2./3.-q*q + q*q*q/2.0);
}

double double4_distance_2d(double4 ri,double4 rj){
  double dist = 0.0;

  dist += (ri.x-rj.x)*(ri.x-rj.x);
  dist += (ri.y-rj.y)*(ri.y-rj.y);

  return sqrt(dist);
}

double double4_distance_3d(double4 ri,double4 rj){
  double dist = 0.0;

  dist += (ri.x-rj.x)*(ri.x-rj.x);
  dist += (ri.y-rj.y)*(ri.y-rj.y);
  dist += (ri.z-rj.z)*(ri.z-rj.z);

  return sqrt(dist);
}

/*
int compute_density_3d(int N, double h, SPHparticle *lsph, linkedListBox *box){
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

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(int64_t ii=node_begin;ii<node_end;ii+=1){

        lsph[ii].rho = 0.0;
        for(int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
          if(nblist[j]>=0){
            nb_hash  = nblist[j];
            nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
            nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

            for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
              dist = double4_distance_3d(lsph[ii].r,lsph[jj].r); // (&(lsph[ii]),&(lsph[jj]));
              lsph[ii].rho += (lsph[jj].nu)*(box->w(dist,h));
            }
          }
        }
      }
    }
  }

  return 0;
}*/

int compute_density_3d(int N, double h, SPHparticle *lsph, linkedListBox *box){
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
        lsph[ii].rho = 0.0; 

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          nb_hash  = nblist[j];
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          for(int64_t ii=node_begin;ii<node_end;ii+=1){ // this loop inside was the problem
            for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
              dist = double4_distance_3d(lsph[ii].r,lsph[jj].r);//(&(lsph[ii]),&(lsph[jj]));
              lsph[ii].rho += (lsph[jj].nu)*(box->w(dist,h));
            }
          }
        }
      }
    }
  }

  return 0;
}

int compute_density_2d(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int res,err;
  double dist = 0.0;
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0,node_end=0;
  int64_t nb_hash=-1, nb_begin= 0, nb_end = 0;
  int64_t nblist[(2*box->width+1)*(2*box->width+1)];

  for (kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    if (kh_exist(box->hbegin, kbegin)){
      kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));
      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      res = neighbour_hash_2d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          nb_hash  = nblist[j];
          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          for(int64_t i=node_begin;i<node_end;i+=1){
            lsph[i].rho = 0.0;
            for(int64_t j=nb_begin;j<nb_end;j+=1){
              dist = sph_distance_2d(lsph[ii].r,lsph[jj].r);
              lsph[i].rho += (lsph[i].nu)*w_bspline_3d(dist,h);//*(box->w(dist,h));
            }
          }
        }
      }
    }
  }

  return 0;
}


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

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(int64_t ii=node_begin;ii<node_end;ii+=1){

        lsph[ii].rho = 0.0;
        for(int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
          if(nblist[j]>=0){
            nb_hash  = nblist[j];
            nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
            nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

            for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
              dist = double4_distance_3d(lsph[ii].r,lsph[jj].r);
              lsph[ii].rho += (lsph[jj].nu)*(box->w(dist,h));
            }
          }
        }
      }
    }
  }

  return 0;
}
