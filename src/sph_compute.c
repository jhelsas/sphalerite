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

double w_bspline_3d(double r,double h)
{
  double R=0.,A_d=0.;
  if(r<0||h<=0.) exit(10);
  
  A_d = 3.0/(2.0*M_PI);
  
  R = r/h;
  if(R>=2.)
    return 0;
  else if((1.<=R)&&(R<2.))
    return (A_d)*(1./6.)*(2.-R)*(2.-R)*(2.-R)/pow(h,D);
  else
    return ((A_d)*((2./3.)-(R*R) + (R*R*R/2.0)))/pow(h,D);
}

double sph_distance_2d(SPHparticle *pi,SPH_particle *pj){
  double dist = 0.0;

  dist += (pi->r.x-pj->r.x)*(pi->r.x-pj->r.x);
  dist += (pi->r.y-pj->r.y)*(pi->r.y-pj->r.y);

  return sqrt(dist);
}

double sph_distance_3d(SPHparticle *pi,SPH_particle *pj){
  double dist = 0.0;

  dist += (pi->r.x-pj->r.x)*(pi->r.x-pj->r.x);
  dist += (pi->r.y-pj->r.y)*(pi->r.y-pj->r.y);
  dist += (pi->r.z-pj->r.z)*(pi->r.z-pj->r.z);

  return sqrt(dist);
}

int compute_density_3d(int N, SPH_particle *lsph, linkedListBox *box){
  int res;
  double dist = 0.0;
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0,node_end=0;
  int64_t nb_hash=-1, nb_begin= 0, nb_end = 0;
  int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

  for (kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    if (kh_exist(box->hbegin, kbegin)){
      kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));
      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      res = neighbour_hash_3d(node_hash,nblist,width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          nb_hash  = nblist[j];
          nb_begin = kh_value(box->hbegin, kh_get(0,box->begin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend, nblist[j]) );

          for(int64_t i=node_begin;i<node_end;i+=1){
            lsph[i].rho = 0.0;
            for(int64_t j=nb_begin;j<nb_end;j+=1){
              dist = sph_distance_3d(&(lsph[i]),&(lsph[j]));
              lsph[i].rho += (lsph[i].nu)*(box->w(r,h));
            }
          }
        }
      }
    }
  }

  return 0;
}

int compute_density_2d(int N, SPH_particle *lsph, linkedListBox *box){
  int res;
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

      res = neighbour_hash_3d(node_hash,nblist,width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          nb_hash  = nblist[j];
          nb_begin = kh_value(box->hbegin, kh_get(0,box->begin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend, nblist[j]) );

          for(int64_t i=node_begin;i<node_end;i+=1){
            lsph[i].rho = 0.0;
            for(int64_t j=nb_begin;j<nb_end;j+=1){
              dist = sph_distance_2d(&(lsph[i]),&(lsph[j]));
              lsph[i].rho += (lsph[i].nu)*(box->w(r,h));
            }
          }
        }
      }
    }
  }

  return 0;
}
