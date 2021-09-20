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

extern const double fk_bspline_32[32];
extern const double fk_bspline_128[128];
extern const double fk_bspline_1024[1024];

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

// https://stackoverflow.com/questions/56417115/efficient-integer-floor-function-in-c
// 341.333333333 = 1024./3.
#pragma omp declare simd
double w_bspline_3d_LUT_1024(double q){
  int kq = (int)(341.333333333*q);
  double phi = q-kq*0.002929688;
  double wq = phi*fk_bspline_1024[kq] + (1.-phi)*fk_bspline_1024[kq+1];
  
  return wq;
}

#pragma omp declare simd
double w_bspline_3d_LUT_128(double q){
  int kq = (int)(42.666666667*q);
  double phi = q-kq*0.0234375;
  double wq = phi*fk_bspline_128[kq] + (1.-phi)*fk_bspline_128[kq+1];
  
  return wq;
}

#pragma omp declare simd
double w_bspline_3d_LUT_32(double q){
  int kq = (int)(10.666666667*q);
  double phi = q-kq*0.09375;
  double wq = (1.-phi)*fk_bspline_32[kq] + phi*fk_bspline_32[kq+1];
  
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

/**********************************************************************/

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
   
    #pragma omp simd reduction(+:rhoii) nontemporal(rhoii)
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      rhoii += nu[jj]*w_bspline_3d_simd(q);
      //rhoii += nu[jj]*w_bspline_3d_LUT(q);
    }
    rho[ii] += rhoii*kernel_constant;
  }

  return 0;
}

int compute_density_3d_innerOmp(int N, double h, SPHparticle *lsph, linkedListBox *box){
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
        }
      }
    }
  }

  return 0;
}

/**********************************************************************/

int compute_density_3d_chunk_noomp(int64_t node_begin, int64_t node_end,
                                   int64_t nb_begin, int64_t nb_end,double h,
                                   double* restrict x, double* restrict y,
                                   double* restrict z, double* restrict nu,
                                   double* restrict rho){
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);

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

      rhoii += nu[jj]*w_bspline_3d_simd(q);
      //rhoii += nu[jj]*w_bspline_3d_LUT(q);
    }
    rho[ii] += rhoii*kernel_constant;
  }

  return 0;
}

int compute_density_3d(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int64_t pair_count = 0;
  const khint32_t num_boxes = kh_size(box->hbegin);
  const khiter_t hbegin_start = kh_begin(box->hbegin), hbegin_finish = kh_end(box->hbegin);

  
  //for (khint32_t kbegin = hbegin_start; kbegin != hbegin_finish; kbegin++)
  #pragma omp parallel for num_threads(24)
  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    int res;
    int64_t node_hash=-1,node_begin=0, node_end=0;
    int64_t nb_begin= 0, nb_end = 0;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){ // I have to call this!
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));

      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      for(int64_t ii=node_begin;ii<node_end;ii+=1)
        lsph->rho[ii] = 0.0; 

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          //nb_hash  = nblist[j];

          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          pair_count += (node_end-node_begin)*(nb_end-nb_begin);

          compute_density_3d_chunk_noomp(node_begin,node_end,nb_begin,nb_end,h,
                                         lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);
        }
      }
    }

    //printf("thread %d - %lf %lf\n",i,lsph->rho[node_begin],lsph->rho[node_end-1]);
  }

  printf("pair_count = %ld\n",pair_count);

  //for(int64_t i=0;i<N;i+=1000)
  //  printf("%ld - %lf\n",i,lsph->rho[i]);

  return 0;
}

/**********************************************************************/

int compute_density_3d_chunk_loopswapped(int64_t node_begin, int64_t node_end,
                                                int64_t node_hash,linkedListBox *box, double h,
                                                double* restrict x, double* restrict y,
                                                double* restrict z, double* restrict nu,
                                                double* restrict rho)
{
  int64_t pair_count;
  int res=0,nb_count=0,fused_count=0;
  //int64_t nb_begin= 0, nb_end = 0;
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);
  int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];
  int64_t nb_begin[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];
  int64_t nb_end[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

  res = neighbour_hash_3d(node_hash,nblist,box->width,box);
  for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
    if(nblist[j]>=0){
      nb_begin[nb_count] = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
      nb_end[nb_count]   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );
      nb_count+=1;
    }
    else{
      nb_begin[nb_count] = -1;
      nb_end[nb_count]   = -1;
    }
  }

  qsort(nb_begin,nb_count,sizeof(int64_t),compare_int64_t);
  qsort(nb_end  ,nb_count,sizeof(int64_t),compare_int64_t);

  fused_count = nb_count;
  {
    int64_t tmp;
    unsigned int j=0;
    while(j<nb_count-1){
      if(nb_begin[j] < 0){
        nb_begin[j] = nb_begin[nb_count-1]+1;
        nb_end[j]   = nb_end[nb_count-1]+1;
      }
      else if(nb_end[j]==nb_begin[j+1]){
        //printf("%ld:%ld , %ld:%d",nb_begin[j],nb_begin[j+1]);
        nb_end[j] = nb_end[j+1];

        tmp = nb_begin[j];
        nb_begin[j] = nb_begin[j+1];
        nb_begin[j+1] = tmp;
        
        tmp = nb_end[j];
        nb_end[j] = nb_end[j+1];
        nb_end[j+1] = tmp;

        nb_begin[j] = nb_begin[nb_count-1]+1;
        nb_end[j]   = nb_end[nb_count-1]+1;

        fused_count -= 1;
      } 
      
      j+=1;
    }
  }

  qsort(nb_begin,nb_count,sizeof(int64_t),compare_int64_t);
  qsort(nb_end  ,nb_count,sizeof(int64_t),compare_int64_t);

  for(unsigned int j=0;j<fused_count;j+=1){
    pair_count += (node_end-node_begin)*(nb_end[j]-nb_begin[j]);
    for(int64_t ii=node_begin;ii<node_end;ii+=1){
      double xii = x[ii];
      double yii = y[ii];
      double zii = z[ii];

      #pragma omp simd 
      for(int64_t jj=nb_begin[j];jj<nb_end[j];jj+=1){
        double q = 0.;

        double xij = xii-x[jj];
        double yij = yii-y[jj];
        double zij = zii-z[jj];

        q += xij*xij;
        q += yij*yij;
        q += zij*zij;

        q = sqrt(q)*inv_h;

        rho[ii] += nu[jj]*w_bspline_3d_simd(q);
      }        
    }      
  }

  for(int64_t ii=node_begin;ii<node_end;ii+=1)
    rho[ii] *= kernel_constant;

  return pair_count;
}

int compute_density_3d_loopswapped(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int64_t pair_count = 0, ppc;
  const khint32_t num_boxes = kh_size(box->hbegin);
  const khiter_t hbegin_start = kh_begin(box->hbegin), hbegin_finish = kh_end(box->hbegin);

  
  //for (khint32_t kbegin = hbegin_start; kbegin != hbegin_finish; kbegin++)
  #pragma omp parallel for num_threads(24)
  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    int res;
    int64_t node_hash=-1,node_begin=0, node_end=0;

    if (kh_exist(box->hbegin, kbegin)){ // I have to call this!
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));

      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      for(int64_t ii=node_begin;ii<node_end;ii+=1)
        lsph->rho[ii] = 0.0;

      ppc = compute_density_3d_chunk_loopswapped(node_begin,node_end,node_hash,box,h,
                                                 lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);
      pair_count += ppc;
    }
  }

  printf("pair_count = %ld\n",pair_count);

  return 0;
}

/*******************************************************************/

int count_box_pairs(linkedListBox *box){
  int64_t pair_count = 0, particle_pair_count = 0;

  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    int res;
    int64_t node_hash=-1,node_begin=0, node_end=0;
    int64_t nb_begin= 0, nb_end = 0;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){ // I have to call this!
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));

      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          //nb_hash  = nblist[j];

          nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          pair_count += 1;
          particle_pair_count += (node_end-node_begin)*(nb_end-nb_begin);
        }
      }
    }
  }

  printf("unique ordered particle_pair_count = %ld\n",particle_pair_count);

  return pair_count;
}

int setup_box_pairs(linkedListBox *box,
                    int64_t *node_begin,int64_t *node_end,
                    int64_t *nb_begin,int64_t *nb_end)
{
  int64_t pair_count = 0;

  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    int res;
    int64_t node_hash=-1;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){ // I have to call this!
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));

      node_hash = kh_key(box->hbegin, kbegin);

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          //nb_hash  = nblist[j];

          node_begin[pair_count] = kh_value(box->hbegin, kbegin);
          node_end[pair_count]   = kh_value(box->hend, kend);
          nb_begin[pair_count]   = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          nb_end[pair_count]     = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          pair_count += 1;//(node_end-node_begin)*(nb_end-nb_begin);
        }
      }
    }
    //printf("thread %d - %lf %lf\n",i,lsph->rho[node_begin],lsph->rho[node_end-1]);
  }

  return pair_count;
}

int compute_density_3d_chunk_noomp_shift(int64_t node_begin, int64_t node_end,
                                         int64_t nb_begin, int64_t nb_end,double h,
                                         double* restrict x, double* restrict y,
                                         double* restrict z, double* restrict nu,
                                         double* restrict rho){
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);

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

      rhoii += nu[jj]*w_bspline_3d_simd(q);
    }
    rho[ii-node_begin] += rhoii*kernel_constant;
  }

  return 0;
}

int compute_density_3d_load_ballanced(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int64_t *node_begin,*node_end,*nb_begin,*nb_end;
  int64_t max_box_pair_count = 0;

  max_box_pair_count = count_box_pairs(box);
  
  node_begin = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  node_end   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_begin   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_end     = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));

  setup_box_pairs(box,node_begin,node_end,nb_begin,nb_end);

  for(int64_t ii=0;ii<N;ii+=1)
    lsph->rho[ii] = 0.0; 

  #pragma omp parallel for num_threads(24) 
  for(size_t i=0;i<max_box_pair_count;i+=1){
    double local_rho[node_end[i]-node_begin[i]];

    for(size_t ii=0;ii<node_end[i]-node_begin[i];ii+=1)
      local_rho[ii] = 0.;

    compute_density_3d_chunk_noomp_shift(node_begin[i],node_end[i],nb_begin[i],nb_end[i],
                                         h,lsph->x,lsph->y,lsph->z,lsph->nu,local_rho);

    #pragma omp critical
    {
      for(size_t ii=node_begin[i];ii<node_end[i];ii+=1)
        lsph->rho[ii] += local_rho[ii - node_begin[i]];
    }
  }
  
  free(node_begin); 
  free(node_end);
  free(nb_begin);
  free(nb_end);

  //for(int64_t i=0;i<N;i+=1000)
  //  printf("%ld - %lf\n",i,lsph->rho[i]);

  return 0;
}

/*******************************************************************/
/*******************************************************************/

/*
 Pit from hell
 */
int compute_density_3d_chunk_symmetrical(int64_t node_begin, int64_t node_end,
                                         int64_t nb_begin, int64_t nb_end,double h,
                                         double* restrict x, double* restrict y,
                                         double* restrict z, double* restrict nu,
                                         double* restrict rhoi, double* restrict rhoj){
  const double inv_h = 1./h;

  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
   
    #pragma omp simd 
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      double wij = w_bspline_3d_simd(q);

      rhoi[ii-node_begin] += nu[jj]*wij;
      rhoj[jj-nb_begin]   += nu[ii]*wij;
    }
  }

  return 0;
}

int compute_density_3d_chunk_symm_blas(int64_t node_begin, int64_t node_end,
                                         int64_t nb_begin, int64_t nb_end,double h,
                                         double* restrict x, double* restrict y,
                                         double* restrict z, double* restrict nu,
                                         double* restrict rhoi, double* restrict rhoj){
  const double inv_h = 1./h;
  double w[(node_end-node_begin)*(nb_end-nb_begin)];

  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    #pragma omp simd
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = x[ii]-x[jj];
      double yij = y[ii]-y[jj];
      double zij = z[ii]-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      w[(ii-node_begin)*(nb_end-nb_begin)+(jj-nb_begin)] = w_bspline_3d_simd(q);
    }
  }

  // Substituir isso por 2 dgemm
  for(int64_t ii=0;ii<node_end-node_begin;ii+=1){
    #pragma omp simd
    for(int64_t jj=0;jj<nb_end-nb_begin;jj+=1){

      rhoi[ii] += nu[ jj+nb_begin ]*w[ii*(nb_end-nb_begin)+jj];
      rhoj[jj] += nu[ii+node_begin]*w[ii*(nb_end-nb_begin)+jj];
    }
  }

  //rhoi[ii-node_begin] += nu[jj]*wij;
  //rhoj[jj-nb_begin]   += nu[ii]*wij;

  /*
  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
   
    #pragma omp simd 
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      double wij = w_bspline_3d_simd(q);

      rhoi[ii-node_begin] += nu[jj]*wij;
      rhoj[jj-nb_begin]   += nu[ii]*wij;
    }
  }*/

  return 0;
}

int compute_density_3d_chunk_symm_colapse(int64_t node_begin, int64_t node_end,
                                          int64_t nb_begin, int64_t nb_end,double h,
                                          double* restrict x, double* restrict y,
                                          double* restrict z, double* restrict nu,
                                          double* restrict rhoi, double* restrict rhoj){
  const double inv_h = 1./h;

  #pragma omp simd 
  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = x[ii]-x[jj];
      double yij = y[ii]-y[jj];
      double zij = z[ii]-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      double wij = w_bspline_3d_simd(q);

      rhoi[ii-node_begin] += nu[jj]*wij;
      rhoj[jj-nb_begin]   += nu[ii]*wij;
    }
  }

  return 0;
}

int cpr_int64_t(const void *a, const void *b){
  return ( *(int64_t*)a - *(int64_t*)b );
}

int unique_box_bounds(int64_t max_box_pair_count,
                      int64_t *node_begin, int64_t *node_end,
                      int64_t *nb_begin, int64_t *nb_end)
{
  int64_t box_skip=0;
  
  if(node_begin==NULL || node_end==NULL || nb_begin==NULL || nb_end==NULL)
    return -1;

  for(int64_t i=0;i<max_box_pair_count;i+=1){
    if(node_begin[i]>=0){
      for(int64_t j=i+1;j<max_box_pair_count;j+=1){
        if((node_begin[j]==nb_begin[i]) && (nb_begin[j]==node_begin[i]) ){
          node_begin[j] = -1; node_end[j] = -1;
          nb_begin[j]   = -1; nb_end[j]   = -1;

          //node_begin[j] -= -1; node_end[j] -= -1;
          //nb_begin[j]   -= -1; nb_end[j]   -= -1;

          box_skip += 1;
        }
      }
    }
  }

  //qsort(node_begin,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  //qsort(node_end  ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  //qsort(nb_begin  ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  //qsort(nb_end    ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);

  /*
  if(node_begin[i]<=nb_begin[i]){
      
      box_keep +=1;
    }

  qsort(node_begin,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(node_end  ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(nb_begin  ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(nb_end    ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);

  max_box_pair_count = box_keep; //max_box_pair_count - box_subtract;

  for(int64_t i=0;i<max_box_pair_count;i+=1){
    node_begin[i] *= -1; node_end[i] *= -1;
    nb_begin[i]   *= -1; nb_end[i]   *= -1;
  }

  qsort(node_begin,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(node_end  ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(nb_begin  ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(nb_end    ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);*/

  return box_skip;
}

int setup_unique_box_pairs(linkedListBox *box,
                           int64_t *node_begin,int64_t *node_end,
                           int64_t *nb_begin,int64_t *nb_end)
{
  int64_t pair_count = 0;

  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    int res;
    int64_t node_hash=-1;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){ // I have to call this!
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));

      node_hash = kh_key(box->hbegin, kbegin);

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(unsigned int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
        if(nblist[j]>=0){
          //nb_hash  = nblist[j];

          if(kh_value(box->hbegin, kbegin) <= kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) ))
          {
            node_begin[pair_count] = kh_value(box->hbegin, kbegin);
            node_end[pair_count]   = kh_value(box->hend, kend);
            nb_begin[pair_count]   = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
            nb_end[pair_count]     = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

            pair_count += 1;//(node_end-node_begin)*(nb_end-nb_begin);
          }
        }
      }
    }
    //printf("thread %d - %lf %lf\n",i,lsph->rho[node_begin],lsph->rho[node_end-1]);
  }

  return pair_count;
}

int compute_density_3d_symmetrical_load_ballance(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int64_t *node_begin,*node_end,*nb_begin,*nb_end;
  int64_t max_box_pair_count = 0, particle_pair_count = 0;
  int64_t box_skip = 0;
  const double kernel_constant = w_bspline_3d_constant(h);

  max_box_pair_count = count_box_pairs(box);
  printf("max_box_pair_count = %ld\n",max_box_pair_count);
  
  node_begin = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  node_end   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_begin   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_end     = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));

  max_box_pair_count = setup_unique_box_pairs(box,node_begin,node_end,nb_begin,nb_end);//setup_box_pairs(box,node_begin,node_end,nb_begin,nb_end);
  printf("unique max_box_pair_count = %ld\n",max_box_pair_count);

  for(int64_t i=0;i<max_box_pair_count;i+=1)
    particle_pair_count += (node_end[i]-node_begin[i])*(nb_end[i]-nb_begin[i]);
  printf("unique unordered particle_pair_count = %ld\n",particle_pair_count);

  //box_skip = unique_box_bounds(max_box_pair_count,node_begin,node_end,nb_begin,nb_end);

  for(int64_t ii=0;ii<N;ii+=1)
    lsph->rho[ii] = 0.0; 

  #pragma omp parallel for schedule(dynamic,5) proc_bind(master)
  for(size_t i=0;i<max_box_pair_count;i+=1){

    double local_rhoi[node_end[i] - node_begin[i]];
    double local_rhoj[  nb_end[i] -   nb_begin[i]];

    for(size_t ii=0;ii<node_end[i]-node_begin[i];ii+=1)
      local_rhoi[ii] = 0.;

    for(size_t ii=0;ii<nb_end[i]-nb_begin[i];ii+=1)
      local_rhoj[ii] = 0.;

    //compute_density_3d_chunk_symmetrical(node_begin[i],node_end[i],nb_begin[i],nb_end[i],h,
    //                                     lsph->x,lsph->y,lsph->z,lsph->nu,local_rhoi,local_rhoj);

    compute_density_3d_chunk_symm_blas(node_begin[i],node_end[i],nb_begin[i],nb_end[i],h,
                                       lsph->x,lsph->y,lsph->z,lsph->nu,local_rhoi,local_rhoj);

    //compute_density_3d_chunk_symm_colapse(node_begin[i],node_end[i],nb_begin[i],nb_end[i],h,
    //                                      lsph->x,lsph->y,lsph->z,lsph->nu,local_rhoi,local_rhoj);

    #pragma omp critical
    {

      for(size_t ii=node_begin[i];ii<node_end[i];ii+=1){
        //#pragma omp atomic
        lsph->rho[ii] += kernel_constant*local_rhoi[ii - node_begin[i]];
      }
      
      if(node_begin[i] != nb_begin[i])
        for(size_t ii=nb_begin[i];ii<nb_end[i];ii+=1){
          //#pragma omp atomic
          lsph->rho[ii] += kernel_constant*local_rhoj[ii - nb_begin[i]];
        }
    }
  }
  
  free(node_begin); 
  free(node_end);
  free(nb_begin);
  free(nb_end);

  return 0;
}

int compute_density_3d_symmetrical_lb_branching(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int64_t *node_begin,*node_end,*nb_begin,*nb_end;
  int64_t max_box_pair_count = 0, particle_pair_count = 0;
  int64_t box_skip = 0;
  const double kernel_constant = w_bspline_3d_constant(h);

  max_box_pair_count = count_box_pairs(box);
  printf("max_box_pair_count = %ld\n",max_box_pair_count);
  
  node_begin = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  node_end   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_begin   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_end     = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));

  max_box_pair_count = setup_box_pairs(box,node_begin,node_end,nb_begin,nb_end);

  box_skip = unique_box_bounds(max_box_pair_count,node_begin,node_end,nb_begin,nb_end);

  for(int64_t i=0;i<max_box_pair_count;i+=1)
    if(node_end[i] >= 0)
      particle_pair_count += (node_end[i]-node_begin[i])*(nb_end[i]-nb_begin[i]);
  printf("unique unordered particle_pair_count = %ld\n",particle_pair_count);

  for(int64_t ii=0;ii<N;ii+=1)
    lsph->rho[ii] = 0.0; 

  #pragma omp parallel for 
  for(size_t i=0;i<max_box_pair_count;i+=1){
    if(node_end[i]<0)
      continue;

    double local_rhoi[node_end[i] - node_begin[i]];
    double local_rhoj[  nb_end[i] -   nb_begin[i]];

    for(size_t ii=0;ii<node_end[i]-node_begin[i];ii+=1)
      local_rhoi[ii] = 0.;

    for(size_t ii=0;ii<nb_end[i]-nb_begin[i];ii+=1)
      local_rhoj[ii] = 0.;

    compute_density_3d_chunk_symmetrical(node_begin[i],node_end[i],nb_begin[i],nb_end[i],h,
                                         lsph->x,lsph->y,lsph->z,lsph->nu,local_rhoi,local_rhoj);

    #pragma omp critical
    {
      for(size_t ii=node_begin[i];ii<node_end[i];ii+=1)
        lsph->rho[ii] += kernel_constant*local_rhoi[ii - node_begin[i]];

      
      if(node_begin[i] != nb_begin[i])
        for(size_t ii=nb_begin[i];ii<nb_end[i];ii+=1)
          lsph->rho[ii] += kernel_constant*local_rhoj[ii - nb_begin[i]];

      /*
      if(node_begin[i]!=nb_begin[i])
        for(size_t ii=nb_begin[i];ii<nb_end[i];ii+=1)
          lsph->rho[ii] += kernel_constant*local_rhoj[ii - nb_begin[i]];*/
    }
  }
  
  free(node_begin); 
  free(node_end);
  free(nb_begin);
  free(nb_end);

  return 0;
}

/*******************************************************************/
/*******************************************************************/

/*
int compute_density_3d_chunk_symmetrical(int64_t node_begin, int64_t node_end,
                                         int64_t nb_begin, int64_t nb_end,double h,
                                         double* restrict x, double* restrict y,
                                         double* restrict z, double* restrict nu,
                                         double* restrict rho)
{
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);

  double rhoi[node_end-node_begin];
  double rhoj[nb_end-nb_begin];

  for(int64_t ii=0;ii<node_end-node_begin;ii+=1)
    rhoi[ii] = 0.;

  for(int64_t jj=0;jj<nb_end-nb_begin;jj+=1)
    rhoj[jj] = 0.;

  for(int64_t ii=node_begin;ii<node_end;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
   
    #pragma omp simd 
    for(int64_t jj=nb_begin;jj<nb_end;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      q = sqrt(q)*inv_h;

      double wij = w_bspline_3d_simd(q);

      rhoi[ii-node_begin] += nu[jj]*wij;
      rhoj[jj-nb_begin]   += nu[ii]*wij;
    }
  }

    for(int64_t ii=node_begin;ii<node_end;ii+=1)
      rho[ii] += rhoi[ii-node_begin]*kernel_constant;
 
  if(nb_begin == node_begin)
    return 0;

    for(int64_t jj=nb_begin;jj<nb_end;jj+=1)
      rho[jj] += rhoj[jj-nb_begin]*kernel_constant;

  return 0;
}*/

/*
int cpr_int64_t(const void *a, const void *b){
  return ( *(int64_t*)a - *(int64_t*)b );
} */

/*
int unique_box_bounds(int64_t max_box_pair_count,
                      int64_t *node_begin, int64_t *node_end,
                      int64_t *nb_begin, int64_t *nb_end)
{
  int64_t box_keep=0;

  if(node_begin==NULL || node_end==NULL || nb_begin==NULL || nb_end==NULL)
    return -1;

  for(int64_t i=0;i<max_box_pair_count;i+=1)
    if(node_begin[i]<=nb_begin[i]){
      node_begin[i] *= -1; node_end[i] *= -1;
      nb_begin[i]   *= -1; nb_end[i]   *= -1;
      box_keep +=1;
    }

  qsort(node_begin,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(node_end  ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(nb_begin  ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(nb_end    ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);

  max_box_pair_count = box_keep; //max_box_pair_count - box_subtract;

  for(int64_t i=0;i<max_box_pair_count;i+=1){
    node_begin[i] *= -1; node_end[i] *= -1;
    nb_begin[i]   *= -1; nb_end[i]   *= -1;
  }

  qsort(node_begin,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(node_end  ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(nb_begin  ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);
  qsort(nb_end    ,max_box_pair_count, sizeof(int64_t),cpr_int64_t);

  return max_box_pair_count;
}

int compute_density_3d_symmetrical_lb(int N, double h, SPHparticle *lsph, linkedListBox *box){
  int64_t *node_begin,*node_end,*nb_begin,*nb_end;
  int64_t max_box_pair_count = 0;

  max_box_pair_count = count_box_pairs(box);
  
  node_begin = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  node_end   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_begin   = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));
  nb_end     = (int64_t*)malloc(max_box_pair_count*sizeof(int64_t));

  setup_box_pairs(box,node_begin,node_end,nb_begin,nb_end);

  max_box_pair_count = unique_box_bounds(max_box_pair_count,node_begin,node_end,nb_begin,nb_end);

  for(int64_t ii=0;ii<N;ii+=1)
    lsph->rho[ii] = 0.0;

  #pragma omp parallel for num_threads(24) 
  for(size_t i=0;i<max_box_pair_count;i+=1){
    if(node_begin[i] > nb_begin[i])
      continue; 

    compute_density_3d_chunk_symmetrical(node_begin[i],node_end[i],nb_begin[i],nb_end[i],
                                         h,lsph->x,lsph->y,lsph->z,lsph->nu,lsph->rho);
    }

  free(node_begin); 
  free(node_end);
  free(nb_begin);
  free(nb_end);

  //for(int64_t i=0;i<N;i+=1000)
  //  printf("%ld - %lf\n",i,lsph->rho[i]);

  return 0;
}*/

/*******************************************************************/
/*******************************************************************/
/*******************************************************************/

#pragma omp declare simd
double pressure_from_density(double rho){
  double p = cbrt(rho);
  p = 0.5*p*rho;
  return p;
}

#pragma omp declare simd
double gamma_from_u(double ux,double uy,double uz){
  double gamma = 1.0;
  gamma += ux*ux;
  gamma += uy*uy;
  gamma += uz*uz;

  gamma = sqrt(gamma);

  return gamma;
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

const double fk_bspline_32[32] = { 
       6.66666667e-01, 6.57754579e-01, 6.32830944e-01, 5.94614705e-01,
       5.45824802e-01, 4.89180177e-01, 4.27399774e-01, 3.63202533e-01,
       2.99307397e-01, 2.38433308e-01, 1.83299207e-01, 1.36445011e-01,
       9.83294731e-02, 6.80686561e-02, 4.47562463e-02, 2.74859298e-02,
       1.53513925e-02, 7.44632048e-03, 2.86439976e-03, 6.99316348e-04,
       4.47562463e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00};

const double fk_bspline_128[128] = { 
       6.66666667e-01, 6.66115256e-01, 6.64487387e-01, 6.61822602e-01,
       6.58160445e-01, 6.53540459e-01, 6.48002188e-01, 6.41585176e-01,
       6.34328964e-01, 6.26273098e-01, 6.17457119e-01, 6.07920573e-01,
       5.97703001e-01, 5.86843948e-01, 5.75382957e-01, 5.63359570e-01,
       5.50813333e-01, 5.37783787e-01, 5.24310476e-01, 5.10432945e-01,
       4.96190735e-01, 4.81623391e-01, 4.66770456e-01, 4.51671473e-01,
       4.36365986e-01, 4.20893537e-01, 4.05293671e-01, 3.89605931e-01,
       3.73869861e-01, 3.58125002e-01, 3.42410900e-01, 3.26767097e-01,
       3.11233137e-01, 2.95848563e-01, 2.80652918e-01, 2.65685747e-01,
       2.50986591e-01, 2.36594995e-01, 2.22550503e-01, 2.08892657e-01,
       1.95661000e-01, 1.82895077e-01, 1.70634431e-01, 1.58916000e-01,
       1.47746458e-01, 1.37112949e-01, 1.27002291e-01, 1.17401303e-01,
       1.08296805e-01, 9.96756140e-02, 9.15245505e-02, 8.38304328e-02,
       7.65800797e-02, 6.97603101e-02, 6.33579430e-02, 5.73597971e-02,
       5.17526914e-02, 4.65234448e-02, 4.16588760e-02, 3.71458040e-02,
       3.29710476e-02, 2.91214257e-02, 2.55837572e-02, 2.23448610e-02,
       1.93915558e-02, 1.67106607e-02, 1.42889945e-02, 1.21133759e-02,
       1.01706240e-02, 8.44755758e-03, 6.93099549e-03, 5.60775662e-03,
       4.46465985e-03, 3.48852404e-03, 2.66616806e-03, 1.98441079e-03,
       1.43007110e-03, 9.89967859e-04, 6.50919937e-04, 3.99746206e-04,
       2.23265538e-04, 1.08296805e-04, 4.16588760e-05, 1.01706240e-05,
       6.50919937e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00};

const double fk_bspline_1024[1024] = {
       6.66666667e-01, 6.66658079e-01, 6.66632368e-01, 6.66589608e-01,
       6.66529876e-01, 6.66453246e-01, 6.66359796e-01, 6.66249599e-01,
       6.66122732e-01, 6.65979271e-01, 6.65819291e-01, 6.65642868e-01,
       6.65450077e-01, 6.65240994e-01, 6.65015696e-01, 6.64774257e-01,
       6.64516753e-01, 6.64243260e-01, 6.63953853e-01, 6.63648609e-01,
       6.63327602e-01, 6.62990909e-01, 6.62638605e-01, 6.62270765e-01,
       6.61887466e-01, 6.61488783e-01, 6.61074792e-01, 6.60645569e-01,
       6.60201188e-01, 6.59741726e-01, 6.59267259e-01, 6.58777861e-01,
       6.58273610e-01, 6.57754579e-01, 6.57220846e-01, 6.56672485e-01,
       6.56109573e-01, 6.55532184e-01, 6.54940396e-01, 6.54334282e-01,
       6.53713920e-01, 6.53079384e-01, 6.52430750e-01, 6.51768095e-01,
       6.51091493e-01, 6.50401020e-01, 6.49696752e-01, 6.48978765e-01,
       6.48247134e-01, 6.47501935e-01, 6.46743244e-01, 6.45971135e-01,
       6.45185686e-01, 6.44386971e-01, 6.43575066e-01, 6.42750048e-01,
       6.41911990e-01, 6.41060970e-01, 6.40197063e-01, 6.39320344e-01,
       6.38430889e-01, 6.37528774e-01, 6.36614075e-01, 6.35686866e-01,
       6.34747225e-01, 6.33795226e-01, 6.32830944e-01, 6.31854457e-01,
       6.30865839e-01, 6.29865166e-01, 6.28852514e-01, 6.27827959e-01,
       6.26791575e-01, 6.25743439e-01, 6.24683626e-01, 6.23612213e-01,
       6.22529274e-01, 6.21434885e-01, 6.20329123e-01, 6.19212062e-01,
       6.18083778e-01, 6.16944347e-01, 6.15793845e-01, 6.14632348e-01,
       6.13459930e-01, 6.12276668e-01, 6.11082637e-01, 6.09877913e-01,
       6.08662571e-01, 6.07436688e-01, 6.06200339e-01, 6.04953599e-01,
       6.03696545e-01, 6.02429251e-01, 6.01151794e-01, 5.99864249e-01,
       5.98566692e-01, 5.97259199e-01, 5.95941844e-01, 5.94614705e-01,
       5.93277856e-01, 5.91931373e-01, 5.90575332e-01, 5.89209808e-01,
       5.87834877e-01, 5.86450616e-01, 5.85057098e-01, 5.83654401e-01,
       5.82242599e-01, 5.80821769e-01, 5.79391986e-01, 5.77953326e-01,
       5.76505864e-01, 5.75049676e-01, 5.73584838e-01, 5.72111425e-01,
       5.70629514e-01, 5.69139179e-01, 5.67640496e-01, 5.66133541e-01,
       5.64618390e-01, 5.63095118e-01, 5.61563801e-01, 5.60024515e-01,
       5.58477335e-01, 5.56922337e-01, 5.55359597e-01, 5.53789190e-01,
       5.52211192e-01, 5.50625678e-01, 5.49032725e-01, 5.47432408e-01,
       5.45824802e-01, 5.44209983e-01, 5.42588027e-01, 5.40959010e-01,
       5.39323007e-01, 5.37680094e-01, 5.36030346e-01, 5.34373840e-01,
       5.32710650e-01, 5.31040853e-01, 5.29364524e-01, 5.27681738e-01,
       5.25992573e-01, 5.24297102e-01, 5.22595402e-01, 5.20887548e-01,
       5.19173617e-01, 5.17453683e-01, 5.15727823e-01, 5.13996112e-01,
       5.12258626e-01, 5.10515440e-01, 5.08766630e-01, 5.07012271e-01,
       5.05252441e-01, 5.03487213e-01, 5.01716663e-01, 4.99940869e-01,
       4.98159904e-01, 4.96373845e-01, 4.94582767e-01, 4.92786746e-01,
       4.90985857e-01, 4.89180177e-01, 4.87369781e-01, 4.85554745e-01,
       4.83735144e-01, 4.81911054e-01, 4.80082550e-01, 4.78249708e-01,
       4.76412605e-01, 4.74571315e-01, 4.72725914e-01, 4.70876478e-01,
       4.69023083e-01, 4.67165804e-01, 4.65304717e-01, 4.63439897e-01,
       4.61571420e-01, 4.59699362e-01, 4.57823799e-01, 4.55944806e-01,
       4.54062459e-01, 4.52176833e-01, 4.50288004e-01, 4.48396048e-01,
       4.46501040e-01, 4.44603057e-01, 4.42702173e-01, 4.40798465e-01,
       4.38892008e-01, 4.36982877e-01, 4.35071149e-01, 4.33156899e-01,
       4.31240203e-01, 4.29321136e-01, 4.27399774e-01, 4.25476193e-01,
       4.23550468e-01, 4.21622675e-01, 4.19692890e-01, 4.17761188e-01,
       4.15827645e-01, 4.13892336e-01, 4.11955338e-01, 4.10016726e-01,
       4.08076576e-01, 4.06134962e-01, 4.04191962e-01, 4.02247650e-01,
       4.00302103e-01, 3.98355395e-01, 3.96407603e-01, 3.94458803e-01,
       3.92509069e-01, 3.90558477e-01, 3.88607104e-01, 3.86655025e-01,
       3.84702315e-01, 3.82749050e-01, 3.80795307e-01, 3.78841159e-01,
       3.76886684e-01, 3.74931957e-01, 3.72977053e-01, 3.71022048e-01,
       3.69067018e-01, 3.67112038e-01, 3.65157185e-01, 3.63202533e-01,
       3.61248159e-01, 3.59294138e-01, 3.57340545e-01, 3.55387457e-01,
       3.53434949e-01, 3.51483097e-01, 3.49531976e-01, 3.47581662e-01,
       3.45632230e-01, 3.43683758e-01, 3.41736319e-01, 3.39789989e-01,
       3.37844845e-01, 3.35900962e-01, 3.33958416e-01, 3.32017282e-01,
       3.30077636e-01, 3.28139553e-01, 3.26203110e-01, 3.24268382e-01,
       3.22335444e-01, 3.20404373e-01, 3.18475243e-01, 3.16548131e-01,
       3.14623112e-01, 3.12700262e-01, 3.10779657e-01, 3.08861372e-01,
       3.06945483e-01, 3.05032065e-01, 3.03121194e-01, 3.01212946e-01,
       2.99307397e-01, 2.97404622e-01, 2.95504697e-01, 2.93607697e-01,
       2.91713698e-01, 2.89822776e-01, 2.87935006e-01, 2.86050465e-01,
       2.84169227e-01, 2.82291369e-01, 2.80416966e-01, 2.78546093e-01,
       2.76678827e-01, 2.74815243e-01, 2.72955417e-01, 2.71099424e-01,
       2.69247340e-01, 2.67399241e-01, 2.65555202e-01, 2.63715299e-01,
       2.61879608e-01, 2.60048204e-01, 2.58221163e-01, 2.56398561e-01,
       2.54580473e-01, 2.52766975e-01, 2.50958142e-01, 2.49154051e-01,
       2.47354777e-01, 2.45560395e-01, 2.43770982e-01, 2.41986612e-01,
       2.40207362e-01, 2.38433308e-01, 2.36664524e-01, 2.34901086e-01,
       2.33143071e-01, 2.31390554e-01, 2.29643610e-01, 2.27902316e-01,
       2.26166746e-01, 2.24436977e-01, 2.22713084e-01, 2.20995143e-01,
       2.19283229e-01, 2.17577418e-01, 2.15877786e-01, 2.14184409e-01,
       2.12497361e-01, 2.10816720e-01, 2.09142560e-01, 2.07474956e-01,
       2.05813986e-01, 2.04159724e-01, 2.02512246e-01, 2.00871628e-01,
       1.99237945e-01, 1.97611273e-01, 1.95991688e-01, 1.94379265e-01,
       1.92774080e-01, 1.91176209e-01, 1.89585728e-01, 1.88002711e-01,
       1.86427235e-01, 1.84859375e-01, 1.83299207e-01, 1.81746806e-01,
       1.80202249e-01, 1.78665611e-01, 1.77136968e-01, 1.75616394e-01,
       1.74103967e-01, 1.72599761e-01, 1.71103853e-01, 1.69616317e-01,
       1.68137230e-01, 1.66666667e-01, 1.65204687e-01, 1.63751281e-01,
       1.62306426e-01, 1.60870094e-01, 1.59442261e-01, 1.58022902e-01,
       1.56611992e-01, 1.55209505e-01, 1.53815416e-01, 1.52429700e-01,
       1.51052331e-01, 1.49683285e-01, 1.48322536e-01, 1.46970060e-01,
       1.45625830e-01, 1.44289821e-01, 1.42962009e-01, 1.41642368e-01,
       1.40330873e-01, 1.39027499e-01, 1.37732220e-01, 1.36445011e-01,
       1.35165848e-01, 1.33894704e-01, 1.32631555e-01, 1.31376375e-01,
       1.30129139e-01, 1.28889822e-01, 1.27658399e-01, 1.26434845e-01,
       1.25219133e-01, 1.24011240e-01, 1.22811140e-01, 1.21618807e-01,
       1.20434217e-01, 1.19257343e-01, 1.18088162e-01, 1.16926648e-01,
       1.15772775e-01, 1.14626518e-01, 1.13487852e-01, 1.12356752e-01,
       1.11233193e-01, 1.10117149e-01, 1.09008596e-01, 1.07907507e-01,
       1.06813859e-01, 1.05727624e-01, 1.04648779e-01, 1.03577299e-01,
       1.02513157e-01, 1.01456328e-01, 1.00406788e-01, 9.93645117e-02,
       9.83294731e-02, 9.73016473e-02, 9.62810090e-02, 9.52675330e-02,
       9.42611942e-02, 9.32619673e-02, 9.22698271e-02, 9.12847483e-02,
       9.03067058e-02, 8.93356743e-02, 8.83716286e-02, 8.74145435e-02,
       8.64643938e-02, 8.55211542e-02, 8.45847996e-02, 8.36553047e-02,
       8.27326442e-02, 8.18167931e-02, 8.09077259e-02, 8.00054177e-02,
       7.91098430e-02, 7.82209767e-02, 7.73387936e-02, 7.64632684e-02,
       7.55943760e-02, 7.47320911e-02, 7.38763885e-02, 7.30272430e-02,
       7.21846293e-02, 7.13485223e-02, 7.05188966e-02, 6.96957272e-02,
       6.88789888e-02, 6.80686561e-02, 6.72647039e-02, 6.64671071e-02,
       6.56758404e-02, 6.48908785e-02, 6.41121963e-02, 6.33397686e-02,
       6.25735701e-02, 6.18135756e-02, 6.10597598e-02, 6.03120976e-02,
       5.95705638e-02, 5.88351331e-02, 5.81057803e-02, 5.73824802e-02,
       5.66652075e-02, 5.59539371e-02, 5.52486438e-02, 5.45493022e-02,
       5.38558872e-02, 5.31683736e-02, 5.24867361e-02, 5.18109496e-02,
       5.11409888e-02, 5.04768285e-02, 4.98184434e-02, 4.91658084e-02,
       4.85188982e-02, 4.78776876e-02, 4.72421515e-02, 4.66122645e-02,
       4.59880014e-02, 4.53693371e-02, 4.47562463e-02, 4.41487038e-02,
       4.35466844e-02, 4.29501628e-02, 4.23591138e-02, 4.17735123e-02,
       4.11933330e-02, 4.06185507e-02, 4.00491401e-02, 3.94850760e-02,
       3.89263333e-02, 3.83728867e-02, 3.78247109e-02, 3.72817808e-02,
       3.67440712e-02, 3.62115568e-02, 3.56842123e-02, 3.51620127e-02,
       3.46449326e-02, 3.41329469e-02, 3.36260303e-02, 3.31241576e-02,
       3.26273035e-02, 3.21354430e-02, 3.16485507e-02, 3.11666014e-02,
       3.06895699e-02, 3.02174310e-02, 2.97501595e-02, 2.92877301e-02,
       2.88301177e-02, 2.83772970e-02, 2.79292427e-02, 2.74859298e-02,
       2.70473328e-02, 2.66134267e-02, 2.61841863e-02, 2.57595862e-02,
       2.53396013e-02, 2.49242063e-02, 2.45133761e-02, 2.41070854e-02,
       2.37053089e-02, 2.33080216e-02, 2.29151981e-02, 2.25268132e-02,
       2.21428418e-02, 2.17632586e-02, 2.13880383e-02, 2.10171558e-02,
       2.06505858e-02, 2.02883032e-02, 1.99302826e-02, 1.95764990e-02,
       1.92269270e-02, 1.88815414e-02, 1.85403171e-02, 1.82032287e-02,
       1.78702512e-02, 1.75413592e-02, 1.72165275e-02, 1.68957310e-02,
       1.65789443e-02, 1.62661424e-02, 1.59572999e-02, 1.56523917e-02,
       1.53513925e-02, 1.50542771e-02, 1.47610203e-02, 1.44715968e-02,
       1.41859815e-02, 1.39041492e-02, 1.36260745e-02, 1.33517323e-02,
       1.30810974e-02, 1.28141446e-02, 1.25508485e-02, 1.22911841e-02,
       1.20351261e-02, 1.17826493e-02, 1.15337284e-02, 1.12883382e-02,
       1.10464536e-02, 1.08080492e-02, 1.05731000e-02, 1.03415805e-02,
       1.01134657e-02, 9.88873037e-03, 9.66734920e-03, 9.44929700e-03,
       9.23454856e-03, 9.02307866e-03, 8.81486208e-03, 8.60987360e-03,
       8.40808799e-03, 8.20948005e-03, 8.01402454e-03, 7.82169626e-03,
       7.63246998e-03, 7.44632048e-03, 7.26322254e-03, 7.08315094e-03,
       6.90608047e-03, 6.73198590e-03, 6.56084202e-03, 6.39262360e-03,
       6.22730542e-03, 6.06486228e-03, 5.90526893e-03, 5.74850018e-03,
       5.59453079e-03, 5.44333554e-03, 5.29488923e-03, 5.14916663e-03,
       5.00614251e-03, 4.86579166e-03, 4.72808886e-03, 4.59300890e-03,
       4.46052654e-03, 4.33061658e-03, 4.20325378e-03, 4.07841294e-03,
       3.95606884e-03, 3.83619624e-03, 3.71876994e-03, 3.60376471e-03,
       3.49115534e-03, 3.38091660e-03, 3.27302328e-03, 3.16745016e-03,
       3.06417201e-03, 2.96316362e-03, 2.86439976e-03, 2.76785523e-03,
       2.67350479e-03, 2.58132323e-03, 2.49128533e-03, 2.40336587e-03,
       2.31753963e-03, 2.23378139e-03, 2.15206594e-03, 2.07236804e-03,
       1.99466249e-03, 1.91892406e-03, 1.84512753e-03, 1.77324769e-03,
       1.70325931e-03, 1.63513718e-03, 1.56885607e-03, 1.50439077e-03,
       1.44171605e-03, 1.38080670e-03, 1.32163749e-03, 1.26418322e-03,
       1.20841865e-03, 1.15431857e-03, 1.10185776e-03, 1.05101100e-03,
       1.00175307e-03, 9.54058747e-04, 9.07902817e-04, 8.63260059e-04,
       8.20105252e-04, 7.78413178e-04, 7.38158617e-04, 6.99316348e-04,
       6.61861154e-04, 6.25767814e-04, 5.91011108e-04, 5.57565818e-04,
       5.25406723e-04, 4.94508604e-04, 4.64846242e-04, 4.36394418e-04,
       4.09127910e-04, 3.83021501e-04, 3.58049970e-04, 3.34188099e-04,
       3.11410666e-04, 2.89692454e-04, 2.69008242e-04, 2.49332811e-04,
       2.30640942e-04, 2.12907414e-04, 1.96107009e-04, 1.80214506e-04,
       1.65204687e-04, 1.51052331e-04, 1.37732220e-04, 1.25219133e-04,
       1.13487852e-04, 1.02513157e-04, 9.22698271e-05, 8.27326442e-05,
       7.38763885e-05, 6.56758404e-05, 5.81057803e-05, 5.11409888e-05,
       4.47562463e-05, 3.89263333e-05, 3.36260303e-05, 2.88301177e-05,
       2.45133761e-05, 2.06505858e-05, 1.72165275e-05, 1.41859815e-05,
       1.15337284e-05, 9.23454856e-06, 7.26322254e-06, 5.59453079e-06,
       4.20325378e-06, 3.06417201e-06, 2.15206594e-06, 1.44171605e-06,
       9.07902817e-07, 5.25406723e-07, 2.69008242e-07, 1.13487852e-07,
       3.36260303e-08, 4.20325378e-09, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00};