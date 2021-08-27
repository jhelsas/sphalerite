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

int compute_density_3d_naive(int N,double h,
                             double* restrict x, double* restrict y,
                             double* restrict z,double* restrict nu,
                             double* restrict Fx){
  #pragma omp parallel for
  for(int64_t ii=0;ii<N;ii+=1){
    Fx[ii] = 0;
    for(int64_t jj=0;jj<N;jj+=1){
      double dist = 0.;

      dist += (x[ii]-x[jj])*(x[ii]-x[jj]);
      dist += (y[ii]-y[jj])*(y[ii]-y[jj]);
      dist += (z[ii]-z[jj])*(z[ii]-z[jj]);

      dist = sqrt(dist);

      Fx[ii] += nu[jj]*w_bspline_3d(dist,h);
    }
  }

  return 0;
}

int compute_density_3d_ref(int N,double h,
                           double* restrict x, double* restrict y,
                           double* restrict z, double* restrict nu,
                           double* restrict Fx){
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);
  #pragma omp parallel for num_threads(24)
  for(int64_t ii=0;ii<N;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
    double rhoii = 0.0;
    Fx[ii] = 0;
    #pragma omp simd reduction(+:rhoii) aligned(x,y,z,nu)
    for(int64_t jj=0;jj<N;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      //q = sqrt(q);//*inv_h;
      q = sqrt(q)*inv_h;

      //rhoii += nu[jj]*w_bspline_3d(q,1.0);//*w_bspline_3d_simd(q); // box->w(sqrt(dist),h);
      rhoii += nu[jj]*w_bspline_3d_simd(q);//*w_bspline_3d_simd(q); // box->w(sqrt(dist),h);
    }
    //Fx[ii] = kernel_constant*rhoii;
    Fx[ii] = kernel_constant*rhoii;
  }

  return 0;
}

int compute_density_3d_ref_tiled(int N,double h,
                                 double* restrict x, double* restrict y,
                                 double* restrict z, double* restrict nu,
                                 double* restrict Fx){
  const double inv_h = 1./h;
  const double kernel_constant = w_bspline_3d_constant(h);
  const int64_t STRIP_i = 250, STRIP_j = 250;
  const int64_t Ni_prime = N - N%STRIP_i;
  const int64_t Nj_prime = N - N%STRIP_j;

  #pragma omp parallel for num_threads(24)
  for(int64_t i=0;i<Ni_prime;i+=STRIP_i){
    for(int64_t ii=i;ii<i+STRIP_i;ii+=1){
      double xii = x[ii];
      double yii = y[ii];
      double zii = z[ii];
      double rhoii = 0.0;

      #pragma omp simd reduction(+:rhoii) aligned(x,y,z,nu)
      for(int64_t jj=0;jj<N;jj+=1){
        double q = 0.;

        double xij = xii-x[jj];
        double yij = yii-y[jj];
        double zij = zii-z[jj];

        q += xij*xij;
        q += yij*yij;
        q+= zij*zij;

        //q = sqrt(q);//*inv_h;
        q = sqrt(q)*inv_h;

        //rhoii += nu[jj]*w_bspline_3d(q,1.0);//*w_bspline_3d_simd(q); // box->w(sqrt(dist),h);
        rhoii += nu[jj]*w_bspline_3d_simd(q);//*w_bspline_3d_simd(q); // box->w(sqrt(dist),h);
      }
      //Fx[ii] = kernel_constant*rhoii;
      Fx[ii] = kernel_constant*rhoii;
    }
  }

  #pragma omp parallel for num_threads(N-Ni_prime)
  for(int64_t ii=Ni_prime;ii<N;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
    double rhoii = 0.0;

    #pragma omp simd reduction(+:rhoii) aligned(x,y,z,nu)
    for(int64_t jj=0;jj<N;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      //q = sqrt(q);//*inv_h;
      q = sqrt(q)*inv_h;

      //rhoii += nu[jj]*w_bspline_3d(q,1.0);//*w_bspline_3d_simd(q); // box->w(sqrt(dist),h);
      rhoii += nu[jj]*w_bspline_3d_simd(q);//*w_bspline_3d_simd(q); // box->w(sqrt(dist),h);
    }
    //Fx[ii] = kernel_constant*rhoii;
    Fx[ii] = kernel_constant*rhoii;
  }

  /*
  #pragma omp parallel for num_threads(24)
  for(int64_t ii=0;ii<N;ii+=1){
    double xii = x[ii];
    double yii = y[ii];
    double zii = z[ii];
    double rhoii = 0.0;

    #pragma omp simd reduction(+:rhoii) aligned(x,y,z,nu)
    for(int64_t jj=0;jj<N;jj+=1){
      double q = 0.;

      double xij = xii-x[jj];
      double yij = yii-y[jj];
      double zij = zii-z[jj];

      q += xij*xij;
      q += yij*yij;
      q += zij*zij;

      //q = sqrt(q);//*inv_h;
      q = sqrt(q)*inv_h;

      //rhoii += nu[jj]*w_bspline_3d(q,1.0);//*w_bspline_3d_simd(q); // box->w(sqrt(dist),h);
      rhoii += nu[jj]*w_bspline_3d_simd(q);//*w_bspline_3d_simd(q); // box->w(sqrt(dist),h);
    }
    //Fx[ii] = kernel_constant*rhoii;
    Fx[ii] = kernel_constant*rhoii;
  }*/

  return 0;
}

int main(){

  int err,dbg=0;
  int64_t N = 100000;
  double h=0.05;
  linkedListBox *box;
  SPHparticle *lsph;

  omp_set_dynamic(0);              /** Explicitly disable dynamic teams **/
  // omp_set_num_threads(numThreads); /** Use N threads for all parallel regions **/


  if(dbg)
    printf("hello - 0\n");
  err = SPHparticle_SoA_malloc(N,&lsph);
  if(err)
    printf("error in SPHparticle_SoA_malloc\n");

  if(dbg)
    printf("hello - 1\n");
  err = gen_unif_rdn_pos( N,123123123,lsph);
  if(err)
    printf("error in gen_unif_rdn_pos\n");

  if(dbg)
    printf("hello - 2\n");
  box = (linkedListBox*)malloc(1*sizeof(linkedListBox));

  box->Xmin.x = -1.0; box->Xmin.y = -1.0; box->Xmin.z = -1.0;
  box->Xmax.x =  2.0; box->Xmax.y =  2.0; box->Xmax.z =  2.0;
  box->Nx = (int)( (box->Xmax.x-box->Xmin.x)/(2*h) );
  box->Ny = (int)( (box->Xmax.y-box->Xmin.y)/(2*h) );
  box->Nz = (int)( (box->Xmax.z-box->Xmin.z)/(2*h) );
  box->N  = (box->Nx)*(box->Ny)*(box->Nz);
  double min_val = fmin((box->Xmax.x-box->Xmin.x)/box->Nx,fmin((box->Xmax.y-box->Xmin.y)/box->Ny,(box->Xmax.z-box->Xmin.z)/box->Nz));
  box->width = (int)( 0.5 + 2*h/min_val );
  box->w = w_bspline_3d;
  box->hbegin = kh_init(0);
  box->hend = kh_init(1);

  double t0,t1,t2,t3,t4,t5;
  t0 = omp_get_wtime();

  if(dbg)
    printf("hello - 3\n");
  err = compute_hash_MC3D(N,lsph,box);

  t1 = omp_get_wtime();
  
  if(dbg)
    printf("hello - 4\n");

  qsort(lsph->hash,N,2*sizeof(int64_t),compare_int64_t);
  
 /* #pragma omp parallel num_threads(1)
  {
    #pragma omp single 
    quicksort_omp(lsph->hash,0,N);
  }*/

  t2 = omp_get_wtime();

  if(dbg)
    printf("hello - 5\n");
  void *swap_arr = malloc(N*sizeof(double));
  err = reorder_lsph_SoA(N,lsph,swap_arr);
  if(err)
    printf("error in reorder_lsph_SoA\n");


  t3 = omp_get_wtime();

  if(dbg)
    printf("hello - 6\n");
  err = setup_interval_hashtables(N,lsph,box);
  if(err)
    printf("error in setup_interval_hashtables\n");

  t4 = omp_get_wtime();

  if(dbg)
    printf("hello - 7\n");

  err = compute_density_3d(N,h,lsph,box);  
  //err = compute_density_3d_innerOmp(N,h,lsph,box);
  if(err)
    printf("error in setup_interval_hashtables\n");

  t5 = omp_get_wtime();

  printf("compute_hash_MC3D calculation time : %lf : %lf %\n",t1-t0,100*(t1-t0)/(t5-t0));
  printf("qsort calculation time : %lf : %lf %\n",t2-t1,100*(t2-t1)/(t5-t0));
  printf("reorder_lsph_SoA calculation time : %lf : %lf %\n",t3-t2,100*(t3-t2)/(t5-t0));
  printf("setup_interval_hashtables calculation time : %lf : %lf %\n",t4-t3,100*(t4-t3)/(t5-t0));
  printf("compute_density_3d calculation time : %lf : %lf %\n",t5-t4,100*(t5-t4)/(t5-t0));
  printf("Total Linked-List compute_density_3d calculation time : %lf : %lf %\n",t5-t0,100*(t5-t0)/(t5-t0));

  t0 = omp_get_wtime();
  if(dbg)
    printf("hello - 8\n");
  
  //err = compute_density_3d_ref(N,h,lsph->x,lsph->y,lsph->z,lsph->nu,lsph->Fx);
  err = compute_density_3d_ref_tiled(N,h,lsph->x,lsph->y,lsph->z,lsph->nu,lsph->Fx);
  //err = compute_density_3d_naive(N,h,lsph->x,lsph->y,lsph->z,lsph->nu,lsph->Fx);
  if(err)
    printf("error in compute_density_3d_ref\n");

  t1 = omp_get_wtime();
  printf("Reference compute_density_3d calculation time : %lf\n",t1-t0);
  
  if(dbg)
    printf("hello - 9\n");
  FILE *fp = fopen("data/sph_density_compute_ref.csv","w");
  for(int64_t i=0;i<N;i+=1)
    fprintf(fp,"%ld %.12lf %.12lf %.12lf\n",i,
                                            lsph->rho[i],
                                            lsph->Fx[i],
                                            fabs(lsph->rho[i]-lsph->Fx[i]));
  fclose(fp);

  if(dbg)
    printf("hello - 10\n");
  SPHparticleSOA_safe_free(N,&lsph);
  safe_free_box(box);
  free(swap_arr);

  return 0;
}