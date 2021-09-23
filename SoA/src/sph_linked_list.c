#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/time.h>
#include <inttypes.h>

#include <omp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_heapsort.h>

#include "MZC3D64.h"
#include "MZC2D64.h"
#include "sph_data_types.h"
#include "sph_linked_list.h"

int safe_free_box(linkedListBox *box){
	kh_destroy(0, box->hbegin);
	kh_destroy(1, box->hend);
	free(box);

	return 0;
}

int compare_int64_t(const void *p,const void *q){
	int64_t *data1,*data2;
	data1 = (int64_t*)p;
	data2 = (int64_t*)q;

	if(data1[0] < data2[0])
		return -1;
	else if(data1[0] == data2[0])
		return 0;
	else
		return 1;
}

#define safe_check_alloc(ptr,N,dtype) {                                          \
                                       (ptr) = (dtype*)malloc((N)*sizeof(dtype));\
                                       if((ptr)==NULL){                          \
                                         success=0;                              \
                                         goto finishlabel;                       \
                                       }                                         \
                                      }

#define safe_check__aligned_alloc(ptr,alignment,N,dtype) {                                        \
                                       (ptr) = (dtype*)aligned_alloc(alignment,(N)*sizeof(dtype));\
                                       if((ptr)==NULL){                                           \
                                         success=0;                                               \
                                         goto finishlabel;                                        \
                                       }                                                          \
                                      }

#define safe_free(ptr) {               \
                        if(ptr != NULL)\
                          free(ptr);   \
                       }
 
int SPHparticle_SoA_malloc(int N,SPHparticle **lsph){
  int success=1;
  //const int alignment = 32;
  (*lsph) = (SPHparticle*)malloc(1*sizeof(SPHparticle));
  if(lsph==NULL){
    success = 0;
    goto finishlabel;
  }
  (*lsph)->x  = NULL; (*lsph)->y  = NULL; (*lsph)->z  = NULL;
  (*lsph)->ux = NULL; (*lsph)->uy = NULL; (*lsph)->uz = NULL;
  (*lsph)->Fx = NULL; (*lsph)->Fy = NULL; (*lsph)->Fz = NULL;
  (*lsph)->nu = NULL; (*lsph)->rho= NULL; 
  (*lsph)->id = NULL; (*lsph)->hash= NULL; 

  safe_check_alloc((*lsph)->x   , N ,double);
  safe_check_alloc((*lsph)->y   , N ,double);
  safe_check_alloc((*lsph)->z   , N ,double);
  safe_check_alloc((*lsph)->ux  , N ,double);
  safe_check_alloc((*lsph)->uy  , N ,double);
  safe_check_alloc((*lsph)->uz  , N ,double);
  safe_check_alloc((*lsph)->Fx  , N ,double);
  safe_check_alloc((*lsph)->Fy  , N ,double);
  safe_check_alloc((*lsph)->Fz  , N ,double);
  safe_check_alloc((*lsph)->nu  , N ,double);
  safe_check_alloc((*lsph)->rho , N ,double);
  safe_check_alloc((*lsph)->id  , N ,int64_t);
  safe_check_alloc((*lsph)->hash,2*N,int64_t);
  

finishlabel:

  if(success)
    return 0;
  else{
    if(*lsph==NULL)
      return 1;

    safe_free((*lsph)->x);  safe_free((*lsph)->y);  safe_free((*lsph)->z);
    safe_free((*lsph)->ux); safe_free((*lsph)->uy); safe_free((*lsph)->uz);
    safe_free((*lsph)->Fx); safe_free((*lsph)->Fy); safe_free((*lsph)->Fz);
    safe_free((*lsph)->nu); safe_free((*lsph)->rho); 
    safe_free((*lsph)->id); safe_free((*lsph)->hash); 

    return 1;
  }
}

int SPHparticleSOA_safe_free(int N,SPHparticle **lsph){
	
  if(*lsph==NULL)
    return 1;

  safe_free((*lsph)->x);  safe_free((*lsph)->y);  safe_free((*lsph)->z);
  safe_free((*lsph)->ux); safe_free((*lsph)->uy); safe_free((*lsph)->uz);
  safe_free((*lsph)->Fx); safe_free((*lsph)->Fy); safe_free((*lsph)->Fz);
  safe_free((*lsph)->nu); safe_free((*lsph)->rho); 
  safe_free((*lsph)->id); safe_free((*lsph)->hash); 

  free((*lsph));

  return 0;
}

int gen_unif_rdn_pos(int64_t N, int seed, SPHparticle *lsph){

	const gsl_rng_type *T=NULL;
	gsl_rng *r=NULL;

	if(lsph==NULL)
		return 1;

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	gsl_rng_set(r,seed);

	for(int64_t i=0;i<N;i+=1){
		lsph->x[i] = gsl_rng_uniform(r);
		lsph->y[i] = gsl_rng_uniform(r);
		lsph->z[i] = gsl_rng_uniform(r);

		lsph->ux[i] = 0.0; lsph->Fx[i] = 0.0;
		lsph->uy[i] = 0.0; lsph->Fy[i] = 0.0;
    lsph->uz[i] = 0.0; lsph->Fz[i] = 0.0;

		lsph->nu[i]   = 1.0/N;
		lsph->rho[i]  = 0.0;
		lsph->id[i]   = (int64_t) i;
		lsph->hash[2*i+0] = (int64_t) 0;
		lsph->hash[2*i+1] = (int64_t) i;
	}

	gsl_rng_free(r);

	return 0;
}

int gen_unif_rdn_pos_box(int64_t N, int seed, linkedListBox *box,SPHparticle *lsph){

  const gsl_rng_type *T=NULL;
  gsl_rng *r=NULL;

  if(lsph==NULL)
    return 1;

  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  gsl_rng_set(r,seed);

  for(int64_t i=0;i<N;i+=1){
    lsph->x[i] = gsl_rng_uniform(r)*(box->Xmax-box->Xmin) + box->Xmin;
    lsph->y[i] = gsl_rng_uniform(r)*(box->Ymax-box->Ymin) + box->Ymin;
    lsph->z[i] = gsl_rng_uniform(r)*(box->Zmax-box->Zmin) + box->Zmin;

    lsph->ux[i] = 0.0; lsph->Fx[i] = 0.0;
    lsph->uy[i] = 0.0; lsph->Fy[i] = 0.0;
    lsph->uz[i] = 0.0; lsph->Fz[i] = 0.0;

    lsph->nu[i]   = 1.0/N;
    lsph->rho[i]  = 0.0;
    lsph->id[i]   = (int64_t) i;
    lsph->hash[2*i+0] = (int64_t) 0;
    lsph->hash[2*i+1] = (int64_t) i;
  }

  gsl_rng_free(r);

  return 0;
}

int compute_hash_MC3D(int64_t N, SPHparticle *lsph, linkedListBox *box){

	if(lsph==NULL)
		return 1;

	if(box==NULL)
		return 1;

	const double etax = box->Nx/(box->Xmax - box->Xmin);
	const double etay = box->Ny/(box->Ymax - box->Ymin);
	const double etaz = box->Nz/(box->Zmax - box->Zmin);

	for(int64_t i=0;i<N;i+=1){
		uint32_t kx,ky,kz;
		kx = (uint32_t)(etax*(lsph->x[i] - box->Xmin));
		ky = (uint32_t)(etay*(lsph->y[i] - box->Ymin));
		kz = (uint32_t)(etaz*(lsph->z[i] - box->Zmin));

		if((kx<0)||(ky<0)||(kz<0))
			return 1;
		else if((kx>=box->Nx)||(ky>=box->Nx)||(kz>=box->Nx))
			return 1;
		else{
			lsph->hash[2*i] = ullMC3Dencode(kx,ky,kz);
		}
	}

	return 0;
}

#define swap_loop(N,lsph,temp_swap,member,type) for(int64_t i=0;i<(N);i+=1)                            \
																	 	              (temp_swap)[i] = (lsph)->member[(lsph)->hash[2*i+1]];\
																		            memcpy((lsph)->member,temp_swap,(N)*sizeof(type))

int reorder_lsph_SoA(int N, SPHparticle *lsph, void *swap_arr){

	int64_t *int64_temp_swap = (int64_t *)swap_arr;
	swap_loop(N,lsph,int64_temp_swap,id ,int64_t);
	double *double_temp_swap = (double *)swap_arr;
	swap_loop(N,lsph,double_temp_swap,nu ,double);
	swap_loop(N,lsph,double_temp_swap,rho,double);
	swap_loop(N,lsph,double_temp_swap,x  ,double);
	swap_loop(N,lsph,double_temp_swap,y  ,double);
	swap_loop(N,lsph,double_temp_swap,z  ,double);
	swap_loop(N,lsph,double_temp_swap,ux ,double);
	swap_loop(N,lsph,double_temp_swap,uy ,double);
	swap_loop(N,lsph,double_temp_swap,uz ,double);
	swap_loop(N,lsph,double_temp_swap,Fx ,double);
	swap_loop(N,lsph,double_temp_swap,Fy ,double);
	swap_loop(N,lsph,double_temp_swap,Fz ,double);

	return 0;
}

int setup_interval_hashtables(int64_t N,SPHparticle *lsph,linkedListBox *box){

	int ret;
	int64_t hash0 = lsph->hash[2*0];
	khiter_t kbegin,kend;

	if(lsph==NULL)
		return 1;

	if(box==NULL)
		return 1;

	kbegin = kh_put(0, box->hbegin, lsph->hash[2*0], &ret); kh_value(box->hbegin, kbegin) = (int64_t)0;
	for(int64_t i=0;i<N;i+=1){
		lsph->hash[i] = lsph->hash[2*i];
		if(lsph->hash[i] == hash0)
			continue;
		hash0 = lsph->hash[i];
		kend   = kh_put(1, box->hend  , lsph->hash[i-1], &ret); kh_value(box->hend  , kend)   = i;
		kbegin = kh_put(0, box->hbegin, lsph->hash[i  ], &ret); kh_value(box->hbegin, kbegin) = i;
	}
	kend   = kh_put(1, box->hend  , lsph->hash[2*(N-1)], &ret); kh_value(box->hend  , kend)   = N;

	return 0;
}

int neighbour_hash_3d(int64_t hash,int64_t *nblist,int width, linkedListBox *box){
	int idx=0,kx=0,ky=0,kz=0;

	kx = ullMC3DdecodeX(hash);
	ky = ullMC3DdecodeY(hash);
	kz = ullMC3DdecodeZ(hash);

	for(int ix=-width;ix<=width;ix+=1){
		for(int iy=-width;iy<=width;iy+=1){
			for(int iz=-width;iz<=width;iz+=1){
				if((kx+ix<0)||(ky+iy<0)||(kz+iz<0))
					nblist[idx++] = -1;
				else if( (kx+ix>=box->Nx)||(ky+iy>=box->Ny)||(kz+iz>=box->Nz) )
					nblist[idx++] = -1;
				else if( kh_get(0, box->hbegin, ullMC3Dencode(kx+ix,ky+iy,kz+iz)) == kh_end(box->hbegin) )
					nblist[idx++] = -1;
				else
					nblist[idx++] = ullMC3Dencode(kx+ix,ky+iy,kz+iz);
			}
		}
	}
	
	return 0;
}

int print_sph_particles_density(int64_t N, double h, long int seed, int runs, SPHparticle *lsph, linkedListBox *box){
	FILE *fp;
	char filename[1024+1];

	sprintf(filename,
					"cd3d(SoA,cll,symmLB,runs=%d)-P(seed=%ld,N=%ld,h=%lg)-B(Nx=%d,Ny=%d,Nz=%d)-D(%lg,%lg,%lg,%lg,%lg,%lg).csv",
					runs,seed,N,h,box->Nx,box->Ny,box->Nz,box->Xmin,box->Ymin,box->Zmin,box->Xmax,box->Ymax,box->Zmax);

	fp = fopen(filename,"w");
	for(int64_t i=0;i<N;i+=1)
		fprintf(fp,"%ld %lf %lf %lf %lf\n",lsph->id[i],lsph->x[i],lsph->y[i],lsph->z[i],lsph->rho[i]);
	fclose(fp);

	return 0;
}

int print_time_stats(int runs, double *times){
  double t[5], dt[5], total_time, dtotal_time;

  printf("fast neighbour search / SoA / outer-openMP / symmetric load balanced\n");

  total_time = 0.;
  for(int k=0;k<5;k+=1){
    t[k]=0.; dt[k]=0.;
    for(int run=0;run<runs;run+=1)
      t[k] += times[5*run+k];
    t[k] /= runs;
    for(int run=0;run<runs;run+=1)
      dt[k] += (times[5*run+k]-t[k])*(times[5*run+k]-t[k]);
    dt[k] /= runs;
    dt[k] = sqrt(dt[k]);

    total_time += t[k];
  }

  dtotal_time = 0.;
  for(int run=0;run<runs;run+=1){
    double rgm = 0.;
    for(int k=0;k<5;k+=1)
      rgm += times[5*run+k];

    dtotal_time += (rgm-total_time)*(rgm-total_time);
  }
  dtotal_time /= runs;
  dtotal_time = sqrt(dtotal_time);

  printf("compute_hash_MC3D calc time                 : %.5lf +- %.6lf s : %.3lg%% +- %.3lg%%\n",t[0],dt[0],100*t[0]/total_time,100*dt[0]/total_time);
  printf("qsort calc time                             : %.5lf +- %.6lf s : %.3lg%% +- %.3lg%%\n",t[1],dt[1],100*t[1]/total_time,100*dt[1]/total_time);
  printf("reorder_lsph_SoA calc time                  : %.5lf +- %.6lf s : %.3lg%% +- %.3lg%%\n",t[2],dt[2],100*t[2]/total_time,100*dt[2]/total_time);
  printf("setup_interval_hashtables calc time         : %.5lf +- %.6lf s : %.3lg%% +- %.3lg%%\n",t[3],dt[3],100*t[3]/total_time,100*dt[3]/total_time);
  printf("compute_density_3d load balanced calc time  : %.5lf +- %.6lf s : %.3lg%% +- %.3lg%%\n",t[4],dt[4],100*t[4]/total_time,100*dt[4]/total_time);
  printf("compute_density_3d load balanced total time : %.5lf +- %.6lf s : %.3lf%%\n",total_time,dtotal_time,100.);

  return 0;
}

