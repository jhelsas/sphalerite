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

int compare_SPHparticle(const void *p,const void *q){
	SPHparticle *data1,*data2;
	data1 = (SPHparticle*)p;
	data2 = (SPHparticle*)q;
	if(data1->hash < data2->hash)
		return -1;
	else if(data1->hash == data2->hash)
		return 0;
	else
		return 1;
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
		lsph[i].r.x = gsl_rng_uniform(r); lsph[i].r.y = gsl_rng_uniform(r);
		lsph[i].r.z = gsl_rng_uniform(r); lsph[i].r.t = gsl_rng_uniform(r);

		lsph[i].u.x = 0.0;  lsph[i].u.y = 0.0;
		lsph[i].u.z = 0.0;  lsph[i].u.t = 0.0;

		lsph[i].F.x = 0.0;  lsph[i].F.y = 0.0;
		lsph[i].F.z = 0.0;  lsph[i].F.t = 0.0;

		lsph[i].nu = 1.0/N; lsph[i].rho  = 0.0;
		lsph[i].id = i;     lsph[i].hash = 0;
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
    lsph[i].r.x = gsl_rng_uniform(r)*(box->Xmax-box->Xmin) + box->Xmin;
    lsph[i].r.y = gsl_rng_uniform(r)*(box->Ymax-box->Ymin) + box->Ymin;
    lsph[i].r.z = gsl_rng_uniform(r)*(box->Zmax-box->Zmin) + box->Zmin;

		lsph[i].u.x = 0.0;  lsph[i].u.y = 0.0;
		lsph[i].u.z = 0.0;  lsph[i].u.t = 0.0;

		lsph[i].F.x = 0.0;  lsph[i].F.y = 0.0;
		lsph[i].F.z = 0.0;  lsph[i].F.t = 0.0;

		lsph[i].nu = 1.0/N; lsph[i].rho  = 0.0;
		lsph[i].id = i;     lsph[i].hash = 0;
  }

  gsl_rng_free(r);

  return 0;
}

int compute_hash_MC3D(int64_t N, SPHparticle *lsph, linkedListBox *box){

	if(lsph==NULL)
		return 1;

	if(box==NULL)
		return 1;

	for(int64_t i=0;i<N;i+=1){
		uint32_t kx,ky,kz;
		kx = (uint32_t)((lsph[i].r.x - box->Xmin)*box->Nx/(box->Xmax - box->Xmin));
		ky = (uint32_t)((lsph[i].r.y - box->Ymin)*box->Ny/(box->Ymax - box->Ymin));
		kz = (uint32_t)((lsph[i].r.z - box->Zmin)*box->Nz/(box->Zmax - box->Zmin));

		//printf("%")

		if((kx<0)||(ky<0)||(kz<0))
			return 1;
		else if((kx>=box->Nx)||(ky>=box->Nx)||(kz>=box->Nx))
			return 1;
		else
			lsph[i].hash = ullMC3Dencode(kx,ky,kz);
	}

	return 0;
}

int setup_interval_hashtables(int64_t N,SPHparticle *lsph,linkedListBox *box){

	int ret;
	int64_t hash0 = lsph[0].hash;
	khiter_t kbegin,kend;

	if(lsph==NULL)
		return 1;

	if(box==NULL)
		return 1;

	kbegin = kh_put(0, box->hbegin, lsph[0].hash, &ret); kh_value(box->hbegin, kbegin) = (int64_t)0;
	for(int i=0;i<N;i+=1){
		if(lsph[i].hash == hash0)
			continue;
		hash0 = lsph[i].hash;
		kend   = kh_put(1, box->hend  , lsph[i-1].hash, &ret); kh_value(box->hend  , kend)   = i;
		kbegin = kh_put(0, box->hbegin, lsph[i  ].hash, &ret); kh_value(box->hbegin, kbegin) = i;
	}
	kend   = kh_put(1, box->hend  , lsph[N-1].hash, &ret); kh_value(box->hend  , kend)   = N;

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

int print_sph_particles_density(const char *prefix, bool is_cll, int64_t N, double h, 
																long int seed, int runs, SPHparticle *lsph, linkedListBox *box){
	FILE *fp;
	char filename[1024+1];

	if(is_cll){
		sprintf(filename,
						"data/cd3d(cll,%s,runs=%d)-P(seed=%ld,N=%ld,h=%lg)-B(Nx=%d,Ny=%d,Nz=%d)-D(%lg,%lg,%lg,%lg,%lg,%lg).csv",
						prefix,runs,seed,N,h,box->Nx,box->Ny,box->Nz,box->Xmin,box->Ymin,box->Zmin,box->Xmax,box->Ymax,box->Zmax);

		fp = fopen(filename,"w");
		fprintf(fp,"id,x,y,z,rho\n");
		for(int64_t i=0;i<N;i+=1)
			fprintf(fp,"%ld,%lf,%lf,%lf,%lf\n",lsph[i].id,lsph[i].r.x,lsph[i].r.y,lsph[i].r.z,lsph[i].rho);
		fclose(fp);
	} 
	else{
		sprintf(filename,
						"data/cd3d(naive,%s,runs=%d)-P(seed=%ld,N=%ld,h=%lg)-B(Nx=%d,Ny=%d,Nz=%d)-D(%lg,%lg,%lg,%lg,%lg,%lg).csv",
						prefix,runs,seed,N,h,box->Nx,box->Ny,box->Nz,box->Xmin,box->Ymin,box->Zmin,box->Xmax,box->Ymax,box->Zmax);

		fp = fopen(filename,"w");
		fprintf(fp,"id,x,y,z,rho\n");
		for(int64_t i=0;i<N;i+=1)
			fprintf(fp,"%ld,%lf,%lf,%lf,%lf\n",lsph[i].id,lsph[i].r.x,lsph[i].r.y,lsph[i].r.z,lsph[i].rho);
		fclose(fp);
	}
	

	return 0;
}

int print_time_stats(const char *prefix, bool is_cll, int64_t N, double h, 
										 long int seed, int runs, SPHparticle *lsph, linkedListBox *box,double *times){
  FILE *fp;
  double t[5], dt[5], total_time, dtotal_time;
	char filename[1024+1];

	if(is_cll){
  	sprintf(filename,
						"data/times-(cll,%s,runs=%d)-P(seed=%ld,N=%ld,h=%lg)-B(Nx=%d,Ny=%d,Nz=%d)-D(%lg,%lg,%lg,%lg,%lg,%lg).csv",
						prefix,runs,seed,N,h,box->Nx,box->Ny,box->Nz,box->Xmin,box->Ymin,box->Zmin,box->Xmax,box->Ymax,box->Zmax);

  	fp = fopen(filename,"w");
		fprintf(fp,"id, compute_hash_MC3D, sorting, reorder_lsph_SoA, setup_interval_hashtables, compute_density\n");
		for(int run=0;run<runs;run+=1)
			fprintf(fp,"%d %lf %lf %lf %lf %lf\n",run,times[5*run+0],times[5*run+1],times[5*run+2],times[5*run+3],times[5*run+4]);
		fclose(fp);

  	total_time = 0.;
  	for(int k=0;k<4;k+=1){
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
  	  for(int k=0;k<4;k+=1)
    	  rgm += times[5*run+k];

    	dtotal_time += (rgm-total_time)*(rgm-total_time);
  	}
  	dtotal_time /= runs;
  	dtotal_time = sqrt(dtotal_time);

  	printf("compute_hash_MC3D          : %.5lf +- %.6lf s : %.3lg%% +- %.3lg%%\n",t[0],dt[0],100*t[0]/total_time,100*dt[0]/total_time);
    printf("qsort calculation time     : %.5lf +- %.6lf s : %.3lg%% +- %.3lg%%\n",t[1],dt[1],100*t[1]/total_time,100*dt[1]/total_time);
    printf("setup_interval_hashtables  : %.5lf +- %.6lf s : %.3lg%% +- %.3lg%%\n",t[2],dt[2],100*t[2]/total_time,100*dt[2]/total_time);
    printf("compute_density_3d         : %.5lf +- %.6lf s : %.3lg%% +- %.3lg%%\n",t[3],dt[3],100*t[3]/total_time,100*dt[3]/total_time);
    printf("compute_density_3d total   : %.5lf +- %.6lf s : %.3lg%%\n",total_time,dtotal_time,100.);
	}
	else{
		sprintf(filename,
						"data/times-(naive,%s,runs=%d)-P(seed=%ld,N=%ld,h=%lg)-B(Nx=%d,Ny=%d,Nz=%d)-D(%lg,%lg,%lg,%lg,%lg,%lg).csv",
						prefix,runs,seed,N,h,box->Nx,box->Ny,box->Nz,box->Xmin,box->Ymin,box->Zmin,box->Xmax,box->Ymax,box->Zmax);

  	fp = fopen(filename,"w");
		fprintf(fp,"id, compute_density\n");
		for(int run=0;run<runs;run+=1)
			fprintf(fp,"%d %lf\n",run,times[5*run+0]);
		fclose(fp);

  	total_time = 0.;
  	for(int k=0;k<1;k+=1){
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
  	  for(int k=0;k<1;k+=1)
    	  rgm += times[5*run+k];

    	dtotal_time += (rgm-total_time)*(rgm-total_time);
  	}
  	dtotal_time /= runs;
  	dtotal_time = sqrt(dtotal_time);

  	printf("compute_density_3d naive %s : %.5lf +- %.6lf s : %.3lf%%\n",prefix,total_time,dtotal_time,100.);
	}


  return 0;
}
