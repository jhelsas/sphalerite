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

#include "klib/khash.h"
#include "mzc/MZC3D64.h"
#include "sph_data_types.h"

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

int neighbour_hash_3d(int64_t hash,int64_t *nblist,int width){
	int idx=0,kx=0,ky=0,kz=0;

	kx = ullMC3DdecodeX(hash);
	ky = ullMC3DdecodeY(hash);
	kz = ullMC3DdecodeZ(hash);

	for(int ix=-width;ix<=width;ix+=1){
		for(int iy=-width;iy<=width;iy+=1){
			for(int iz=-width;iz<=width;iz+=1){
				if((kx+ix<0)||(ky+iy<0)||(kz+iz<0))
					nblist[idx++] = -1;
				else
					nblist[idx++] = ullMC3Dencode(kx+ix,ky+iy,kz+iz);
			}
		}
	}
	return 0;
}

int gen_unif_rdn_pos(){


	return 0;
}

KHASH_MAP_INIT_INT64(0, int64_t)
KHASH_MAP_INIT_INT64(1, int64_t)

int main(){

	int j=0,numThreads=6;
	int N = 100000;
	const gsl_rng_type *T=NULL;
	gsl_rng *r=NULL;
	double S=0.,S2=0.;
	linkedListBox *box;
	SPHparticle *lsph;

	int ret, is_missing;
	khiter_t kbegin,kend;
	khash_t(0) *hbegin = kh_init(0);
	khash_t(1) *hend   = kh_init(1);

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	gsl_rng_set(r,123123123);

	//printf("sizeof keyval : %lu\n",sizeof(keyval));
	printf("sizeof double4: %lu\n",sizeof(double4));
	printf("sizeof SPHparticle : %lu\n",sizeof(SPHparticle));

	lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));

	for(int64_t i=0;i<N;i+=1){
		lsph[i].r.x = gsl_rng_uniform(r); lsph[i].r.y = gsl_rng_uniform(r);
		lsph[i].r.z = gsl_rng_uniform(r); lsph[i].r.t = gsl_rng_uniform(r);

		lsph[i].u.x = 0.0;  lsph[i].u.y = 0.0;
		lsph[i].u.z = 0.0;  lsph[i].u.t = 0.0;

		lsph[i].F.x = 0.0;  lsph[i].F.y = 0.0;
		lsph[i].F.z = 0.0;  lsph[i].F.t = 0.0;

		lsph[i].nu = 1.0/N; lsph[i].rho = 0.0;
		lsph[i].id = i;     lsph[i].hash = 0;
	}

	box = (linkedListBox*)malloc(1*sizeof(linkedListBox));

	box->Nx = box->Ny = box->Nz = 100; 
	box->N  = (box->Nx)*(box->Ny)*(box->Nz);
	box->Xmin.x = 0.0; box->Xmin.y = 0.0; box->Xmin.z = 0.0;
	box->Xmax.x = 1.0; box->Xmax.y = 1.0; box->Xmax.z = 1.0;

	for(int64_t i=0;i<N;i+=1){
		uint32_t kx,ky,kz;
		kx = (int)((lsph[i].r.x - box->Xmin.x)*box->Nx/(box->Xmax.x - box->Xmin.x));
		ky = (int)((lsph[i].r.y - box->Xmin.y)*box->Ny/(box->Xmax.y - box->Xmin.y));
		kz = (int)((lsph[i].r.z - box->Xmin.z)*box->Nz/(box->Xmax.z - box->Xmin.z));

		lsph[i].hash = ullMC3Dencode(kx,ky,kz);
		//printf("%lu : %lu \n",lsph[i].id,lsph[i].hash);
	}

	qsort(lsph,N,sizeof(SPHparticle),compare_SPHparticle);

	kbegin = kh_put(0, hbegin, lsph[0].hash, &ret); kh_value(hbegin, kbegin) = (int64_t)0;
	int idx_key=0;
	int64_t hash0 = lsph[0].hash;
	for(int i=0;i<N;i+=1){
		if(lsph[i].hash == hash0)
			continue;
		hash0 = lsph[i].hash;
		kend   = kh_put(1, hend  , lsph[i-1].hash, &ret); kh_value(hend  , kend)   = i;
		kbegin = kh_put(0, hbegin, lsph[i  ].hash, &ret); kh_value(hbegin, kbegin) = i;
	}
	kend   = kh_put(1, hend  , lsph[N-1].hash, &ret); kh_value(hend  , kend)   = N;

	
	for (kbegin = kh_begin(hbegin); kbegin != kh_end(hbegin); kbegin++){
		if (kh_exist(hbegin, kbegin)){
			kend = kh_get(1, hend, kh_key(hbegin, kbegin));
			//printf("hash - begin/end = %lu - (%lu,%lu): %lld\n",
			//	kh_key(hbegin, kbegin),
			//	kh_value(hbegin, kbegin),
			//	kh_value(hend, kend),
			//	((long long int)kh_value(hend, kend))-((long long int)kh_value(hbegin, kbegin)));
		}
	}

	
	int64_t nblist[3*3*3]={0};
	for(int64_t i=0;i<3;i+=1){
		int res = neighbour_hash_3d(lsph[i].hash,nblist,1);

		printf("origin hash %lu : (%d,%d,%d) \n",lsph[i].hash,ullMC3DdecodeX(lsph[i].hash),ullMC3DdecodeY(lsph[i].hash),ullMC3DdecodeZ(lsph[i].hash));
		for(int64_t j=0;j<3*3*3;j+=1){
			if(nblist[j]>=0)
				printf("    neighbour hash %lu : (%d,%d,%d) \n",nblist[j],ullMC3DdecodeX(nblist[j]),ullMC3DdecodeY(nblist[j]),ullMC3DdecodeZ(nblist[j]));
			else
				printf("no neighbour here\n");
		}
		printf("\n\n");
	}

	printf("sizeof int64_t : %lu\n",sizeof(int64_t));
  
	int64_t *hash_arr;

	hash_arr = (int64_t*)malloc(N*sizeof(int64_t));

	for(int k=0;k<N;k+=1)
		hash_arr[k] = gsl_rng_get(r) % N;

	gsl_rng_free(r);
	free(box);
	free(lsph);

	kh_destroy(0, hbegin);
	kh_destroy(1, hend);

	return 0;
}

/********************************************************************************/

// https://stackoverflow.com/questions/16007640/openmp-parallel-quicksort

unsigned int rand_interval(unsigned int min, unsigned int max)
{
    // https://stackoverflow.com/questions/2509679/
    int r;
    const unsigned int range = 1 + max - min;
    const unsigned int buckets = RAND_MAX / range;
    const unsigned int limit = buckets * range;

    do
    {
        r = rand();
    } 
    while (r >= limit);

    return min + (r / buckets);
}

void fillupRandomly (int *m, int size, unsigned int min, unsigned int max){
    for (int i = 0; i < size; i++)
    m[i] = rand_interval(min, max);
} 


void init(int *a, int size){
   for(int i = 0; i < size; i++)
       a[i] = 0;
}

void printArray(int *a, int size){
   for(int i = 0; i < size; i++)
       printf("%d ", a[i]);
   printf("\n");
}

int isSorted(int *a, int size){
   for(int i = 0; i < size - 1; i++)
      if(a[i] > a[i + 1])
        return 0;
   return 1;
}


int partition(int * a, int p, int r)
{
    int lt[r-p];
    int gt[r-p];
    int i;
    int j;
    int key = a[r];
    int lt_n = 0;
    int gt_n = 0;

    for(i = p; i < r; i++){
        if(a[i] < a[r]){
            lt[lt_n++] = a[i];
        }else{
            gt[gt_n++] = a[i];
        }   
    }   

    for(i = 0; i < lt_n; i++){
        a[p + i] = lt[i];
    }   

    a[p + lt_n] = key;

    for(j = 0; j < gt_n; j++){
        a[p + lt_n + j + 1] = gt[j];
    }   

    return p + lt_n;
}

void quicksort(int * a, int p, int r)
{
    int div;

    if(p < r){ 
        div = partition(a, p, r); 
        #pragma omp task shared(a) if(r - p > TASK_SIZE) 
        quicksort(a, p, div - 1); 
        #pragma omp task shared(a) if(r - p > TASK_SIZE)
        quicksort(a, div + 1, r); 
    }
}