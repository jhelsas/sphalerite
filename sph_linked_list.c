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

KHASH_MAP_INIT_INT64(0, size_t)
KHASH_MAP_INIT_INT64(1, size_t)
int main(){
	int ret, is_missing;
	khiter_t k,kbegin,kend;
	khash_t(0) *hbegin = kh_init(0);
	khash_t(1) *hend   = kh_init(1);
	size_t key=5,val0=0,val1=0;

	key = 1; val0 = 123; val1=1000000+123;
	k = kh_put(0, hbegin, key, &ret); kh_value(hbegin, k) = val0;
	k = kh_put(1, hend  , key, &ret); kh_value(hend  , k) = val1;
	printf("ret %d\n",ret);

	key = 3; val0 = 456; val1=1000000+456;
	k = kh_put(0, hbegin, key, &ret); kh_value(hbegin, k) = val0;
	k = kh_put(1, hend  , key, &ret); kh_value(hend  , k) = val1;
	printf("ret %d\n",ret);

	key = 5; val0 = 789; val1=1000000+789;
	k = kh_put(0, hbegin, key, &ret); kh_value(hbegin, k) = val0;
	k = kh_put(1, hend  , key, &ret); kh_value(hend  , k) = val1;
	printf("ret %d\n",ret);

	///////////////////////////////////////

	key = 5;
	k = kh_get(0, hbegin, key);
	is_missing = (k == kh_end(hbegin));
	printf("k=%d is_missing=%d kh_end=%d\n",k,is_missing,kh_end(hbegin));

	key = 1;
	k = kh_get(0, hbegin, key);
	is_missing = (k == kh_end(hbegin));
	printf("k=%d is_missing=%d kh_end=%d\n",k,is_missing,kh_end(hbegin));

	key = 2;
	k = kh_get(0, hbegin, key);
	is_missing = (k == kh_end(hbegin));
	printf("k=%d is_missing=%d kh_end=%d\n",k,is_missing,kh_end(hbegin));

	key = 3;
	k = kh_get(0, hbegin, key);
	is_missing = (k == kh_end(hbegin));
	printf("k=%d is_missing=%d kh_end=%d\n",k,is_missing,kh_end(hbegin));

	key = 4;
	k = kh_get(0, hbegin, key);
	is_missing = (k == kh_end(hbegin));
	printf("k=%d is_missing=%d kh_end=%d\n",k,is_missing,kh_end(hbegin));

	key = 3;
	k = kh_get(0, hbegin, key);
	kh_del(0, hbegin, k);
	/*
	for (kbegin = kh_begin(hbegin); kbegin != kh_end(hbegin); ++kbegin){
		printf("hkey = %lu\n",kh_key(hbegin, k));
		if (kh_exist(hbegin, kbegin)){
			kend = kh_get(1, hend, kh_key(hbegin, k));
			printf("-key=%lu exists! val = %lu\n",kh_key(hbegin, k),kh_value(hbegin, k));
			printf("----key=%lu exists! val = %lu\n",kh_key(hend, kend),kh_value(hend, kend));
		}
	}*/

	for (kbegin = kh_begin(hbegin); kbegin != kh_end(hbegin); ++kbegin){
		printf("hkey = %lu\n",kh_key(hbegin, k));
		if (kh_exist(hbegin, kbegin)){
			//kend = kh_get(1, hend, kh_key(hbegin, k));
			printf("-key=%lu exists! val = %lu\n",kh_key(hbegin, k),kh_value(hbegin, k));
			//printf("----key=%lu exists! val = %lu\n",kh_key(hend, kend),kh_value(hend, kend));
		}
	}

	kh_destroy(0, hbegin);
	kh_destroy(1, hend);
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

int neighbour_hash_3d(uint64_t hash,uint64_t *nblist,int width){
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

/*
int main(){

	int j=0,numThreads=6;
	int N = 1000000;
	size_t *dshift;
	const gsl_rng_type *T=NULL;
	gsl_rng *r=NULL;
	double S=0.,S2=0.;
	linkedListBox *box;
	SPHparticle *lsph;

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	gsl_rng_set(r,123123123);

	printf("sizeof keyval : %lu\n",sizeof(keyval));
	printf("sizeof double4: %lu\n",sizeof(double4));
	printf("sizeof SPHparticle : %lu\n",sizeof(SPHparticle));

	lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));

	for(size_t i=0;i<N;i+=1){
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

	box->Nx = box->Ny = box->Nz = 30; 
	box->N  = (box->Nx)*(box->Ny)*(box->Nz);
	box->Xmin.x = 0.0; box->Xmin.y = 0.0; box->Xmin.z = 0.0;
	box->Xmax.x = 1.0; box->Xmax.y = 1.0; box->Xmax.z = 1.0;

	for(size_t i=0;i<N;i+=1){
		uint32_t kx,ky,kz;
		kx = (int)((lsph[i].r.x - box->Xmin.x)*box->Nx/(box->Xmax.x - box->Xmin.x));
		ky = (int)((lsph[i].r.y - box->Xmin.y)*box->Ny/(box->Xmax.y - box->Xmin.y));
		kz = (int)((lsph[i].r.z - box->Xmin.z)*box->Nz/(box->Xmax.z - box->Xmin.z));

		lsph[i].hash = ullMC3Dencode(kx,ky,kz);
		//printf("%lu : %lu \n",lsph[i].id,lsph[i].hash);
	}

	qsort(lsph,N,sizeof(SPHparticle),compare_SPHparticle);

	dshift = (size_t*)malloc(N*sizeof(size_t));
	dshift[0] = 0;
	for(size_t i=1;i<N;i+=1)
		dshift[i] = lsph[i].hash - lsph[i-1].hash;

	uint64_t nblist[5*5*5]={0};
	for(size_t i=0;i<N;i+=35000){
		int res = neighbour_hash_3d(lsph[i].hash,nblist,1);

		printf("origin hash %lu : (%d,%d,%d) \n",lsph[i].hash,ullMC3DdecodeX(lsph[i].hash),ullMC3DdecodeY(lsph[i].hash),ullMC3DdecodeZ(lsph[i].hash));
		for(size_t j=0;j<5*5*5;j+=1){
			if(nblist[j]>=0)
				printf("    neighbour hash %lu : (%d,%d,%d) \n",nblist[j],ullMC3DdecodeX(nblist[j]),ullMC3DdecodeY(nblist[j]),ullMC3DdecodeZ(nblist[j]));
		}
		printf("\n\n");
	}

	printf("sizeof size_t : %lu\n",sizeof(size_t));
  
	uint64_t *hash_arr;

	hash_arr = (uint64_t*)malloc(N*sizeof(uint64_t));

	for(int k=0;k<N;k+=1)
		hash_arr[k] = gsl_rng_get(r) % N;

	gsl_rng_free(r);
	free(box);
	free(lsph);

	return 0;
}*/

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