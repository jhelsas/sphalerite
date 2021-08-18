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

int compute_hash_MC3D(int64_t N, SPHparticle *lsph, linkedListBox *box){

	if(lsph==NULL)
		return 1;

	if(box==NULL)
		return 1;

	for(int64_t i=0;i<N;i+=1){
		uint32_t kx,ky,kz;
		kx = (uint32_t)((lsph[i].r.x - box->Xmin.x)*box->Nx/(box->Xmax.x - box->Xmin.x));
		ky = (uint32_t)((lsph[i].r.y - box->Xmin.y)*box->Ny/(box->Xmax.y - box->Xmin.y));
		kz = (uint32_t)((lsph[i].r.z - box->Xmin.z)*box->Nz/(box->Xmax.z - box->Xmin.z));

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

int compute_hash_MC2D(int64_t N, SPHparticle *lsph, linkedListBox *box){

	if(lsph==NULL)
		return 1;

	if(box==NULL)
		return 1;

	for(int64_t i=0;i<N;i+=1){
		uint32_t kx,ky,kz;
		kx = (int)((lsph[i].r.x - box->Xmin.x)*box->Nx/(box->Xmax.x - box->Xmin.x));
		ky = (int)((lsph[i].r.y - box->Xmin.y)*box->Ny/(box->Xmax.y - box->Xmin.y));
		kz = -1;

		if((kx<0)||(ky<0))
			return 1;
		else if((kx>=box->Nx)||(ky>=box->Nx))
			return 1;
		else
			lsph[i].hash = ullMC2Dencode(kx,ky);
	}

	return 0;
}

int setup_interval_hashtables(int64_t N,SPHparticle *lsph,linkedListBox *box){

	int ret, is_missing;
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

int neighbour_hash_2d(int64_t hash,int64_t *nblist,int width, linkedListBox *box){
	int idx=0,kx=0,ky=0;

	kx = ullMC2DdecodeX(hash);
	ky = ullMC2DdecodeY(hash);

	for(int ix=-width;ix<=width;ix+=1){
		for(int iy=-width;iy<=width;iy+=1){
				if((kx+ix<0)||(ky+iy<0))
					nblist[idx++] = -1;
				else if((kx+ix>=box->Nx)||(ky+iy>=box->Ny))
					nblist[idx++] = -1;
				else if( kh_get(0, box->hbegin, ullMC2Dencode(kx+ix,ky+iy)) == kh_end(box->hbegin) )
					nblist[idx++] = -1;
				else
					nblist[idx++] = ullMC2Dencode(kx+ix,ky+iy);
		}
	}
	return 0;
}

int print_boxes_populations(linkedListBox *box){
	khiter_t kbegin,kend;

	for (kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
		if (kh_exist(box->hbegin, kbegin)){
			kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));
			printf("hash - (begin,end) = %lu - (%lu,%lu): %lld\n",
				kh_key(box->hbegin, kbegin),
				kh_value(box->hbegin, kbegin),
				kh_value(box->hend, kend),
				((long long int)kh_value(box->hend, kend))-((long long int)kh_value(box->hbegin, kbegin)));
		}
	}

	return 0;
}

int print_neighbour_list_MC3D(int64_t Nmax,unsigned int width,unsigned int stride,SPHparticle *lsph,linkedListBox *box){

	int64_t nblist[(2*width+1)*(2*width+1)*(2*width+1)];
	
	for(int64_t i=0;i<Nmax;i+=(int64_t)stride){
		int res = neighbour_hash_3d(lsph[i].hash,nblist,width,box);

		printf("origin hash %lu : (%d,%d,%d) \n",lsph[i].hash,ullMC3DdecodeX(lsph[i].hash),
															  ullMC3DdecodeY(lsph[i].hash),
															  ullMC3DdecodeZ(lsph[i].hash));
		for(unsigned int j=0;j<(2*width+1)*(2*width+1)*(2*width+1);j+=1){
			if(nblist[j]>=0)
				printf("    neighbour hash %lu : (%d,%d,%d) \n",nblist[j],ullMC3DdecodeX(nblist[j]),
																		  ullMC3DdecodeY(nblist[j]),
																		  ullMC3DdecodeZ(nblist[j]));
			else
				printf("no neighbour here\n");
		}
		printf("\n\n");
	}

	return 0;
}

int print_neighbour_list_MC2D(int64_t Nmax,unsigned int width,unsigned int stride,SPHparticle *lsph,linkedListBox *box){

	int64_t nblist[(2*width+1)*(2*width+1)];
	
	for(int64_t i=0;i<Nmax;i+=(int64_t)stride){
		int res = neighbour_hash_2d(lsph[i].hash,nblist,width,box);

		printf("origin hash %lu : (%d,%d) \n",lsph[i].hash,ullMC2DdecodeX(lsph[i].hash),
															  											 ullMC2DdecodeY(lsph[i].hash));
		for(unsigned int j=0;j<(2*width+1)*(2*width+1);j+=1){
			if(nblist[j]>=0)
				printf("    neighbour hash %lu : (%d,%d) \n",nblist[j],ullMC3DdecodeX(nblist[j]),
																		  												 ullMC3DdecodeY(nblist[j]));
			else
				printf("no neighbour here\n");
		}
		printf("\n\n");
	}

	return 0;
}

int print_neighbour_list_MC3D_lsph_file(int64_t Nmax,unsigned int width,unsigned int stride,SPHparticle *lsph,linkedListBox *box){
	FILE *fp;
	int64_t nblist[(2*width+1)*(2*width+1)*(2*width+1)];

	fp = fopen("data/nblist_MC3D.csv","w");

	for(int64_t i=0;i<Nmax;i+=(int64_t)stride){
		uint32_t kx,ky,kz;
		int res = neighbour_hash_3d(lsph[i].hash,nblist,width,box);
		
		kx = (int)((lsph[i].r.x - box->Xmin.x)*box->Nx/(box->Xmax.x - box->Xmin.x));
		ky = (int)((lsph[i].r.y - box->Xmin.y)*box->Ny/(box->Xmax.y - box->Xmin.y));
		kz = (int)((lsph[i].r.z - box->Xmin.z)*box->Nz/(box->Xmax.z - box->Xmin.z));
		
		fprintf(fp,"base   hash %lu : (%d,%d,%d) \n",lsph[i].hash,kx,ky,kz);
		fprintf(fp,"origin hash %lu : (%d,%d,%d) \n",lsph[i].hash,ullMC3DdecodeX(lsph[i].hash),
					         									  												ullMC3DdecodeY(lsph[i].hash),
							    								  													ullMC3DdecodeZ(lsph[i].hash));

		for(unsigned int j=0;j<(2*width+1)*(2*width+1)*(2*width+1);j+=1)
			if(nblist[j]>=0){
				fprintf(fp,"  %lu:(%d,%d,%d)",nblist[j],ullMC3DdecodeX(nblist[j]),
																								ullMC3DdecodeY(nblist[j]),
																								ullMC3DdecodeZ(nblist[j]));
				if(j<(2*width+1)*(2*width+1)*(2*width+1)-1)
					fprintf(fp,",");
			}
		
		fprintf(fp,"\n\n");
	}

	fclose(fp);

	return 0;
}

int print_neighbour_list_MC3D_lsph_ids_file(int N, SPHparticle *lsph, linkedListBox *box){
	int err, res;
  double dist = 0.0;
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0, node_end=0;
  int64_t nb_hash=-1  , nb_begin= 0, nb_end = 0;
  int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

  FILE *fp = fopen("data/nblist_ll.json","w");
  fprintf(fp,"{\n");
  for (kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){
    
    if (kh_exist(box->hbegin, kbegin)){

      kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));
      node_hash = kh_key(box->hbegin, kbegin);
      node_begin = kh_value(box->hbegin, kbegin);
      node_end   = kh_value(box->hend, kend);

      res = neighbour_hash_3d(node_hash,nblist,box->width,box);
      for(int64_t ii=node_begin;ii<node_end;ii+=1){

      	fprintf(fp,"\"%ld\":[",lsph[ii].id);
      	for(int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
      		if(nblist[j]>=0){
      			nb_hash  = nblist[j];
          	nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          	nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          	for(int64_t jj=nb_begin;jj<nb_end;jj+=1)
          		fprintf(fp,"%ld, ",lsph[jj].id);
      		}
      	}
      	if((kbegin==kh_end(box->hbegin)-1)&&(ii==node_end-1))
      		fprintf(fp,"-1]\n");
    		else
    			fprintf(fp,"-1],\n");
      }
    }
  }
  fprintf(fp,"}");

  fflush(fp);
  fclose(fp);

	return 0;
}

int print_neighbour_list_MC2D_lsph_file(int64_t Nmax,unsigned int width,unsigned int stride,SPHparticle *lsph,linkedListBox *box){
	FILE *fp;
	int64_t nblist[(2*width+1)*(2*width+1)];

	fp = fopen("data/nblist_MC2D.csv","w");

	for(int64_t i=0;i<Nmax;i+=(int64_t)stride){
		uint32_t kx,ky;
		int res = neighbour_hash_2d(lsph[i].hash,nblist,width,box);
		
		kx = (int)((lsph[i].r.x - box->Xmin.x)*box->Nx/(box->Xmax.x - box->Xmin.x));
		ky = (int)((lsph[i].r.y - box->Xmin.y)*box->Ny/(box->Xmax.y - box->Xmin.y));
		
		fprintf(fp,"base   hash %lu : (%d,%d) \n",lsph[i].hash,kx,ky);
		fprintf(fp,"origin hash %lu : (%d,%d) \n",lsph[i].hash, ullMC2DdecodeX(lsph[i].hash),
					         									  											ullMC2DdecodeY(lsph[i].hash));		
		
		for(unsigned int j=0;j<(2*width+1)*(2*width+1);j+=1)
			if(nblist[j]>=0){
				fprintf(fp,"  %lu:(%d,%d)",nblist[j], ullMC2DdecodeX(nblist[j]),
																    					ullMC2DdecodeY(nblist[j]));
				if(j<(2*width+1)*(2*width+1)-1)
					fprintf(fp,",");
			}
		
		fprintf(fp,"\n\n");
	}

	fclose(fp);

	return 0;
}

/*
int main(){

  int j=0,numThreads=6,err;
  int64_t N = 100000;
  double S=0.,S2=0.;
  linkedListBox *box;
  SPHparticle *lsph;

  lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));

  err = gen_unif_rdn_pos( N,123123123,lsph);

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox));

  box->Nx = box->Ny = box->Nz = 10;
  box->N  = (box->Nx)*(box->Ny)*(box->Nz);
  box->Xmin.x = 0.0; box->Xmin.y = 0.0; box->Xmin.z = 0.0;
  box->Xmax.x = 1.0; box->Xmax.y = 1.0; box->Xmax.z = 1.0;
  box->hbegin = kh_init(0);
  box->hend = kh_init(1);

  err = compute_hash_MC3D(N,lsph,box);

  qsort(lsph,N,sizeof(SPHparticle),compare_SPHparticle);

  err = setup_interval_hashtables(N,lsph,box);

  //print_boxes_populations(box);
  //print_neighbour_list_MC3D(N,1,N/13,lsph,box);
  print_neighbour_list_MC3D_lsph_file(N,1,1,lsph,box);

  free(lsph);
  safe_free_box(box);

  return 0;
}*/