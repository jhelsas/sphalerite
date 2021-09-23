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

  
  const int alignment = 32;
  safe_check__aligned_alloc((*lsph)->x   , alignment, N ,double);
  safe_check__aligned_alloc((*lsph)->y   , alignment, N ,double);
  safe_check__aligned_alloc((*lsph)->z   , alignment, N ,double);
  safe_check__aligned_alloc((*lsph)->ux  , alignment, N ,double);
  safe_check__aligned_alloc((*lsph)->uy  , alignment, N ,double);
  safe_check__aligned_alloc((*lsph)->uz  , alignment, N ,double);
  safe_check__aligned_alloc((*lsph)->Fx  , alignment, N ,double);
  safe_check__aligned_alloc((*lsph)->Fy  , alignment, N ,double);
  safe_check__aligned_alloc((*lsph)->Fz  , alignment, N ,double);
  safe_check__aligned_alloc((*lsph)->nu  , alignment, N ,double);
  safe_check__aligned_alloc((*lsph)->rho , alignment, N ,double);
  safe_check__aligned_alloc((*lsph)->id  , alignment, N ,int64_t);
  safe_check__aligned_alloc((*lsph)->hash, alignment,2*N,int64_t);
  
  /*
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
  safe_check_alloc((*lsph)->hash,2*N,int64_t);*/
  

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

int arg_parse(int argc, char **argv, int64_t *N, double *h,
              linkedListBox *box){
  bool intern_h = true;

  box->Xmin = -1.0; box->Ymin = -1.0; box->Zmin = -1.0;
  box->Xmax =  2.0; box->Ymax =  2.0; box->Zmax =  2.0;
  
  if(argc%2==0){
    printf("wrong number of arguments!\n");
    printf("Maybe an option is missing a value?\n");
  }

  for(int i=1;i<argc;i+=2){
    if( strcmp(argv[i],"-N") == 0 ){
      *N = (int64_t) atol(argv[i+1]);
      printf("N particles = %ld\n",*N);
    }
    else if( strcmp(argv[i],"-h") == 0 ){
      *h = atof(argv[i+1]);
      printf("h = %lf\n",*h);
      intern_h = false;
    }
    else if( strcmp(argv[i],"-Xmin") == 0 ){
      box->Xmin = atof(argv[i+1]);
      printf("Xmin = %lf\n",box->Xmin);
    }
    else if( strcmp(argv[i],"-Ymin") == 0 ){
      box->Ymin = atof(argv[i+1]);
      printf("Ymin = %lf\n",box->Ymin);
    }
    else if( strcmp(argv[i],"-Zmin") == 0 ){
      box->Zmin = atof(argv[i+1]);
      printf("Zmin = %lf\n",box->Zmin);
    }
    else if( strcmp(argv[i],"-Xmax") == 0 ){
      box->Xmax = atof(argv[i+1]);
      printf("Xmax = %lf\n",box->Xmax);
    }
    else if( strcmp(argv[i],"-Ymax") == 0 ){
      box->Ymax = atof(argv[i+1]);
      printf("Ymax = %lf\n",box->Ymax);
    }
    else if( strcmp(argv[i],"-Zmax") == 0 ){
      box->Zmax = atof(argv[i+1]);
      printf("Zmax = %lf\n",box->Zmax);
    }
    else if( strcmp(argv[i],"-Nx") == 0 ){
      box->Nx   = atol(argv[i+1]);
      printf("Nx = %ld\n",box->Nx);
    }
    else if( strcmp(argv[i],"-Ny") == 0 ){
      box->Ny   = atol(argv[i+1]);
      printf("Ny = %ld\n",box->Ny);
    }
    else if( strcmp(argv[i],"-Nz") == 0 ){
      box->Nz   = atol(argv[i+1]);
      printf("Nz = %ld\n",box->Nz);
    }
    else{
      printf("unknown option: %s %s\n",argv[i],argv[i+1]);
    }
  }

  if(intern_h){
    box->Nx = (int)( (box->Xmax-box->Xmin)/(2*(*h)) );
    box->Ny = (int)( (box->Ymax-box->Ymin)/(2*(*h)) );
    box->Nz = (int)( (box->Zmax-box->Zmin)/(2*(*h)) );
  }

  box->N  = (box->Nx)*(box->Ny)*(box->Nz);
  
  double min_val = fmin((box->Xmax-box->Xmin)/box->Nx,fmin((box->Ymax-box->Ymin)/box->Ny,(box->Zmax-box->Zmin)/box->Nz));
  box->width  = (int)( 0.5 + 2*(*h)/min_val );
  box->hbegin = kh_init(0);
  box->hend   = kh_init(1);

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

int compute_hash_MC2D(int64_t N, SPHparticle *lsph, linkedListBox *box){

	if(lsph==NULL)
		return 1;

	if(box==NULL)
		return 1;

	const double etax = box->Nx/(box->Xmax - box->Xmin);
	const double etay = box->Ny/(box->Ymax - box->Ymin);

	for(int64_t i=0;i<N;i+=1){
		uint32_t kx,ky;
		kx = (uint32_t)(etax*(lsph->x[i] - box->Xmin));
		ky = (uint32_t)(etay*(lsph->y[i] - box->Ymin));

		if((kx<0)||(ky<0))
			return 1;
		else if((kx>=box->Nx)||(ky>=box->Nx))
			return 1;
		else
			lsph->hash[2*i] = ullMC2Dencode(kx,ky);
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
	// TODO: possible bug? verify 2*(N-1) instead of N-1
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
			printf("hash - (begin,end) = %lu - (%lu,%lu): %ld\n",
				kh_key(box->hbegin, kbegin),
				kh_value(box->hbegin, kbegin),
				kh_value(box->hend, kend),
				((int64_t)kh_value(box->hend, kend))-((int64_t)kh_value(box->hbegin, kbegin)));
		}
	}

	return 0;
}

int print_neighbour_list_MC3D(int64_t Nmax,unsigned int width,unsigned int stride,SPHparticle *lsph,linkedListBox *box){

	int64_t nblist[(2*width+1)*(2*width+1)*(2*width+1)];
	
	for(int64_t i=0;i<Nmax;i+=(int64_t)stride){
		int res = neighbour_hash_3d(lsph->hash[i],nblist,width,box);
		if(res!=0)
			printf("invalid neighbour_hash_3d calculation\n");
		printf("origin hash %lu : (%d,%d,%d) \n",lsph->hash[i],ullMC3DdecodeX(lsph->hash[i]),
															  ullMC3DdecodeY(lsph->hash[i]),
															  ullMC3DdecodeZ(lsph->hash[i]));
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
		int res = neighbour_hash_2d(lsph->hash[i],nblist,width,box);
		if(res!=0)
			printf("invalid neighbour_hash_3d calculation\n");

		printf("origin hash %lu : (%d,%d) \n",lsph->hash[i],ullMC2DdecodeX(lsph->hash[i]),
															  											  ullMC2DdecodeY(lsph->hash[i]));
		for(unsigned int j=0;j<(2*width+1)*(2*width+1);j+=1){
			if(nblist[j]>=0)
				printf("    neighbour hash %lu : (%d,%d) \n",nblist[j],ullMC3DdecodeX(lsph->hash[i]),
																		  												 ullMC3DdecodeY(lsph->hash[i]));
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

	const double etax = box->Nx/(box->Xmax - box->Xmin);
	const double etay = box->Ny/(box->Ymax - box->Ymin);
	const double etaz = box->Nz/(box->Zmax - box->Zmin);

	for(int64_t i=0;i<Nmax;i+=(int64_t)stride){
		uint32_t kx,ky,kz;
		int res = neighbour_hash_3d(lsph->hash[i],nblist,width,box);
		if(res!=0)
			printf("invalid neighbour_hash_3d calculation\n");

		kx = (uint32_t)(etax*(lsph->x[i] - box->Xmin));
		ky = (uint32_t)(etay*(lsph->y[i] - box->Ymin));
		kz = (uint32_t)(etaz*(lsph->z[i] - box->Zmin));
		
		fprintf(fp,"base   hash %lu : (%d,%d,%d) \n",lsph->hash[i],kx,ky,kz);
		fprintf(fp,"origin hash %lu : (%d,%d,%d) \n",lsph->hash[i],ullMC3DdecodeX(lsph->hash[i]),
					         									  												 ullMC3DdecodeY(lsph->hash[i]),
							    								  													 ullMC3DdecodeZ(lsph->hash[i]));

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
	int res;
  khiter_t kbegin,kend;
  int64_t node_hash=-1,node_begin=0, node_end=0;
  int64_t nb_begin= 0, nb_end = 0;
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
			if(res!=0)
				printf("invalid neighbour_hash_3d calculation\n");
      for(int64_t ii=node_begin;ii<node_end;ii+=1){

      	fprintf(fp,"\"%ld\":[",lsph->id[ii]);
      	for(int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){
      		if(nblist[j]>=0){
      			//nb_hash  = nblist[j];
          	nb_begin = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) );
          	nb_end   = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) );

          	for(int64_t jj=nb_begin;jj<nb_end;jj+=1)
          		fprintf(fp,"%ld, ",lsph->id[jj]);
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

	const double etax = box->Nx/(box->Xmax - box->Xmin);
	const double etay = box->Ny/(box->Ymax - box->Ymin);

	for(int64_t i=0;i<Nmax;i+=(int64_t)stride){
		uint32_t kx,ky;
		int res = neighbour_hash_2d(lsph->hash[i],nblist,width,box);
		if(res!=0)
			printf("invalid neighbour_hash_3d calculation\n");

		kx = (uint32_t)(etax*(lsph->x[i] - box->Xmin));
		ky = (uint32_t)(etay*(lsph->y[i] - box->Ymin));
		
		fprintf(fp,"base   hash %lu : (%d,%d) \n",lsph->hash[i],kx,ky);
		fprintf(fp,"origin hash %lu : (%d,%d) \n",lsph->hash[i],ullMC2DdecodeX(lsph->hash[i]),
					         									  											ullMC2DdecodeY(lsph->hash[i]));		
		
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

////////////////////////////////////////////////////////////////////
// https://stackoverflow.com/questions/16007640/openmp-parallel-quicksort

#define TASK_SIZE 24

int64_t partition(int64_t * a, int64_t p, int64_t r)
{
  int64_t lt[2*(r-p)];
  int64_t gt[2*(r-p)];
  int64_t i;
  int64_t j;
  int64_t key0 = a[2*r+0];
  int64_t key1 = a[2*r+1];
  int64_t lt_n = 0;
  int64_t gt_n = 0;

  for(i = p; i < r; i++){
    if(a[2*i] < a[2*r]){
      lt[2*lt_n+0] = a[2*i+0];
      lt[2*lt_n+1] = a[2*i+1];
      lt_n++;
    }else{
      gt[2*gt_n+0] = a[2*i+0];
      gt[2*gt_n+1] = a[2*i+1];
      gt_n++;
    }   
  }   

  for(i = 0; i < lt_n; i++){
    a[2*(p + i)+0] = lt[2*i+0];
    a[2*(p + i)+1] = lt[2*i+1];
  }   

  a[2*(p + lt_n)+0] = key0;
  a[2*(p + lt_n)+1] = key1;

  for(j = 0; j < gt_n; j++){
    a[2*(p + lt_n + j + 1)+0] = gt[2*j+0];
    a[2*(p + lt_n + j + 1)+1] = gt[2*j+1];
  }   

  return p + lt_n;
}

void quicksort_omp(int64_t * a, int64_t p, int64_t r)
{
	int64_t div;

  if(p < r){ 
    div = partition(a, p, r); 
    #pragma omp task shared(a) if(r - p > TASK_SIZE) 
    quicksort_omp(a, p, div - 1); 
    #pragma omp task shared(a) if(r - p > TASK_SIZE)
    quicksort_omp(a, div + 1, r); 
  }
}