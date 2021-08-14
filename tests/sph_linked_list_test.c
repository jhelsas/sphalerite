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

/* 
typedef struct SPHparticle{
  int64_t *id,*hash;
  double *nu,*rho;
  double *x,*y,*z;
  double *ux,*uy,*uz;
  double *Fx,*Fy,*Fz;
}
*/

#define safe_check_alloc(ptr,N,dtype) {                                          \
                                       (ptr) = (dtype*)malloc((N)*sizeof(dtype));\
                                       if((ptr)==NULL){                          \
                                         success=0;                              \
                                         goto finishlabel;                       \
                                       }                                         \
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


  safe_check_alloc((*lsph)->x   ,N,double);
  safe_check_alloc((*lsph)->y   ,N,double);
  safe_check_alloc((*lsph)->z   ,N,double);
  safe_check_alloc((*lsph)->ux  ,N,double);
  safe_check_alloc((*lsph)->uy  ,N,double);
  safe_check_alloc((*lsph)->uz  ,N,double);
  safe_check_alloc((*lsph)->Fx  ,N,double);
  safe_check_alloc((*lsph)->Fy  ,N,double);
  safe_check_alloc((*lsph)->Fz  ,N,double);
  safe_check_alloc((*lsph)->nu  ,N,double);
  safe_check_alloc((*lsph)->rho ,N,double);
  safe_check_alloc((*lsph)->id  ,N,int64_t);
  safe_check_alloc((*lsph)->hash,N,int64_t);

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

int main(){

  int j=0,numThreads=6,err;
  int64_t N = 100000;
  double h=0.05;
  linkedListBox *box;
  SPHparticle *lsph;

  //lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));
  err = SPHparticle_SoA_malloc(N,&lsph);
  if(err)
    printf("error in SPHparticle_SoA_malloc\n");

  err = gen_unif_rdn_pos( N,123123123,lsph);
  if(err)
    printf("error in gen_unif_rdn_pos\n");

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox));

  box->Xmin.x = -1.0; box->Xmin.y = -1.0; box->Xmin.z = -1.0;
  box->Xmax.x =  2.0; box->Xmax.y =  2.0; box->Xmax.z =  2.0;
  box->Nx = (int)( (box->Xmax.x-box->Xmin.x)/(2*h) );
  box->Ny = (int)( (box->Xmax.y-box->Xmin.y)/(2*h) );
  box->Nz = (int)( (box->Xmax.z-box->Xmin.z)/(2*h) );
  box->N  = (box->Nx)*(box->Ny)*(box->Nz);
  box->width = 1;
  box->hbegin = kh_init(0);
  box->hend = kh_init(1);

  printf("box: %d %d %d\n",box->Nx,box->Ny,box->Nz);

  err = compute_hash_MC3D(N,lsph,box);

  qsort(lsph,N,sizeof(SPHparticle),compare_SPHparticle);

  err = setup_interval_hashtables(N,lsph,box);
  
  print_neighbour_list_MC3D_lsph_ids_file(N,lsph,box);

  /*
  FILE *fp = fopen("data/nblist_ref.json","w");
  fprintf(fp,"{\n");
  for(int64_t i=0;i<N;i+=1){
    fprintf(fp,"\"%ld\":[",lsph[i].id);
    for(int64_t j=0;j<N;j+=1){
      double dist = 0.;

      dist += (lsph[i].r.x-lsph[j].r.x)*(lsph[i].r.x-lsph[j].r.x);
      dist += (lsph[i].r.y-lsph[j].r.y)*(lsph[i].r.y-lsph[j].r.y);
      dist += (lsph[i].r.z-lsph[j].r.z)*(lsph[i].r.z-lsph[j].r.z);

      dist = sqrt(dist);

      if(dist<=2*h){
        fprintf(fp,"%ld, ",lsph[j].id);
      }
    }
    if(i<N-1)
      fprintf(fp,"-1],\n");
    else
      fprintf(fp,"-1]\n");
  }
  fprintf(fp,"}\n");
  fclose(fp);*/

  free(lsph);
  safe_free_box(box);

  return 0;
}