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
  int success=1;
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

int main(){

  int err;
  int64_t N = 10000;
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

  err = compute_hash_MC3D(N,lsph,box);
  
  //for(int64_t i=0;i<N;i+=1)
  //  printf("%ld %ld\n",lsph->hash[2*i],lsph->hash[2*i+1]);

  qsort(lsph->hash,N,2*sizeof(int64_t),compare_int64_t);

  //for(int64_t i=0;i<N;i+=1)
  //  printf("%ld %ld\n",lsph->hash[2*i],lsph->hash[2*i+1]);

  void *swap_arr = malloc(N*sizeof(double));
  err = reorder_lsph_SoA(N,lsph,swap_arr);
  if(err)
    printf("error in reorder_lsph_SoA\n");

  err = setup_interval_hashtables(N,lsph,box);
  if(err)
    printf("error in setup_interval_hashtables\n");

  err = print_neighbour_list_MC3D_lsph_ids_file(N,lsph,box);
  if(err)
    printf("error in print_neighbour_list_MC3D_lsph_ids_file\n");
  
  FILE *fp = fopen("data/nblist_ref.json","w");
  fprintf(fp,"{\n");
  for(int64_t i=0;i<N;i+=1){
    fprintf(fp,"\"%ld\":[",lsph->id[i]);
    for(int64_t j=0;j<N;j+=1){
      double dist = 0.;

      dist += (lsph->x[i]-lsph->x[j])*(lsph->x[i]-lsph->x[j]);
      dist += (lsph->y[i]-lsph->y[j])*(lsph->y[i]-lsph->y[j]);
      dist += (lsph->z[i]-lsph->z[j])*(lsph->z[i]-lsph->z[j]);

      dist = sqrt(dist);

      if(dist<=2*h){
        fprintf(fp,"%ld, ",lsph->id[j]);
      }
    }
    if(i<N-1)
      fprintf(fp,"-1],\n");
    else
      fprintf(fp,"-1]\n");
  }
  fprintf(fp,"}\n");
  fclose(fp);

  SPHparticleSOA_safe_free(N,&lsph);
  safe_free_box(box);
  free(swap_arr);

  return 0;
}