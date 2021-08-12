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

#include "../sph_data_types.h"
#include "../sph_linked_list.h"

/* 
typedef struct SPHparticle{
  int64_t *id,*hash;
  double *nu,*rho;
  double *x,*y,*z;
  double *ux,*uy,*uz;
  double *Fx,*Fy,*Fz;
}
*/

int SPHparticle_SoA_malloc(int N,SPHparticle **lsph){
  bool success=true;
  (*lsph) = (SPHparticle*)malloc(1*sizeof(SPHparticle));

  (*lsph).x    = (double*)malloc(N*sizeof(double));
  (*lsph).y    = (double*)malloc(N*sizeof(double));
  (*lsph).z    = (double*)malloc(N*sizeof(double));

  (*lsph).ux   = (double*)malloc(N*sizeof(double));
  (*lsph).uy   = (double*)malloc(N*sizeof(double));
  (*lsph).uz   = (double*)malloc(N*sizeof(double));

  (*lsph).Fx   = (double*)malloc(N*sizeof(double));
  (*lsph).Fy   = (double*)malloc(N*sizeof(double));
  (*lsph).Fz   = (double*)malloc(N*sizeof(double));

  (*lsph).nu   = (double*)malloc(N*sizeof(double));
  (*lsph).rho  = (double*)malloc(N*sizeof(double));
  
  (*lsph).id   = (int64_t*)malloc(N*sizeof(int64_t));
  (*lsph).hash = (int64_t*)malloc(N*sizeof(int64_t));

  if(success)
    return false;
  else{
    if((*lsph).x != NULL)
      free((*lsph).x);

    if((*lsph).y != NULL)
      free((*lsph).y);

    if((*lsph).z != NULL)
      free((*lsph).z);

    if((*lsph).ux != NULL)
      free((*lsph).ux);

    if((*lsph).uy != NULL)
      free((*lsph).uy);

    if((*lsph).uz != NULL)
      free((*lsph).uz);

    if((*lsph).Fx != NULL)
      free((*lsph).Fx);

    if((*lsph).Fy != NULL)
      free((*lsph).Fz);

    if((*lsph).Fz != NULL)
      free((*lsph).Fz);

    if((*lsph).nu != NULL)
      free((*lsph).nu);

    if((*lsph).rho != NULL)
      free((*lsph).rho);

    if((*lsph).id != NULL)
      free((*lsph).id);

    if((*lsph).hash != NULL)
      free((*lsph).hash);

    return true;
  }
}

int main(){

  int j=0,numThreads=6,err;
  int64_t N = 100000;
  double h=0.05;
  linkedListBox *box;
  SPHparticle *lsph;

  //lsph = (SPHparticle*)malloc(N*sizeof(SPHparticle));
  err = SPHparticle_SoA_malloc(&lsph);
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