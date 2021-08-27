#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h>
#include <inttypes.h>

#include <mpi.h>
#include <omp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_heapsort.h>

#include "sph_data_types.h"
#include "sph_linked_list.h"

int64_t int64_cmpfunc (const void * a, const void * b) //what is it returning?
{
   return ( *(int64_t*)a - *(int64_t*)b ); //What is a and b?
}

KHASH_MAP_INIT_INT64(3, int)

int main(){
  MPI_Init(NULL,NULL);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int err;
  const int64_t N_global = 1000;
  int64_t N = N_global/size;
  double h=0.125;
  linkedListBox *box;
  SPHparticle *lsph;

  if(rank<size-1)
    N = N_global/size;
  else
    N = N_global - (N_global/size)*(size-1);

  err = SPHparticle_SoA_malloc(N,&lsph);
  if(err)
    printf("error in SPHparticle_SoA_malloc\n");

  err = gen_unif_rdn_pos(N,123123123,lsph);
  if(err)
    printf("error in gen_unif_rdn_pos\n");

  box = (linkedListBox*)malloc(1*sizeof(linkedListBox));

  box->Xmin.x = 0.0; box->Xmin.y = 0.0; box->Xmin.z = 0.0;
  box->Xmax.x = 1.0; box->Xmax.y = 1.0; box->Xmax.z = 1.0;
  box->Nx = (int)( (box->Xmax.x-box->Xmin.x)/(2*h) );
  box->Ny = (int)( (box->Xmax.y-box->Xmin.y)/(2*h) );
  box->Nz = (int)( (box->Xmax.z-box->Xmin.z)/(2*h) );
  box->N  = (box->Nx)*(box->Ny)*(box->Nz);
  box->width = 1;
  box->hbegin = kh_init(0);
  box->hend = kh_init(1);
  if(rank == 0)
    printf("box : %d %d %d\n",box->Nx,box->Ny,box->Nz);

  err = compute_hash_MC3D(N,lsph,box);
  
  qsort(lsph->hash,N,2*sizeof(int64_t),compare_int64_t);
  void *swap_arr = malloc(N*sizeof(double));
  err = reorder_lsph_SoA(N,lsph,swap_arr);
  if(err)
    printf("error in reorder_lsph_SoA\n");

  err = setup_interval_hashtables(N,lsph,box);
  if(err)
    printf("error in setup_interval_hashtables\n");

  int local_keys = kh_size(box->hbegin);
  int max_global_keys = 0;
  printf("%d: total number of keys: %d\n", rank,local_keys);
  MPI_Allreduce(&local_keys,&max_global_keys,1,MPI_INT, MPI_MAX,MPI_COMM_WORLD);

  printf("%d: total number of keys: %d\n", rank, size*max_global_keys);
  
  int64_t *local_hashes = (int64_t*)malloc(max_global_keys*sizeof(int64_t));
  int64_t *global_hashes = (int64_t*)malloc(size*max_global_keys*sizeof(int64_t));
  printf("max_global_keys = %d , total_number_of_keys = %d\n",max_global_keys, size*max_global_keys);
  {
    int64_t hash_index=0;
    for (khiter_t k = kh_begin(box->hbegin); k != kh_end(box->hbegin); ++k)
      if (kh_exist(box->hbegin, k)){
        local_hashes[hash_index] = kh_key(box->hbegin, k);
        hash_index+=1;
      }

    for(int k = hash_index;k<max_global_keys;k+=1)
      local_hashes[k] = -1;
  }
  
  /*
  int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
  */  
  MPI_Allgather(local_hashes,max_global_keys,MPI_INT64_T,global_hashes,
                max_global_keys,MPI_INT64_T, MPI_COMM_WORLD);

  qsort(global_hashes,max_global_keys*size,sizeof(int64_t),int64_cmpfunc);

  int64_t hash_count = 0;
  int64_t hash0 = global_hashes[0];
  for(int64_t i=1;i<size*max_global_keys;i+=1){
    if(global_hashes[i]>hash0)
      hash_count += 1;
      hash0 = global_hashes[i];
      global_hashes[hash_count] = hash0;
    }
  hash_count += 1;

  int64_t *rglobal_hashes = realloc(global_hashes,hash_count);
  if(rglobal_hashes!=NULL){
    if(global_hashes != rglobal_hashes){
      int64_t *temp = global_hashes;
      global_hashes = rglobal_hashes;
      free(global_hashes);
    }
  }

  printf("hash_count = %d\n",hash_count);
  for(unsigned int i=0;i<hash_count;i+=1)
    printf("%d : %d\n",i,global_hashes[i]);

  /*  
  if(rank==0){
    printf("hash_count = %ld\n",hash_count);
    for(int64_t i=0;i<max_global_keys*size;i+=1)
      printf("%ld ",global_hashes[i]);
    printf("\n");
  }
  */
 
  khash_t(3) *ht_global = kh_init(3);

  /*
  for(int64_t i=0;i<hash_count;i+=1){
    // kend   = kh_put(1, box->hend  , lsph->hash[i-1], &ret); 
    // kh_value(box->hend  , kend)   = i;
    khiter_t k = kh_put(3, ht_global, global_hashes[i], &ret); 
    printf("%ld ret : %d\n",i,ret);
    kh_value(ht_global, k) = 0;
  }*/

  //printf("ht_size = %d hash_count = %d\n",kh_size(ht_global),hash_count);

  
  int val;
  for(int64_t i=1;i<N;i+=1){
    khiter_t k = kh_get(3, ht_global, lsph->hash[i]);
    // kh_get(name, h, k)
    
    if(k != kh_end(ht_global)){
      val = kh_value(ht_global, k) ;
      val += 1;
      kh_value(ht_global, k) = val;
      printf("val = %d\n",val);
    }
    else{
      printf("panic! at the disco - %ld\n",i);
    }
  }

  printf("hsize = %d\n",kh_size(ht_global));

/*  printf("histogram at rank %d\n",rank);
  if(rank==0)
    for (khiter_t k = kh_begin(ht_global); k != kh_end(ht_global); ++k)
      if (kh_exist(ht_global, k)){
        printf("%ld : %lf\n",kh_value(ht_global, k),kh_value(ht_global, k) );
      }*/

  /*
  if(rank==0)
    for(int i=0;i<max_global_keys;i+=1)
      printf("%ld ",local_hashes[i]);
  printf("\n");
  */

  //printf("rank %d:\n",rank);
  //for(int i=0;i<kh_size(box->hbegin);i+=1)
  //  printf("%d: %ld\n",i,total_hashes[i]); 
  

  SPHparticleSOA_safe_free(N,&lsph);
  safe_free_box(box);
  free(swap_arr);  
  free(local_hashes);
  free(global_hashes);
  kh_destroy(3, ht_global);

  MPI_Finalize();
}

/*
int main(){

  int err;
  int64_t N = 100000;
  double h=0.05;
  linkedListBox *box;
  SPHparticle *lsph;

  err = SPHparticle_SoA_malloc(N,&lsph);
  if(err)
    printf("error in SPHparticle_SoA_malloc\n");

  err = gen_unif_rdn_pos(N,123123123,lsph);
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
  
  qsort(lsph->hash,N,2*sizeof(int64_t),compare_int64_t);

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
} */