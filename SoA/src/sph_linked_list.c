/*
 * SPDX-License-Identifier:  BSD-3-Clause
 * 
 * sph_linked_list.c : 
 *     Header containing the declarions for several
 *     cell linked list operations, including hash
 *     calculations, setup hash tables and neighbour finding.
 *
 * (C) Copyright 2021 José Hugo Elsas
 * Author: José Hugo Elsas <jhelsas@gmail.com>
 *
 */

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

	if(data1[0] < data2[0])         // data[0] is the hash value, 
		return -1;                    // data[1] is the corresponding 
	else if(data1[0] == data2[0])   // in the unsorted array
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

/*
 *  Function SPHparticle_SoA_malloc:
 *    Easily allocate the SPHparticle array
 * 
 *    Arguments:
 *       N          <int>     : Number of Particles
 *     lsph <SPHparticle**>   : SoA Particle struct reference
 */
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


/*
 *  Function SPHparticleSOA_safe_free:
 *    Easily free the SPHparticle array
 * 
 *    Arguments:
 *       N          <int>     : Number of Particles
 *     lsph <SPHparticle**>   : SoA Particle struct reference
 */
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

/*
 *  Function gen_unif_rdn_pos:
 *    Generate particles positions uniformely in a [0,1]^3 box
 * 
 *    Arguments:
 *       N          <int>     : Number of Particles
 *     lsph <SPHparticle**>   : SoA Particles array
 *     seed         <int>     : seed for the PRNG
 */
int gen_unif_rdn_pos(int64_t N, int seed, SPHparticle *lsph){

	const gsl_rng_type *T=NULL;
	gsl_rng *r=NULL;

	if(lsph==NULL)
		return 1;

	gsl_rng_env_setup();                      // Initialize environment for PRNG generator

	T = gsl_rng_default;                      // Set generator type as default type
	r = gsl_rng_alloc(T);                     // allocate state for it
	gsl_rng_set(r,seed);                      // set seed accordingly

	for(int64_t i=0;i<N;i+=1){
		lsph->x[i] = gsl_rng_uniform(r);        // Generate a uniform value for X component between 0 and 1
		lsph->y[i] = gsl_rng_uniform(r);        // Generate a uniform value for Y component between 0 and 1
		lsph->z[i] = gsl_rng_uniform(r);        // Generate a uniform value for Z component between 0 and 1

		lsph->ux[i] = 0.0; lsph->Fx[i] = 0.0;   // set velocity and force components as zero
		lsph->uy[i] = 0.0; lsph->Fy[i] = 0.0;   // as they will not be needed
    lsph->uz[i] = 0.0; lsph->Fz[i] = 0.0;

		lsph->nu[i]   = 1.0/N;                  // set each particle mass as to the total mass be 1
		lsph->rho[i]  = 0.0;                    // initialize density to zero
		lsph->id[i]   = (int64_t) i;            // set the particle's id
		lsph->hash[2*i+0] = (int64_t) 0;        // initialize particle's hash value as zero
		lsph->hash[2*i+1] = (int64_t) i;        // set particle's index position by the position in current array
	}

	gsl_rng_free(r);                          // release the PRNG

	return 0;
}

/*
 *  Function gen_unif_rdn_pos_box:
 *    Generate particles positions uniformely in a 
 *    [Xmin,Xmax]x[Ymin,Ymax]x[Zmin,Zmax] box
 * 
 *    Arguments:
 *       N          <int>     : Number of Particles
 *     lsph <SPHparticle**>   : SoA Particles array
 *     seed         <int>     : seed for the PRNG
 *      box  <linkedListBox*> : Cell Linked List box
 */
int gen_unif_rdn_pos_box(int64_t N, int seed, linkedListBox *box,SPHparticle *lsph){

  const gsl_rng_type *T=NULL;
  gsl_rng *r=NULL;

  if(lsph==NULL)
    return 1;

	gsl_rng_env_setup();                      // Initialize environment for PRNG generator

	T = gsl_rng_default;                      // Set generator type as default type
	r = gsl_rng_alloc(T);                     // allocate state for it
	gsl_rng_set(r,seed);                      // set seed accordingly

  for(int64_t i=0;i<N;i+=1){
    lsph->x[i] = gsl_rng_uniform(r)*(box->Xmax-box->Xmin) // Generate a uniform value for X 
                     + box->Xmin;                         // component between Xmin and Xmax
    lsph->y[i] = gsl_rng_uniform(r)*(box->Ymax-box->Ymin) // Generate a uniform value for Y
                     + box->Ymin;                         // component between Ymin and Ymax
    lsph->z[i] = gsl_rng_uniform(r)*(box->Zmax-box->Zmin) // Generate a uniform value for Z 
                     + box->Zmin;                         // component between Zmin and Zmax

		lsph->ux[i] = 0.0; lsph->Fx[i] = 0.0;   // set velocity and force components as zero
		lsph->uy[i] = 0.0; lsph->Fy[i] = 0.0;   // as they will not be needed
    lsph->uz[i] = 0.0; lsph->Fz[i] = 0.0;

		lsph->nu[i]   = 1.0/N;                  // set each particle mass as to the total mass be 1
		lsph->rho[i]  = 0.0;                    // initialize density to zero
		lsph->id[i]   = (int64_t) i;            // set the particle's id
		lsph->hash[2*i+0] = (int64_t) 0;        // initialize particle's hash value as zero
		lsph->hash[2*i+1] = (int64_t) i;        // set particle's index position by the position in current array
  }

  gsl_rng_free(r);

  return 0;
}

/*
 *  Function compute_hash_MC3D:
 *    Compute the Morton Z hashes for all particles based on 
 *    the parameters given by the Cell Linked List box
 * 
 *    Arguments:
 *       N          <int>     : Number of Particles
 *     lsph <SPHparticle**>   : SoA Particles array
 *     box  <linkedListBox*>  : Box of cell linked lists
 */
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
		kx = (uint32_t)(etax*(lsph->x[i] - box->Xmin));      // Compute cell index in the X direction
		ky = (uint32_t)(etay*(lsph->y[i] - box->Ymin));      // Compute cell index in the Y direction
		kz = (uint32_t)(etaz*(lsph->z[i] - box->Zmin));      // Compute cell index in the Z direction

		if((kx<0)||(ky<0)||(kz<0))                           // The can't be negative indexes
			return 1;
		else if((kx>=box->Nx)||(ky>=box->Nx)||(kz>=box->Nx)) // Nor indexes greater than the upper bounds
			return 1;
		else{
			lsph->hash[2*i] = ullMC3Dencode(kx,ky,kz);         // If ok, compute the corresponding Morton Z hash
		}
	}

	return 0;
}

/*
 * To swap two arrays, copy data from the original array into 
 * a swap buffer in the right order, and then overrite the original array. 
 */
#define swap_loop(N,lsph,temp_swap,member,type) for(int64_t i=0;i<(N);i+=1)                            \
																	 	              (temp_swap)[i] = (lsph)->member[(lsph)->hash[2*i+1]];\
																		            memcpy((lsph)->member,temp_swap,(N)*sizeof(type))

/*
 *  Function reorder_lsph_SoA:
 *    reorder the array according to the indexes reordered according to 
 *    their hashes. 
 * 
 *    Arguments:
 *       N          <int>     : Number of Particles
 *     lsph <SPHparticle**>   : SoA Particles array
 *     swap_arr  <void*>      : buffer for reordering data in the array
 */
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

/*
 *  Function setup_interval_hashtables:
 *    Setup the information where each cell begins and ends in the SPH
 *    Array in a pair of hash tables for quick consultation. 
 *    This effectively completes the creation of the implicit 
 *    cell linked list structure initiated by sorting the array 
 *    according to hashes. 
 * 
 *    Arguments:
 *       N          <int>     : Number of Particles
 *     lsph <SPHparticle**>   : SoA Particles array
 *     box  <linkedListBox*>  : Box of cell linked lists
 */
int setup_interval_hashtables(int64_t N,SPHparticle *lsph,linkedListBox *box){

	int ret;
	int64_t hash0 = lsph->hash[2*0];                             // Store the first hash in hash0
	khiter_t kbegin,kend;

	if(lsph==NULL)
		return 1;

	if(box==NULL)
		return 1;

	kbegin = kh_put(0, box->hbegin, lsph->hash[2*0], &ret);      // Insert start hash as first value
	kh_value(box->hbegin, kbegin) = (int64_t)0;                  // Set the start index of first cell as 0
	for(int64_t i=0;i<N;i+=1){
		lsph->hash[i] = lsph->hash[2*i];                           // Clean the first half of the array only to contain hashes
		if(lsph->hash[i] == hash0)                                 // If the the particle hash is the same as the previous
			continue;                                                // skip because they are in the same cell
		hash0 = lsph->hash[i];                                     // otherwise, store the new hash in hash0
		
		kend   = kh_put(1, box->hend  , lsph->hash[i-1], &ret);    // If so, insert the previous hash as an cell end
		kh_value(box->hend  , kend)   = i;                         // and set this index as the upper index of that cell
		
		kbegin = kh_put(0, box->hbegin, lsph->hash[i  ], &ret);    // and then insert this new hash as the begining
		kh_value(box->hbegin, kbegin) = i;                         // of a new cell in with the lower index as this index
	}
	kend   = kh_put(1, box->hend  , lsph->hash[2*(N-1)], &ret);  // finally set the last hash for the end of the last cell,
	kh_value(box->hend  , kend)   = N;                           // and the number of particles as the index of the last cell

	return 0;
}

/*
 *  Function neighbour_hash_3d:
 *    Find the hashes of all valid cells that neighbor cell with
 *    int hash. 
 * 
 *    Arguments:
 *       hash    <int64_t>    : Hash of the central cell 
 *       nblist <int64_t*>    : Array for hashes of the neighbor cells
 *       width     <int>      : Width of the neighborhood 
 *     box  <linkedListBox*>  : Box of cell linked lists
 */
int neighbour_hash_3d(int64_t hash,int64_t *nblist,int width, linkedListBox *box){
	int idx=0,kx=0,ky=0,kz=0;

	kx = ullMC3DdecodeX(hash);   // To find a given cell's neighbors, start by first 
	ky = ullMC3DdecodeY(hash);   // recovering the integer index of that cell in 
	kz = ullMC3DdecodeZ(hash);   // the overall box domain

	for(int ix=-width;ix<=width;ix+=1){            // iterate over all cells surrounding the original cell
		for(int iy=-width;iy<=width;iy+=1){          // as long as they are within width of the original cell
			for(int iz=-width;iz<=width;iz+=1){        
				if((kx+ix<0)||(ky+iy<0)||(kz+iz<0))                              // if the cell is off bounds for having lower index then 0
					nblist[idx++] = -1;                                            // annotate the cell as invalid
				else if( (kx+ix>=box->Nx)||(ky+iy>=box->Ny)||(kz+iz>=box->Nz) )  // or if it is off bounds for having higher index then
					nblist[idx++] = -1;                                            // the maximum, also annotate as invalid
				else if( kh_get(0, box->hbegin, ullMC3Dencode(kx+ix,ky+iy,kz+iz)) == kh_end(box->hbegin) ) // Also, if no corresponding hash
					nblist[idx++] = -1;                                            // is not found in the hash table, also annotate as invalid
				else                                                             // at last, if it has no problems
					nblist[idx++] = ullMC3Dencode(kx+ix,ky+iy,kz+iz);              // annotate the hash of the corresponding cell
			}
		}
	}
	
	return 0;
}

/*
 *  Function count_box_pairs:
 *    Count the number of valid cell pairs
 * 
 *    Arguments:
 *     box  <linkedListBox*>  : Box of cell linked lists
 */
int count_box_pairs(linkedListBox *box){
  int64_t pair_count = 0;                                                     // initialize the number of pairs as zero

  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){ // iterate over the cells
    int64_t node_hash=-1; 
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){                                       // check if the cell is valid
      node_hash = kh_key(box->hbegin, kbegin);                                // get the hash of that cell
      
      neighbour_hash_3d(node_hash,nblist,box->width,box);                     // fetch the list of neighbors
      for(int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){ // and iterate over those neighbors
        if(nblist[j]>=0){                                                     // if the neighbor is valid
          pair_count += 1;                                                    // count it as a valid for a pair
        }
      }
    }
  }

  return pair_count;
}

/*
 *  Function setup_box_pairs:
 *    Pre-compute indexes for all valid cell pairs. 
 * 
 *  Arguments:
 *     box  <linkedListBox*>  : Box of cell linked lists
 *  Returns:
 *     node_begin <int64_t*>  : Array for receiver cell begin indexes
 *     node_end   <int64_t*>  : Array for receiver cell end indexes
 *     nb_begin   <int64_t*>  : Array for sender cell begin indexes
 *     nb_end     <int64_t*>  : Array for sender cell end indexes
 */
int setup_box_pairs(linkedListBox *box,
                    int64_t *node_begin,int64_t *node_end,
                    int64_t *nb_begin,int64_t *nb_end)
{
  int64_t pair_count = 0,particle_pair_count = 0;

  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){    // iterate over the cells
    int64_t node_hash=-1;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){                                                       // check if the cell is valid
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));                     // get the iterator for the end index

      node_hash = kh_key(box->hbegin, kbegin);                                                // and also get the corresponding hash

      neighbour_hash_3d(node_hash,nblist,box->width,box);                                     // fetch the list of neighbors
      for(int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){                 // and iterate over those neighbors
        if(nblist[j]>=0){                                                                     // check if the neighbor is valid
          node_begin[pair_count] = kh_value(box->hbegin, kbegin);                             // get the cell's begin index
          node_end[pair_count]   = kh_value(box->hend, kend);                                 // get the cell's end index
          nb_begin[pair_count]   = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) ); // get the neighbor's begin index
          nb_end[pair_count]     = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) ); // get the neighbor's end index

          particle_pair_count += (nb_end[pair_count]-nb_begin[pair_count])*                   // just for bookeeping/profiling, 
                                       (node_end[pair_count]-node_begin[pair_count]);         // annotate the number of particle pairs
          pair_count += 1;                                                                    // that are part of the computation
        }
      }
    }
  }

  printf("particle_pair_count = %ld\n",particle_pair_count);

  return pair_count;
}

/*
 *  Function setup_unique_box_pairs:
 *    Pre-compute indexes for all valid unique cell pairs. 
 *    by unique means that if [node_begin,node_end)x[nb_begin,nb_end)
 *    is in the list, [nb_begin,nb_end)x[node_begin,node_end) is not. 
 * 
 *  Arguments:
 *     box  <linkedListBox*>  : Box of cell linked lists
 *  Returns:
 *     node_begin <int64_t*>  : Array for receiver cell begin indexes
 *     node_end   <int64_t*>  : Array for receiver cell end indexes
 *     nb_begin   <int64_t*>  : Array for sender cell begin indexes
 *     nb_end     <int64_t*>  : Array for sender cell end indexes
 */
int setup_unique_box_pairs(linkedListBox *box,
                           int64_t *node_begin,int64_t *node_end,
                           int64_t *nb_begin,int64_t *nb_end)
{
  int64_t pair_count = 0, particle_pair_count = 0;

  for (khint32_t kbegin = kh_begin(box->hbegin); kbegin != kh_end(box->hbegin); kbegin++){      // iterate over the cells
    int64_t node_hash=-1;
    int64_t nblist[(2*box->width+1)*(2*box->width+1)*(2*box->width+1)];

    if (kh_exist(box->hbegin, kbegin)){                                                         // check if the cell is valid
      khint32_t kend = kh_get(1, box->hend, kh_key(box->hbegin, kbegin));                       // get the iterator for the end index

      node_hash = kh_key(box->hbegin, kbegin);                                                  // and also get the corresponding hash

      neighbour_hash_3d(node_hash,nblist,box->width,box);                                       // fetch the list of neighbors
      for(int j=0;j<(2*box->width+1)*(2*box->width+1)*(2*box->width+1);j+=1){                   // and iterate over those neighbors
        if(nblist[j]>=0){                                                                       // check if the neighbor is valid
          if(kh_value(box->hbegin, kbegin) <=                                                   // and if the node_index is smaller
                     kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) ))                 // then the neighbor index
          {
            node_begin[pair_count] = kh_value(box->hbegin, kbegin);                             // get the cell's begin index
            node_end[pair_count]   = kh_value(box->hend, kend);                                 // get the cell's end index
            nb_begin[pair_count]   = kh_value(box->hbegin, kh_get(0, box->hbegin, nblist[j]) ); // get the neighbor's begin index
            nb_end[pair_count]     = kh_value(box->hend  , kh_get(1, box->hend  , nblist[j]) ); // get the neighbor's end index

            particle_pair_count += (nb_end[pair_count]-nb_begin[pair_count])*                   // just for bookeeping/profiling, 
                                         (node_end[pair_count]-node_begin[pair_count]);         // annotate the number of particle pairs
            pair_count += 1;                                                                    // that are part of the computation
          }
        }
      }
    }
  }

  printf("particle_pair_count = %ld\n",particle_pair_count);

  return pair_count;
}

/*
 *  Function print_sph_particles_density:
 *    Write in disk the density values for the computed SPH particles. 
 * 
 *  Arguments:
 *     prefix  <const char*>  : prefix for the file name 
 *     is_cll    <bool>       : boolean defined naive or cell linked list case
 *       N      <int64_t>     : Number of SPH particles
 *       h      <double>      : smoothing length 
 *     seed    <long int>     : seed utilized for the PRNG
 *     runs     <int>         : Number of runs executed
 *     lsph  <SPHparticle*>   : Array of SPH particles
 *     box  <linkedListBox*>  : Box of cell linked lists
 *  Returns:
 *     0  <int>   : Error code
 *    fp <FILE*>  : Written file with particle densities
 */
int print_sph_particles_density(const char *prefix, bool is_cll, int64_t N, double h, 
																long int seed, int runs, SPHparticle *lsph, linkedListBox *box){
	FILE *fp;
	char filename[1024+1];

	if(is_cll){
		sprintf(filename,
						"data/cd3d(%s,runs=%d)-P(seed=%ld,N=%ld,h=%lg)-B(Nx=%d,Ny=%d,Nz=%d)-D(%lg,%lg,%lg,%lg,%lg,%lg).csv",
						prefix,runs,seed,N,h,box->Nx,box->Ny,box->Nz,box->Xmin,box->Ymin,box->Zmin,box->Xmax,box->Ymax,box->Zmax);

		fp = fopen(filename,"w");
		fprintf(fp,"id,x,y,z,rho\n");
		for(int64_t i=0;i<N;i+=1)
			fprintf(fp,"%ld,%lf,%lf,%lf,%lf\n",lsph->id[i],lsph->x[i],lsph->y[i],lsph->z[i],lsph->rho[i]);
		fclose(fp);
	} 
	else{
		sprintf(filename,
						"data/cd3d(%s,runs=%d)-P(seed=%ld,N=%ld,h=%lg)-B(Nx=%d,Ny=%d,Nz=%d)-D(%lg,%lg,%lg,%lg,%lg,%lg).csv",
						prefix,runs,seed,N,h,box->Nx,box->Ny,box->Nz,box->Xmin,box->Ymin,box->Zmin,box->Xmax,box->Ymax,box->Zmax);

		fp = fopen(filename,"w");
		fprintf(fp,"id,x,y,z,rho\n");
		for(int64_t i=0;i<N;i+=1)
			fprintf(fp,"%ld,%lf,%lf,%lf,%lf\n",lsph->id[i],lsph->x[i],lsph->y[i],lsph->z[i],lsph->rho[i]);
		fclose(fp);
	}
	

	return 0;
}

/*
 *  Function print_sph_particles_density:
 *    Write in disk the density values for the computed SPH particles. 
 * 
 *  Arguments:
 *     prefix  <const char*>  : prefix for the file name 
 *     is_cll    <bool>       : boolean defined naive or cell linked list case
 *       N      <int64_t>     : Number of SPH particles
 *       h      <double>      : smoothing length 
 *     seed    <long int>     : seed utilized for the PRNG
 *     runs     <int>         : Number of runs executed
 *     lsph  <SPHparticle*>   : Array of SPH particles
 *     box  <linkedListBox*>  : Box of cell linked lists
 *    times   <double*>       : timings for each of the processing steps
 *  Returns:
 *     0  <int>   : Error code
 *    fp <FILE*>  : Written file with particle densities
 */
int print_time_stats(const char *prefix, bool is_cll, int64_t N, double h, 
										 long int seed, int runs, SPHparticle *lsph, linkedListBox *box,double *times){
  FILE *fp;
	char filename[1024+1];

	if(is_cll){
    const int COMPUTE_BLOCKS = 5;
    double t[COMPUTE_BLOCKS], dt[COMPUTE_BLOCKS], total_time, dtotal_time;
  	sprintf(filename,
						"data/times-(%s,runs=%d)-P(seed=%ld,N=%ld,h=%lg)-B(Nx=%d,Ny=%d,Nz=%d)-D(%lg,%lg,%lg,%lg,%lg,%lg).csv",
						prefix,runs,seed,N,h,box->Nx,box->Ny,box->Nz,box->Xmin,box->Ymin,box->Zmin,box->Xmax,box->Ymax,box->Zmax);

  	fp = fopen(filename,"w");
		fprintf(fp,"id, compute_hash_MC3D, sorting, reorder_lsph_SoA, setup_interval_hashtables, compute_density\n");
		for(int run=0;run<runs;run+=1)
			fprintf(fp,"%d,%lf,%lf,%lf,%lf,%lf\n",run,times[COMPUTE_BLOCKS*run+0],times[COMPUTE_BLOCKS*run+1],times[COMPUTE_BLOCKS*run+2],
                                                times[COMPUTE_BLOCKS*run+3],times[COMPUTE_BLOCKS*run+4]);
		fclose(fp);

  	total_time = 0.;
  	for(int k=0;k<COMPUTE_BLOCKS;k+=1){
    	t[k]=0.; dt[k]=0.;
    	for(int run=0;run<runs;run+=1)
      	t[k] += times[COMPUTE_BLOCKS*run+k];
    	t[k] /= runs;
    	for(int run=0;run<runs;run+=1)
      	dt[k] += (times[COMPUTE_BLOCKS*run+k]-t[k])*(times[COMPUTE_BLOCKS*run+k]-t[k]);
    	dt[k] /= runs;
    	dt[k] = sqrt(dt[k]);

    	total_time += t[k];
  	}

  	dtotal_time = 0.;
  	for(int run=0;run<runs;run+=1){
	    double rgm = 0.;
  	  for(int k=0;k<COMPUTE_BLOCKS;k+=1)
    	  rgm += times[COMPUTE_BLOCKS*run+k];

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
	}
	else{
    const int COMPUTE_BLOCKS = 1;
    double t[COMPUTE_BLOCKS], dt[COMPUTE_BLOCKS], total_time, dtotal_time;
		sprintf(filename,
						"data/times-(%s,runs=%d)-P(seed=%ld,N=%ld,h=%lg)-B(Nx=%d,Ny=%d,Nz=%d)-D(%lg,%lg,%lg,%lg,%lg,%lg).csv",
						prefix,runs,seed,N,h,box->Nx,box->Ny,box->Nz,box->Xmin,box->Ymin,box->Zmin,box->Xmax,box->Ymax,box->Zmax);

  	fp = fopen(filename,"w");
		fprintf(fp,"id, compute_density\n");
		for(int run=0;run<runs;run+=1)
			fprintf(fp,"%d %lf\n",run,times[COMPUTE_BLOCKS*run+0]);
		fclose(fp);

  	total_time = 0.;
  	for(int k=0;k<COMPUTE_BLOCKS;k+=1){
    	t[k]=0.; dt[k]=0.;
    	for(int run=0;run<runs;run+=1)
      	t[k] += times[COMPUTE_BLOCKS*run+k];
    	t[k] /= runs;
    	for(int run=0;run<runs;run+=1)
      	dt[k] += (times[COMPUTE_BLOCKS*run+k]-t[k])*(times[COMPUTE_BLOCKS*run+k]-t[k]);
    	dt[k] /= runs;
    	dt[k] = sqrt(dt[k]);

    	total_time += t[k];
  	}

  	dtotal_time = 0.;
  	for(int run=0;run<runs;run+=1){
	    double rgm = 0.;
  	  for(int k=0;k<1;k+=1)
    	  rgm += times[COMPUTE_BLOCKS*run+k];

    	dtotal_time += (rgm-total_time)*(rgm-total_time);
  	}
  	dtotal_time /= runs;
  	dtotal_time = sqrt(dtotal_time);

  	printf("compute_density_3d naive %s : %.5lf +- %.6lf s : %.3lf%%\n",prefix,total_time,dtotal_time,100.);
	}


  return 0;
}

