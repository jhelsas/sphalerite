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
		lsph[i].r.x = gsl_rng_uniform(r);        // Generate a uniform value for X component between 0 and 1
		lsph[i].r.y = gsl_rng_uniform(r);        // Generate a uniform value for Y component between 0 and 1
		lsph[i].r.z = gsl_rng_uniform(r);        // Generate a uniform value for Z component between 0 and 1
		lsph[i].r.t = gsl_rng_uniform(r);        // Generate a uniform value for T component between 0 and 1

		lsph[i].u.x = 0.0;  lsph[i].u.y = 0.0;   // set velocity and force components as zero
		lsph[i].u.z = 0.0;  lsph[i].u.t = 0.0;   // as they will not be needed

		lsph[i].F.x = 0.0;  lsph[i].F.y = 0.0;
		lsph[i].F.z = 0.0;  lsph[i].F.t = 0.0;

		lsph[i].nu = 1.0/N;                      // set each particle mass as to the total mass be 1
		lsph[i].rho  = 0.0;                      // initialize density to zero
		lsph[i].id = i;                          // set the particle's id     
		lsph[i].hash = 0;                        // initialize particle's hash value as zero
	}

	gsl_rng_free(r);                           // release the PRNG

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
    lsph[i].r.x = gsl_rng_uniform(r)*(box->Xmax-box->Xmin)  // Generate a uniform value for X 
                          + box->Xmin;                      // component between Xmin and Xmax
    lsph[i].r.y = gsl_rng_uniform(r)*(box->Ymax-box->Ymin)  // Generate a uniform value for Y
                          + box->Ymin;                      // component between Ymin and Ymax
    lsph[i].r.z = gsl_rng_uniform(r)*(box->Zmax-box->Zmin)  // Generate a uniform value for Z 
                          + box->Zmin;                      // component between Zmin and Zmax

		lsph[i].u.x = 0.0;  lsph[i].u.y = 0.0;   // set velocity and force components as zero
		lsph[i].u.z = 0.0;  lsph[i].u.t = 0.0;

		lsph[i].F.x = 0.0;  lsph[i].F.y = 0.0;   // as they will not be needed
		lsph[i].F.z = 0.0;  lsph[i].F.t = 0.0;

		lsph[i].nu = 1.0/N;                    // set each particle mass as to the total mass be 1
    lsph[i].rho  = 0.0;                    // initialize density to zero
		lsph[i].id = i;                        // set the particle's id     
    lsph[i].hash = 0;                      // initialize particle's hash value as zero
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

	for(int64_t i=0;i<N;i+=1){
		uint32_t kx,ky,kz;
		kx = (uint32_t)((lsph[i].r.x - box->Xmin)*box->Nx/(box->Xmax - box->Xmin)); // Compute cell index in the X direction 
		ky = (uint32_t)((lsph[i].r.y - box->Ymin)*box->Ny/(box->Ymax - box->Ymin)); // Compute cell index in the Y direction
		kz = (uint32_t)((lsph[i].r.z - box->Zmin)*box->Nz/(box->Zmax - box->Zmin)); // Compute cell index in the Z direction

    if((kx<0)||(ky<0)||(kz<0))                           // The can't be negative indexes
      return 1;
    else if((kx>=box->Nx)||(ky>=box->Nx)||(kz>=box->Nx)) // Nor indexes greater than the upper bounds
      return 1;
		else
			lsph[i].hash = ullMC3Dencode(kx,ky,kz);
	}

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
	int64_t hash0 = lsph[0].hash;                             // Store the first hash in hash0
	khiter_t kbegin,kend;

	if(lsph==NULL)
		return 1;

	if(box==NULL)
		return 1;

	kbegin = kh_put(0, box->hbegin, lsph[0].hash, &ret);     // Insert start hash as first value
  kh_value(box->hbegin, kbegin) = (int64_t)0;              // Set the start index of first cell as 0
	for(int i=0;i<N;i+=1){
		if(lsph[i].hash == hash0)                              // If the the particle hash is the same as the previous
			continue;                                            // skip because they are in the same cell
		hash0 = lsph[i].hash;                                  // otherwise, store the new hash in hash0

		kend   = kh_put(1, box->hend  , lsph[i-1].hash, &ret); // If so, insert the previous hash as an cell end 
    kh_value(box->hend  , kend)   = i;                     // and set this index as the upper index of that cell

		kbegin = kh_put(0, box->hbegin, lsph[i  ].hash, &ret); // and then insert this new hash as the begining
    kh_value(box->hbegin, kbegin) = i;                     // of a new cell in with the lower index as this index
	}
	kend   = kh_put(1, box->hend  , lsph[N-1].hash, &ret);   // finally set the last hash for the end of the last cell,
  kh_value(box->hend  , kend)   = N;                       // and the number of particles as the index of the last cell

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
				if((kx+ix<0)||(ky+iy<0)||(kz+iz<0))                               // if the cell is off bounds for having lower index then 0
					nblist[idx++] = -1;                                             // annotate the cell as invalid
				else if( (kx+ix>=box->Nx)||(ky+iy>=box->Ny)||(kz+iz>=box->Nz) )   // or if it is off bounds for having higher index then                            
					nblist[idx++] = -1;                                             // the maximum, also annotate as invalid
				else if( kh_get(0, box->hbegin, ullMC3Dencode(kx+ix,ky+iy,kz+iz)) == kh_end(box->hbegin) )  // Also, if no corresponding hash
					nblist[idx++] = -1;                                             // is not found in the hash table, also annotate as invalid
				else                                                              // at last, if it has no problems
					nblist[idx++] = ullMC3Dencode(kx+ix,ky+iy,kz+iz);               // annotate the hash of the corresponding cell
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
