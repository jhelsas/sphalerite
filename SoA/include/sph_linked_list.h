/*
 * SPDX-License-Identifier:  BSD-3-Clause
 * 
 * sph_linked_list.h : 
 *     Header containing the declarions for several
 *     cell linked list operations, including hash
 *     calculations, setup hash tables and neighbour finding.
 *
 * (C) Copyright 2021 José Hugo Elsas
 * Author: José Hugo Elsas <jhelsas@gmail.com>
 *
 */

#ifndef SPH_LINKED_LIST_H
#define SPH_LINKED_LIST_H

#include "sph_data_types.h"

int safe_free_box(linkedListBox *box);
int compare_int64_t(const void *p,const void *q);

int SPHparticle_SoA_malloc(int N,SPHparticle **lsph);
int SPHparticleSOA_safe_free(int N,SPHparticle **lsph);

int gen_unif_rdn_pos(int64_t N, int seed, SPHparticle *lsph);
int gen_unif_rdn_pos_box(int64_t N, int seed, linkedListBox *box,SPHparticle *lsph);

int compute_hash_MC3D(int64_t N, SPHparticle *lsph, linkedListBox *box);

int setup_interval_hashtables(int64_t N,SPHparticle *lsph,linkedListBox *box);
int reorder_lsph_SoA(int N, SPHparticle *lsph, void *swap_arr);

int neighbour_hash_3d(int64_t hash,int64_t *nblist,int width, linkedListBox *box);

int print_time_stats(const char *prefix,  bool is_cll,int64_t N, double h, 
										 long int seed, int runs, SPHparticle *lsph, linkedListBox *box,double *times);

int print_sph_particles_density(const char *prefix,  bool is_cll,int64_t N, double h, 
																long int seed, int runs, SPHparticle *lsph, linkedListBox *box);

#endif