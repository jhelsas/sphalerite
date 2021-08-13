#ifndef SPH_LINKED_LIST_H
#define SPH_LINKED_LIST_H

int safe_free_box(linkedListBox *box);
int compare_SPHparticle(const void *p,const void *q);

int gen_unif_rdn_pos(int64_t N, int seed, SPHparticle *lsph);

int compute_hash_MC3D(int64_t N, SPHparticle *lsph, linkedListBox *box);
int compute_hash_MC2D(int64_t N, SPHparticle *lsph, linkedListBox *box);

int setup_interval_hashtables(int64_t N,SPHparticle *lsph,linkedListBox *box);

int neighbour_hash_3d(int64_t hash,int64_t *nblist,int width, linkedListBox *box);
int neighbour_hash_2d(int64_t hash,int64_t *nblist,int width, linkedListBox *box);

int print_boxes_populations(linkedListBox *box);

int print_neighbour_list_MC3D(int64_t Nmax,unsigned int width,unsigned int stride,SPHparticle *lsph,linkedListBox *box);
int print_neighbour_list_MC2D(int64_t Nmax,unsigned int width,unsigned int stride,SPHparticle *lsph,linkedListBox *box);

int print_neighbour_list_MC3D_lsph_file(int64_t Nmax,unsigned int width,unsigned int stride,SPHparticle *lsph,linkedListBox *box);
int print_neighbour_list_MC2D_lsph_file(int64_t Nmax,unsigned int width,unsigned int stride,SPHparticle *lsph,linkedListBox *box);
int print_neighbour_list_MC3D_lsph_ids_file(int N, SPHparticle *lsph, linkedListBox *box);

#endif