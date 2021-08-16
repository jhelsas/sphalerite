#ifndef SPH_COMPUTE_H
#define SPH_COMPUTE_H

#pragma omp declare simd
double w_bspline_3d(double r,double h);

#pragma omp declare simd
int compute_density_3d(int N, double h, SPHparticle *lsph, linkedListBox *box);

int compute_density_2d(int N, double h, SPHparticle *lsph, linkedListBox *box);

#endif