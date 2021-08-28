#ifndef SPH_COMPUTE_H
#define SPH_COMPUTE_H

#pragma omp declare simd
double w_bspline_3d(double r, double h);

double w_bspline_3d_constant(double h);

#pragma omp declare simd
double w_bspline_3d_simd(double q);

#pragma omp declare simd
double dwdq_bspline_3d_simd(double q);

#define w_bspline_3d_LUT(q) w_bspline_3d_LUT_128(q)

#pragma omp declare simd
double w_bspline_3d_LUT_1024(double q);

#pragma omp declare simd
double w_bspline_3d_LUT_128(double q);

#pragma omp declare simd
double w_bspline_3d_LUT_32(double q);

int compute_density_3d(int N, double h, SPHparticle *lsph, linkedListBox *box);

int compute_density_3d_innerOmp(int N, double h, SPHparticle *lsph, linkedListBox *box);

int compute_density_3d_loopswapped(int N, double h, SPHparticle *lsph, linkedListBox *box);

int compute_density_2d(int N, double h, SPHparticle *lsph, linkedListBox *box);

#endif