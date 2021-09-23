#ifndef SPH_COMPUTE_SHARED_H
#define SPH_COMPUTE_SHARED_H

double w_bspline_3d(double r,double h);

double w_bspline_3d_constant(double h);

#pragma omp declare simd
double w_bspline_3d_simd(double q);

#endif
