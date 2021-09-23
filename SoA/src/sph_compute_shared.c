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

#include "sph_data_types.h"

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

double w_bspline_3d(double r,double h){
  const double A_d = 3./(2.*M_PI*h*h*h);
  double q=0.;
  
  if(r<0||h<=0.)
    exit(10);
  
  q = r/h;
  if(q<=1)
    return A_d*(2./3.-q*q + q*q*q/2.0);
  else if((1.<=q)&&(q<2.))
    return A_d*(1./6.)*(2.-q)*(2.-q)*(2.-q);
  else 
    return 0.;
}

double w_bspline_3d_constant(double h){
  return 3./(2.*M_PI*h*h*h);
}

#pragma omp declare simd
double w_bspline_3d_simd(double q){
  double wq = 0.0;
  double wq1 = (0.6666666666666666 - q*q + 0.5*q*q*q);
  double wq2 = 0.16666666666666666*(2.-q)*(2.-q)*(2.-q); 
  
  if(q<2.)
    wq = wq2;

  if(q<1.)
    wq = wq1;
  
  return wq;
}