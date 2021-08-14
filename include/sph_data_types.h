#ifndef SPH_DATA_TYPES_H
#define SPH_DATA_TYPES_H

#include <stdint.h>
#include "klib/khash.h"

typedef struct double4{
  double x;
  double y;
  double z;
  double t;
} double4;

typedef struct SPHparticle{
  int64_t *id,*idx,*hash;
  double *nu,*rho;
  double *x,*y,*z;
  double *ux,*uy,*uz;
  double *Fx,*Fy,*Fz;
} SPHparticle;

KHASH_MAP_INIT_INT64(0, int64_t)
KHASH_MAP_INIT_INT64(1, int64_t)

typedef struct linkedListBox{
    int Nx,Ny,Nz,N,width;
    double h;
    double4 Xmin,Xmax;
    double (*w)(double,double);
    double (*dwdq)(double,double);
    khash_t(0) *hbegin;
    khash_t(1) *hend ;
} linkedListBox;

#endif