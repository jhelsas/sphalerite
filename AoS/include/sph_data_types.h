#ifndef SPH_DATA_TYPES_H
#define SPH_DATA_TYPES_H

#include <stdint.h>
#include "klib/khash.h"

typedef struct double4 {
    double x;
    double y;
    double z;
    double t;
} double4;

typedef struct SPHparticle
{
    double4 r,u,F;
	int64_t id,hash;
	double nu,rho;
} SPHparticle;

KHASH_MAP_INIT_INT64(0, int64_t)
KHASH_MAP_INIT_INT64(1, int64_t)

typedef struct linkedListBox{
    int Nx,Ny,Nz,N,width;
    double Xmin,Ymin,Zmin;
    double Xmax,Ymax,Zmax;
    khash_t(0) *hbegin;
    khash_t(1) *hend ;
} linkedListBox;

#endif