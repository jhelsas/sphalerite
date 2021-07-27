#include <stdint.h>
#include "../klib/khash.h"

typedef struct size_t2{
    int64_t begin;
    int64_t end;
} size_t2;

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
    int Nx,Ny,Nz,N;
    double4 Xmin,Xmax;
    khash_t(0) *hbegin;
    khash_t(1) *hend ;
} linkedListBox;

