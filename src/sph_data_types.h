#include <stdint.h>
#include "../klib/khash.h"

typedef struct SPHparticle{
  int64_t *id,*hash;
  double *nu,*rho;
  double *x,*y,*z;
  double *ux,*uy,*uz;
  double *Fx,*Fy,*Fz;
}

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

