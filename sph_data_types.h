#include <stdint.h>

typedef struct size_t2{
    size_t begin;
    size_t end;
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
	size_t id,hash;
	double nu,rho;
} SPHparticle;

typedef struct linkedListBox{
    int Nx,Ny,Nz,N;
    double4 Xmin,Xmax;
} linkedListBox;