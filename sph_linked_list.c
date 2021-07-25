#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "sph_data_types.h"

int main(){

	int j=0,N=0;
	const gsl_rng_type *T=NULL;
	gsl_rng *r=NULL;
	double S=0.,S2=0.;
	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	gsl_rng_set(r,123123123);

	printf("sizeof size_t : %lu\n",sizeof(size_t));

	N = 100000000;
	S = 0.0;
	S2 = 0.0;
	for(int i=0;i<N;i+=1){
		double val = gsl_rng_uniform(r);
		S += val; S2 += val*val/N;
	}
	printf("mean = %lf , stddev = %lf\n",S/N,sqrt(S2/N - (S*S)/(N*N)));

	gsl_rng_free(r);
	return 0;
}