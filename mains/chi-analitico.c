#include <stdio.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>

#define C_landau 0.228136013 /* (15÷(128×π^2))^(1/3)*/
#define T0 0.645266087 /*(4.0*C_landau*(sqrt(1.0/2.0)))*/
#define l 1.0
     
double f (double zetap, void * params) {
  double alpha = *(double *) params;
  double arg;
  arg=(alpha*alpha)/3.0-zetap*zetap;
  double f = exp(2.0*zetap)*gsl_sf_bessel_J0(sqrt(arg));
  
  return f;
}
     
int main (void)
{
  double chi, err;
  double alpha ,alpha_min,alpha_max,zeta,zeta_min,zeta_max,da,dz;
  FILE *dadosout;

  gsl_integration_workspace * w 
  = gsl_integration_workspace_alloc (1000);
       
  
  alpha_min=-20.0; alpha_max=20.0;
  da=0.05; dz=0.05;
  zeta_min=-10.0; zeta_max=0.0;

  gsl_function F;
  F.function = &f;
  F.params = &alpha;
    
  dadosout=fopen("chi.csv","w");
  
  fprintf(dadosout,"alpha, T0*exp(zeta), T0*exp(zeta)*l*sqrt(3.0)*chi\n");
  for(alpha=alpha_min;alpha<=alpha_max;alpha+=da){
    for(zeta=zeta_min;zeta<=zeta_max;zeta+=dz){
      if(zeta<(-fabs(alpha)/sqrt(3.0)))
        fprintf(dadosout,"%f %f %f\n",alpha,T0*exp(zeta),0.0);
      else{
        gsl_integration_qags (&F, -zeta, fabs(alpha)/sqrt(3.0), 0, 1e-7, 1000,w, &chi, &err);
        fprintf(dadosout,"%f, %f, %f\n",alpha,T0*exp(zeta),T0*exp(zeta)*l*sqrt(3.0)*chi);
      }
    }
  }
     
  fclose(dadosout);dadosout=NULL;
  gsl_integration_workspace_free (w);
     
  return 0;
}
