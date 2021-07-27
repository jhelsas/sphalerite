
double signum(double z)
{
  if(z>=0) return 1.0;
  else return (-1.0);
}

double w_bspline(int D,double r,double h)
{
  double R,A_d=0.;
  if(h<=0.) exit(10);
  
  if(D==1)
    A_d=1;
  else if(D==2)
    A_d=(15.)/(7.*M_PI);
  else if(D==3)
    A_d=(3.0)/(2.0*M_PI);
  
  R=fabs(r)/h;
  if(R>=2.)
    return 0;
  else if((1.<=R)&&(R<2.))
    return (A_d)*(1./6.)*(2.-R)*(2.-R)*(2.-R)/pow(h,D);
  else
    return ((A_d)*((2./3.)-(R*R) + (R*R*R/2.0)))/pow(h,D);
}

double Dw_bspline(int D,double r,double h)
{
  double R,A_d=0.;
  if(h<=0.) exit(10);
  
  if(D==1)
    A_d=1;
  else if(D==2)
    A_d=(15.)/(7.*M_PI);
  else if(D==3)
    A_d=(3.0)/(2.0*M_PI);
    
  R=fabs(r)/h;
  if(R>=2.)
    return 0;
  else if((1.<=R)&&(R<2.))
    return (signum(r)*(-A_d*(2.0-R)*(2.0-R)))/(2.0*(h*pow(h,D)));
  else
    return (signum(r)*(A_d*(-2.0*R+(3./2.)*R*R)))/(h*pow(h,D));
  
}

double DDw_bspline(int D,double r,double h)
{
  double R,A_d=0.;
  if(h<=0.) exit(10);
  
  if(D==1)
    A_d=1;
  else if(D==2)
    A_d=(15.)/(7.*M_PI);
  else if(D==3)
    A_d=(3.0)/(2.0*M_PI);
    
  R=fabs(r)/h;
  if(R>=2.)
    return 0;
  else if((1.<=R)&&(R<2.))
    return ((A_d)*(2.0-R))/((h*h*pow(h,D)));
  else
    return (A_d*(-2.0+3.0*R))/(h*h*pow(h,D));
  
}

double w_gauss(int D,double r,double h)
{
  double R;
  if(h<=0.0) exit(10);
  R=fabs(r)/h;
  if(R>=2.0)
    return 0.0;
  else
    return (3.0*exp(-(4.5*R*R))/(sqrt(2*M_PI)*h));
}

double Dw_gauss(int D,double r,double h)
{
  double R;
  if(h<=0.) exit(10);
  R=fabs(r)/h;
  if(R>=2.)
    return 0;
  else
    return -signum(r)*((27.0*R*exp(-(4.5*R*R)))/(sqrt(2*M_PI)*h*h));
}

double DDw_gauss(int D,double r,double h)
{
  double R;
  if(h<=0.) exit(10);
  R=fabs(r)/h;
  if(R>=2.)
    return 0;
  else
    return ((27.0*(9.0*R*R-1.0)*exp(-(4.5*R*R)))/(sqrt(2*M_PI)*h*h*h));
}

/************************************************/

double w_sgauss(int D,double r,double h)
{
  double R;
  if(h<=0.0) exit(10);
  R=fabs(r)/h;
  if(R>=2.0)
    return 0.0;
  else
    return ((1.5-R*R)*exp(-R*R))/(sqrt(M_PI)*h);
}

double Dw_sgauss(int D,double r,double h)
{
  double R;
  if(h<=0.) exit(10);
  R=fabs(r)/h;
  if(R>=2.)
    return 0;
  else
    return -signum(r)*(((2.0*(R*R*(R-1)))*exp(-R*R))/(sqrt(M_PI)*h*h));
}

double DDw_sgauss(int D,double r,double h)
{
  double R;
  if(h<=0.) exit(10);
  R=fabs(r)/h;
  if(R>=2.)
    return 0;
  else
    return ((2.0*(3.0*R*R-2.0*R+2.0*R*R*R*R-R*R*R)*exp(-R*R))/(sqrt(M_PI)*h*h*h));
}