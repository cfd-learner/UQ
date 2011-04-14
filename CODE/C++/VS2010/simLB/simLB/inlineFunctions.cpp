
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// INLINE
////////////////////////////////////////////////////////////////////////////////

// equilibrium functions for f

inline double f0(double rho, double u, double v, double T)
{
	//rest particle equilibrium function
	return (rho/4.0)*(4 + 10*pow(T,2) + pow(u,4) - 5*pow(v,2) + pow(v,4) + 10*T*(-1 + pow(u,2) + pow(v,2)) + pow(u,2)*(-5 + 4*pow(v,2)));
}

inline double f1(double rho, double u, double v, double T)
{
	return (rho/6.0)*(-6*pow(T,2) - u*(1 + u)*(-4 + pow(u,2) + 3*pow(v,2)) - T*(-4 + 6*u + 9*pow(u,2) + 3*pow(v,2)));
}

inline double f5(double rho, double u, double v, double T)
{
	return (rho/4.0)*((T + u + pow(u,2))*(T + v + pow(v,2)));
}

inline double f9(double rho, double u, double v, double T)
{
	return (rho/24.0)*(3*pow(T,2) + (-1 + u)*u*(1 + u)*(2 + u) + T*(-1 + 6*u*(1 + u)));
}

// equilibrium functions for h

inline double h0(double K, double T, double u, double v)
{
return (10*(16 + 3*K)*pow(T,3) + 3*T*(8 + 4*K - 40*pow(u,2) - 5*K*pow(u,2) + 20*pow(u,4) 
	+ K*pow(u,4) + (-5*(8 + K) + 4*(15 + K)*pow(u,2))*pow(v,2) + (20 + K)*pow(v,4)) 
	+ 30*pow(T,2)*(-4 + 9*pow(u,2) + 9*pow(v,2) + K*(-1 + pow(u,2) + pow(v,2))) 
	+ 3*(pow(u,2) + pow(v,2))*(4 + pow(u,4) - 5*pow(v,2) + pow(v,4) + pow(u,2)*(-5 + 4*pow(v,2))))/24.0;
}

inline double h1(double K, double T, double u, double v)
{
	return (-2*(16 + 3*K)*pow(T,3) - u*(1 + u)*(pow(u,2) + pow(v,2))*(-4 + pow(u,2) + 3*pow(v,2))
		- T*(u*(-4*(4 + K) - 4*(7 + K)*u + (14 + K)*pow(u,2) + (19 + K)*pow(u,3)) + (-4 + 3*u*(10
		+ K + (14 + K)*u))*pow(v,2) + 3*pow(v,4)) - pow(T,2)*(-16 + 6*u*(6 + 13*u) + 30*pow(v,2)
		+ K*(-4 + 6*u + 9*pow(u,2) + 3*pow(v,2))))/12.0;
}

inline double h5(double K, double T, double u, double v)
{
	return ((16 + 3*K)*pow(T,3) + 3*u*(1 + u)*v*(1 + v)*(pow(u,2) + pow(v,2)) + 3*pow(T,2)*((6 + K)*u 
		+ (9 + K)*pow(u,2) + v*(6 + K + (9 + K)*v)) + 3*T*(pow(u,3) + pow(u,4) + pow(v,3)*(1 + v)
		+ u*v*(6 + K + (9 + K)*v) + pow(u,2)*v*(9 + K + (12 + K)*v)))/24.0;
}

inline double h9(double K, double T, double u, double v)
{
	return ((16 + 3*K)*pow(T,3) + T*u*(-8 + K*(-1 + u)*(1 + u)*(2 + u) + u*(-7 + 2*u*(11 + 8*u))) 
		+ T*(-1 + 6*u*(1 + u))*pow(v,2) + (-1 + u)*u*(1 + u)*(2 + u)*(pow(u,2) + pow(v,2)) 
		+ pow(T,2)*(-4 + 36*u + 51*pow(u,2) + K*(-1 + 6*u*(1 + u)) + 3*pow(v,2)))/48.0;
}