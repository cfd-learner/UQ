#include "LBM_2D_NS_cuKernel.h"

__constant__ unsigned int mirrorNS[13] = {0,1,4,3,2,8,7,6,5,9,12,11,10};
__constant__ unsigned int mirrorEW[13] = {0,3,2,1,4,6,5,8,7,11,10,9,12};

__constant__ int xx[4] = {-1,1,0,0};
__constant__ int yy[4] = {0,0,-1,1};
__constant__ unsigned	int ii[4][4] = {{1,5,8,9},{3,6,7,11},{2,5,6,10},{4,7,8,12}};
__constant__ unsigned	int inv[4][4] = {{3,6,7,11},{1,5,8,9},{4,7,8,12},{2,5,6,10}};

__device__ int errEq = 0;

////////////////////////////////////////////////////////////////////////////////
// INLINE
////////////////////////////////////////////////////////////////////////////////

// equilibrium functions for f

__device__ inline double f0(double rho, double u, double v, double T)
{
	//rest particle equilibrium function
	return (rho/4.0)*(4 + 10*pow(T,2) + pow(u,4) - 5*pow(v,2) + pow(v,4) + 10*T*(-1 + pow(u,2) + pow(v,2)) + pow(u,2)*(-5 + 4*pow(v,2)));
}

__device__ inline double f1(double rho, double u, double v, double T)
{
	return (rho/6.0)*(-6*pow(T,2) - u*(1 + u)*(-4 + pow(u,2) + 3*pow(v,2)) - T*(-4 + 6*u + 9*pow(u,2) + 3*pow(v,2)));
}

__device__ inline double f5(double rho, double u, double v, double T)
{
	return (rho/4.0)*((T + u + pow(u,2))*(T + v + pow(v,2)));
}

__device__ inline double f9(double rho, double u, double v, double T)
{
	return (rho/24.0)*(3*pow(T,2) + (-1 + u)*u*(1 + u)*(2 + u) + T*(-1 + 6*u*(1 + u)));
}

// equilibrium functions for h

__device__ inline double h0(double K, double T, double u, double v)
{
return (10*(16 + 3*K)*pow(T,3) + 3*T*(8 + 4*K - 40*pow(u,2) - 5*K*pow(u,2) + 20*pow(u,4) 
	+ K*pow(u,4) + (-5*(8 + K) + 4*(15 + K)*pow(u,2))*pow(v,2) + (20 + K)*pow(v,4)) 
	+ 30*pow(T,2)*(-4 + 9*pow(u,2) + 9*pow(v,2) + K*(-1 + pow(u,2) + pow(v,2))) 
	+ 3*(pow(u,2) + pow(v,2))*(4 + pow(u,4) - 5*pow(v,2) + pow(v,4) + pow(u,2)*(-5 + 4*pow(v,2))))/24.0;
}

__device__ inline double h1(double K, double T, double u, double v)
{
	return (-2*(16 + 3*K)*pow(T,3) - u*(1 + u)*(pow(u,2) + pow(v,2))*(-4 + pow(u,2) + 3*pow(v,2))
		- T*(u*(-4*(4 + K) - 4*(7 + K)*u + (14 + K)*pow(u,2) + (19 + K)*pow(u,3)) + (-4 + 3*u*(10
		+ K + (14 + K)*u))*pow(v,2) + 3*pow(v,4)) - pow(T,2)*(-16 + 6*u*(6 + 13*u) + 30*pow(v,2)
		+ K*(-4 + 6*u + 9*pow(u,2) + 3*pow(v,2))))/12.0;
}

__device__ inline double h5(double K, double T, double u, double v)
{
	return ((16 + 3*K)*pow(T,3) + 3*u*(1 + u)*v*(1 + v)*(pow(u,2) + pow(v,2)) + 3*pow(T,2)*((6 + K)*u 
		+ (9 + K)*pow(u,2) + v*(6 + K + (9 + K)*v)) + 3*T*(pow(u,3) + pow(u,4) + pow(v,3)*(1 + v)
		+ u*v*(6 + K + (9 + K)*v) + pow(u,2)*v*(9 + K + (12 + K)*v)))/24.0;
}

__device__ inline double h9(double K, double T, double u, double v)
{
	return ((16 + 3*K)*pow(T,3) + T*u*(-8 + K*(-1 + u)*(1 + u)*(2 + u) + u*(-7 + 2*u*(11 + 8*u))) 
		+ T*(-1 + 6*u*(1 + u))*pow(v,2) + (-1 + u)*u*(1 + u)*(2 + u)*(pow(u,2) + pow(v,2)) 
		+ pow(T,2)*(-4 + 36*u + 51*pow(u,2) + K*(-1 + 6*u*(1 + u)) + 3*pow(v,2)))/48.0;
}

////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////

__device__ void
cuEq2D(double rho, double u_, double v_, double T_, double Tc, double R, double K, double* eqf, double* eqh)
{
	//returns the equilibrium values for each velocity vector given the current
	// macroscopic values and the corresponding reference quantities
	
	double sRTc = sqrt(R*Tc);
	double u = u_/sRTc;
	double v = v_/sRTc;
	double T = T_/Tc;

	eqf[0] = f0(rho,u,v,T);
	eqf[1] = f1(rho,u,v,T);
	eqf[2] = f1(rho,v,u,T);
	eqf[3] = f1(rho,-u,v,T);
	eqf[4] = f1(rho,-v,u,T);
	eqf[5] = f5(rho,u,v,T);
	eqf[6] = f5(rho,-u,v,T);
	eqf[7] = f5(rho,-u,-v,T);
	eqf[8] = f5(rho,u,-v,T);
	eqf[9] = f9(rho,u,v,T);
	eqf[10] = f9(rho,v,u,T);
	eqf[11] = f9(rho,-u,v,T);
	eqf[12] = f9(rho,-v,u,T);

	double rRTc = rho*R*Tc;

	eqh[0] = rRTc*h0(K,T,u,v);
	eqh[1] = rRTc*h1(K,T,u,v);
	eqh[2] = rRTc*h1(K,T,v,u);
	eqh[3] = rRTc*h1(K,T,-u,v);
	eqh[4] = rRTc*h1(K,T,-v,u);
	eqh[5] = rRTc*h5(K,T,u,v);
	eqh[6] = rRTc*h5(K,T,-u,v);
	eqh[7] = rRTc*h5(K,T,-u,-v);
	eqh[8] = rRTc*h5(K,T,u,-v);
	eqh[9] = rRTc*h9(K,T,u,v);
	eqh[10] =rRTc*h9(K,T,v,u);
	eqh[11] =rRTc*h9(K,T,-u,v);
	eqh[12] =rRTc*h9(K,T,-v,u);

	//check
	double sum_f = 0;
	double sum_h = 0;
	for (int i = 0; i < 13; i++)
	{
		sum_f += eqf[i];
		sum_h += eqh[i];
	}
	double U = sqrt(u_*u_ + v_*v_);
	double E = (U*U + (K+2.0)*R*T_)/2.0;

	double diff1 = sum_h/rho - E;
	double diff2 = sum_f - rho;

	if (diff1 > 1 || diff2 > 1)
	{
		errEq = 1;
	}
}

__device__ int
cuSign(double a)
{
	//returns: a = 0 -> 0, a = neg -> -1, a = pos -> 1
	if (a > 0)
	{
		return 1;
	}
	else if (a < 0)
	{
		return -1;
	}
	else
	{
		return 0;
	}
}

__device__ void
cuMirIndex(int i, int mir_x, int mir_y, int& i_x, int& i_y)
{
	// gives the mirrored index about the axes defined by inputs mir_x, and mir_y

	// initialise with no mirror
	i_x = i;
	i_y = i;

	// mirror about x
	if (mir_x == 1)	
	{
		i_x = mirrorNS[i];
	}
	
	// mirror about y
	if(mir_y == 1)
	{
		i_y = mirrorEW[i];
	}

}

__device__
int cuIndex(int i, int pm, int d, simData* sim, int& mir)
{
	int Ny = (*sim).Ny;
	int Nx = (*sim).Nx;

	// d = 0 -> x
	// d = 1 -> y
	i = i + pm;
	mir = 0;

	if (d == 0)
	{
		if (i < 0)
		{
			if ((*sim).periodicX > 0)
			{
				i = Nx + i;
			}
			else if ((*sim).mirrorW > 0)
			{
				i = -i - 1;
				mir = 1;
			}
			else
			{
				i = 0; //zeroth order extrapolation
			}
		}
		else if (i > Nx - 1)
		{
			if ((*sim).periodicX > 0)
			{
				i = i - Nx;
			}
			else if ((*sim).mirrorE > 0)
			{
				i = 2*Nx - i - 1;
				mir = 1;
			}
			else
			{
				i = Nx - 1; //zeroth order extrapolation
			}
		}
	}

	else if (d == 1)
	{
		if (i < 0)
		{
			if ((*sim).periodicY > 0)
			{
				i = Ny + i;
			}
			else if ((*sim).mirrorS > 0)
			{
				i = -i - 1;
				mir = 1;
			}
			else
			{
				i = 0; //zeroth order extrapolation
			}
		}
		else if (i > Ny - 1)
		{
			if ((*sim).periodicY > 0)
			{
				i = i - Ny;
			}
			else if ((*sim).mirrorN > 0)
			{
				i = 2*Ny - i - 1;
				mir = 1;
			}
			else
			{
				i = Ny - 1; //zeroth order extrapolation
			}
		}
	}

	return i;
}

__device__ void
cuStencil(double* f_d, double* Sx, double* Sy, int n, int i, simData* sim, int* solid_d)
{
	// create a stencil of given length for the Flux method specified

	if (i == 0)
	{
		return;
	}

	int Ny = (*sim).Ny;

	// thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//global index
	int ix = blockIdx.x*blockDim.x + tx;
	int iy = blockIdx.y*blockDim.y + ty;

	// velocity vector, in integer values for indexing
	int ex = cuSign((*sim).e[i].x);
	int ey = cuSign((*sim).e[i].y);

	int low = (n - 1) / 2;	// max/min num of stencil from centre

	// calculate stencil along x and y
	int jx1 = ix - ex*low;	//start of stencil index
	int jy1 = iy - ey*low;

	int jx, jy;
	int mir_x, mir_y;
	int i_x, i_y;

	int sld_x = 0;
	int sld_y = 0;
	int sldx[5];
	int sldy[5];

	int sld;

	for (int j = 0; j < n; j++)	// put together stencil along axis lines, start on low side, go to hi side: [j-low <-> j+low]
	{
		jx = cuIndex(jx1,ex*j,0,sim,mir_y);
		jy = cuIndex(jy1,ey*j,1,sim,mir_x);

		cuMirIndex(i,mir_x,mir_y,i_x,i_y);	// mirror indexes if required

		sld = GSLD(ix,iy);

		// x
		Sx[j] = GF(jx,iy,i_y);

		if (sld >= 0)
		{
			sldx[j] = 1;
			sld_x = 1;
		}
		else
		{
			sldx[j] = 0;
		}

		// y
		Sy[j] = GF(ix,jy,i_x);

		if (sld >= 0)
		{
			sldy[j] = 1;
			sld_y = 1;
		}
		else
		{
			sldy[j] = 0;
		}
	}

	if (sld_x == 1)
	{
		int diff;	// difference between indexes of sldx
		int x_;		// location of edge solid node, outside coords
		int j_;		// location of edge solid node, stencil coords
		
		// find where edge solid node is
		for (int j = 0; j < n - 1; j++)
		{
			diff = sldx[j] - sldx[j+1];
			if (diff > 0)
			{
				j_ = j;
				x_ = cuIndex(jx1,ex*j,0,sim,mir_y);
				break;
			}
			else if (diff < 0)
			{
				j_ = j + 1;
				x_ = cuIndex(jx1,ex*(j + 1),0,sim,mir_y);
				break;
			}
		}

		// update solid part of stencil with linear extrapolation of actual stencil

		int xm1 = cuIndex(x_,ex*diff,0,sim,mir_y);		// closest fluid node
		int xm2 = cuIndex(x_,ex*2*diff,0,sim,mir_y);		// next closest fluid node

		Sx[j_] = 2*GF(xm1,iy,i) - GF(xm2,iy,i);

		int j__ = j_ - diff;

		if (j__ >= 0 && j__ < n) // update next one along if stencil is long enough
		{
			Sx[j__] = Sx[j_];
		}
	}

	if (sld_y == 1)
	{
		int diff;	// difference between indexes of sldx
		int y_;		// location of edge solid node, outside coords
		int j_;		// location of edge solid node, stencil coords
		
		// find where edge solid node is
		for (int j = 0; j < n - 1; j++)
		{
			diff = sldy[j] - sldy[j+1];
			if (diff > 0)
			{
				j_ = j;
				y_ = cuIndex(jy1,ey*j,1,sim,mir_x);
				break;
			}
			else if (diff < 0)
			{
				j_ = j + 1;
				y_ = cuIndex(jy1,ey*(j + 1),1,sim,mir_x);
				break;
			}
		}

		// update solid part of stencil with linear extrapolation of actual stencil

		int ym1 = cuIndex(y_,ey*diff,0,sim,mir_x);		// closest fluid node
		int ym2 = cuIndex(y_,ey*2*diff,0,sim,mir_x);		// next closest fluid node

		Sy[j_] = 2*GF(ix,ym1,i) - GF(ix,ym2,i);

		int j__ = j_ - diff;

		if (j__ >= 0 && j__ < n)	// update next one along if stencil is long enough
		{
			Sy[j__] = Sy[j_];
		}
	}
}

__device__ double
cuMinmod(double a, double b)
{
	// calculate minmod function

	double out;

	if ((abs(a) < abs(b)) && (a*b > 0))
		out = a;
	else if ((abs(b) < abs(a)) && (a*b > 0))
		out = b;
	else if (a == b)
		out = a;
	else if (a*b <= 0)
		out = 0;
	return out;
}

__device__ double
cuNND(double* S, double e)
{
	// calculate the flux by the NND method (2nd order accurate, CFL_max = 2/3)

	int n = 1;

	double Fp_I, Fp_Ip1, Fp_Im1;

	double vP = abs(e);

	if (vP == 0)
	{
		return 0;
	}

	Fp_I   = vP*S[n];
	Fp_Ip1 = vP*S[n+1];
	Fp_Im1 = vP*S[n-1];

	double dFp_Ip12, dFp_Im12;

	dFp_Ip12 = Fp_Ip1 - Fp_I;
	dFp_Im12 = Fp_I - Fp_Im1;

	double F_Ip12;

	F_Ip12 = Fp_I + 0.5*cuMinmod(dFp_Ip12,dFp_Im12);

	return F_Ip12;
}

__device__ double
cuWENO5(double* S, double e)
{
	// calculate the flux term of the WENO5 scheme for one flow direction

	double epsilon = (double) 1e-6;
	double posFlow = abs(e);
	double F_Ip12;

	if (posFlow == 0)
	{
		return 0;
	}
	else
	{
		double Sp[5];
		for (int i = 0; i < 5; i++)
		{
			Sp[i] = S[i]*posFlow;
		}

		int n = 2;

		double B0p, B1p, B2p, alpha0p, alpha1p, alpha2p;
		double omega0p, omega1p, omega2p, f0p, f1p, f2p;

		B0p = (13.0/12.0)*pow(Sp[n-2] - 2.0*Sp[n-1] + Sp[n],2) + (1.0/4.0)*pow(Sp[n-2] - 4.0*Sp[n-1] + 3.0*Sp[n],2);
		B1p = (13.0/12.0)*pow(Sp[n-1] - 2.0*Sp[n] + Sp[n+1],2) + (1.0/4.0)*pow(Sp[n-1] - Sp[n+1],2);
		B2p = (13.0/12.0)*pow(Sp[n] - 2.0*Sp[n+1] + Sp[n+2],2) + (1.0/4.0)*pow(3.0*Sp[n] - 4.0*Sp[n+1] + Sp[n+2],2);

		alpha0p = (1.0/10.0)*pow(1.0/(epsilon + B0p),2);
		alpha1p = (6.0/10.0)*pow(1.0/(epsilon + B1p),2);
		alpha2p = (3.0/10.0)*pow(1.0/(epsilon + B2p),2);

		omega0p = alpha0p/(alpha0p + alpha1p + alpha2p);
		omega1p = alpha1p/(alpha0p + alpha1p + alpha2p);
		omega2p = alpha2p/(alpha0p + alpha1p + alpha2p);

		f0p = (2.0/6.0)*Sp[n-2] - (7.0/6.0)*Sp[n-1] + (11.0/6.0)*Sp[n];
		f1p = -(1.0/6.0)*Sp[n-1] + (5.0/6.0)*Sp[n] + (2.0/6.0)*Sp[n+1];
		f2p = (2.0/6.0)*Sp[n] + (5.0/6.0)*Sp[n+1] - (1.0/6.0)*Sp[n+2];

		F_Ip12 = omega0p*f0p + omega1p*f1p + omega2p*f2p;
	}

	return F_Ip12;
}

__device__ void
cuFLUX(double* f, int i, simData* sim, int* solid_d, double2& flux)
{
	// choose flux method

	double ex, ey;

	ex = (*sim).e[i].x;
	ey = (*sim).e[i].y;

	if ((*sim).FMETHOD == 0)
	{
		//NND
		double Sx[3];
		double Sy[3];

		cuStencil(f, Sx, Sy, 3, i, sim, solid_d);

		flux.x = cuNND(Sx,ex);
		flux.y = cuNND(Sy,ey);
	}
	else if ((*sim).FMETHOD == 1)
	{
		// WENO5
		double Sx[5];
		double Sy[5];

		cuStencil(f, Sx, Sy, 5, i, sim, solid_d);

		flux.x = cuWENO5(Sx, ex);
		flux.y = cuWENO5(Sy, ey);
	}
}

__device__ double
cuCombineFLUX(double2* fluxOut, int i, simData* sim)
{
	// calculates the combined flux given all the fluxes out of each node

	int Ny = (*sim).Ny;

	// thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//global index
	int ix = blockIdx.x*blockDim.x + tx;
	int iy = blockIdx.y*blockDim.y + ty;

	// velocity vector, in integer values for indexing
	int ex = cuSign((*sim).e[i].x);
	int ey = cuSign((*sim).e[i].y);

	int mir_x, mir_y;

	int x_ = cuIndex(ix,-ex,0,sim,mir_y);
	int y_ = cuIndex(iy,-ey,1,sim,mir_x);

	if (i == 0)
	{
		return 0;
	}
	else
	{
		double dt = (*sim).dt;
		double dx = (*sim).dx;
		double dy = (*sim).dy;

		int i_x, i_y;

		cuMirIndex(i,mir_x,mir_y,i_x,i_y);	// mirror indexes if required


		double flux_out = dt*(GFOUTx(ix,iy,i)/dx + GFOUTy(ix,iy,i)/dy);

		double flux_in = dt*(GFOUTx(x_,iy,i_y)/dx + GFOUTy(ix,y_,i_x)/dy);

		return flux_out - flux_in;
	}
}

__device__ void
cuPosFlux(double* f, double* h, int i, int* solid_d, simData* sim, double2* f_flux, double2* h_flux)
{
	// function to calculate all positive fluxes along axis (x & y) lines for fluid nodes
	// store fluxes in global memory

	// thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//global index
	int ix = blockIdx.x*blockDim.x + tx;
	int iy = blockIdx.y*blockDim.y + ty;

	int Ny = (*sim).Ny;

	// temporary flux storage

	double2 flux_f, flux_h;

	int sld = GSLD(ix,iy);

	if (sld < 0)	// only perform fluxes on fluid and permanent nodes
	{
		cuFLUX(f,i,sim,solid_d,flux_f);
		cuFLUX(h,i,sim,solid_d,flux_h);

		// store fluxes in global memory
		GFLUXFx(ix,iy,i) = flux_f.x;
		GFLUXFy(ix,iy,i) = flux_f.y;

		GFLUXHx(ix,iy,i) = flux_h.x;
		GFLUXHy(ix,iy,i) = flux_h.y;
	}
}

__global__ void
MACRO_PROPERTIES(double* f_d, double* h_d, simData* sim, double* rho_d, double2* u_d, double* T_d, double* p_d)
{

	// Block index

	//global index
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	int Ny = (*sim).Ny;

	double rho, T, p, f;
	double2 u;
	double rho_ux, rho_uy, sum_h;

	rho = 0;
	rho_ux = 0;
	rho_uy = 0;
	sum_h = 0;

	//#pragma unroll 13
	for (int i = 0; i < 13; i++)
	{
		f = GF(ix,iy,i);

		rho += f;
		rho_ux += f*(*sim).e[i].x;
		rho_uy += f*(*sim).e[i].y;
		sum_h += GH(ix,iy,i);
	}

	u.x = rho_ux / rho;
	u.y = rho_uy / rho;

	double usq = sqrt(u.x*u.x + u.y*u.y);

	usq = usq*usq;

	T = 2.0*(sum_h/rho - usq/2.0)/((*sim).b*(*sim).R);

	p = rho*(*sim).R*T;

	// save macro-properties to device memory
	GRHO(ix,iy) = rho;
	GU(ix,iy) = u;
	GT(ix,iy) = T;
	GP(ix,iy) = p;
}

__global__ void
	GLOBAL_FLUXES(double* f, double* h, simData* sim, int* solid, double2* fluxf, double2* fluxh)
{
	// calculate all outgoing fluxes for the given distribution functions and save to global memory.

	__syncthreads();

	for (int i = 0; i < 13; i++)
	{
		// calc fluxes, save to global
		cuPosFlux(f, h, i, solid, sim, fluxf, fluxh);
	}
}

__global__ void 
WALL_FLUXES(double2* f_flux, double2* h_flux, int* solid_d, simData* sim, double* TW, double2* uW)
{
	// calculate fluxes into solid nodes and set fluxes out of solid to cancel them out

	// thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//global index
	int ix = blockIdx.x*blockDim.x + tx;
	int iy = blockIdx.y*blockDim.y + ty;

	int Ny = (*sim).Ny;

	int xm1, ym1, i, inv_i;
	int mir_x, mir_y;

	double dx = (*sim).dx;
	double dy = (*sim).dy;

	double flux_in, flux_out, alpha;

	double feqW[13];
	double heqW[13];

	double ux, uy;

	int inx,iny;

	int sld = GSLD(ix,iy);

	if ( sld >= 0)	//wall nodes
	{
		flux_in = 0;
		flux_out = 0;

		cuEq2D(1.0, uW[sld].x, uW[sld].y, TW[sld], (*sim).Tc, (*sim).R, (*sim).K, feqW, heqW);

		// check surrounds for fluid node
		for (int j = 0; j < 4; j++)
		{
			xm1 = cuIndex(ix,xx[j],0,sim,mir_y);
			ym1 = cuIndex(iy,yy[j],1,sim,mir_x);

			int sld2 = GSLD(xm1,ym1);

			if (sld2 == -1) // if fluid
			{
				for (int k = 0; k < 4; k++)
				{
					i = ii[j][k];	// index of velocities into solid
					inv_i = inv[j][k];	// inverted velocities, out of solid

					// switches for turning off velocities that don't impinge on solid, or leave solid, through the cell pointed to by xx & yy
					inx = abs(xx[j]);	
					iny = abs(yy[j]);

					flux_in += inx*GFLUXFx(xm1,ym1,i)/dx + iny*GFLUXFy(xm1,ym1,i)/dy;		// flux into solid

					// absolute value of velocities
					ux = abs((*sim).e[inv_i].x);	
					uy = abs((*sim).e[inv_i].y);

					flux_out += inx*feqW[inv_i]*ux/dx + iny*feqW[inv_i]*uy/dy;				// flux out of solid, back along inverse velocity
				}
			}
		}

		alpha = flux_in/flux_out;	// correction factor to equalise flux in to flux out

		// load required fluxes into flux array
		for (int i = 0; i < 13; i++)
		{
			ux = abs((*sim).e[i].x);	
			uy = abs((*sim).e[i].y);

			 GFLUXFx(ix,iy,i) = alpha*ux*feqW[i];
			 GFLUXFy(ix,iy,i) = alpha*uy*feqW[i];

			 GFLUXHx(ix,iy,i) = alpha*ux*heqW[i];
			 GFLUXHy(ix,iy,i) = alpha*uy*heqW[i];
		}
	}
}

__global__ void
	RK1_STEP1_KERNEL(double* f_d, double* f1_d, double* h_d, double* h1_d, int* solid_d, double* TW, double2* uW, double* rho_d, double2* u_d, double* T_d, simData* sim)
{
	//perform RK1 stepping

	//global index
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	int Ny = (*sim).Ny;

	double rho = GRHO(ix,iy);
	double2 u = GU(ix,iy);
	double T = GT(ix,iy);

	double feq[13];
	double heq[13];

	int sld;

	sld = GSLD(ix,iy);

	// --- RK STEP ONE----
	// calculate the first stage updated distribution functions 

	if (sld > -1 || sld == -2)	// check if node is solid or permanent, if it is, just propogate values
	{
		for (int i = 0; i < 13; i++)
		{
			GF1(ix,iy,i) = GF(ix,iy,i);
			GH1(ix,iy,i) = GH(ix,iy,i);
		}
	}
	else
	{
		// calculate relaxation times from macroscopic properties
		double tauf = (*sim).mu/(rho*(*sim).R*T);
		double tauh = tauf/(*sim).Pr;
		double tauhf = (tauh*tauf)/(tauf - tauh);

		double Tc = (*sim).Tc;
		double R = (*sim).R;
		double K = (*sim).K;

		cuEq2D(rho, u.x, u.y, T, Tc, R, K, feq, heq);

		double f1_, edotu;

		for (int i = 0; i < 13; i++)
		{
			// temp variables
			f1_ = (GF(ix,iy,i) + ((*sim).dt/tauf)*feq[i])/(1.0 + (*sim).dt/tauf);	//calc to temp variable first, for use later

			GF1(ix,iy,i) = f1_;		// save to global

			// temp variables
			edotu = (*sim).e[i].x*u.x + (*sim).e[i].y*u.y;

			GH1(ix,iy,i) = (GH(ix,iy,i) - (*sim).dt*edotu*((feq[i] - f1_)/tauhf) + ((*sim).dt/tauh)*heq[i])/(1.0 + (*sim).dt/tauh);
		}
	}
}

__global__ void
	RK1_COMBINE_KERNEL(double* f_d, double* f1_d, double2* fluxf1_d, double* h_d, double* h1_d, double2* fluxh1_d, int* solid_d, double* TW, double2* uW, double* rho_d, double2* u_d, double* T_d, simData* sim)
{
	//combine step of RK1

	//global index
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	int Ny = (*sim).Ny;

	// macroscopic properties
	double rho = GRHO(ix,iy);
	double2 u = GU(ix,iy);
	double T = GT(ix,iy);

	// calculate relaxation times from macroscopic properties
	double tauf = (*sim).mu/(rho*(*sim).R*T);
	double tauh = tauf/(*sim).Pr;
	double tauhf = (tauh*tauf)/(tauf - tauh);

	double feq[13];
	double heq[13];

	double Tc = (*sim).Tc;
	double R = (*sim).R;
	double K = (*sim).K;

	cuEq2D(rho, u.x, u.y, T, Tc, R, K, feq, heq);

	double flux_f, flux_h, edotu;

	int sld = GSLD(ix,iy);

	for (int i = 0; i < 13; i++)
	{

		// --- RK COMBINATION ----
		// calculate the updated distribution functions
		if (sld == -1)
		{
			flux_f = cuCombineFLUX(fluxf1_d, i, sim);

			GF(ix,iy,i) = GF(ix,iy,i) - flux_f + ((*sim).dt/tauf)*(feq[i] - GF1(ix,iy,i));

			flux_h = cuCombineFLUX(fluxh1_d, i, sim);

			edotu = (*sim).e[i].x*u.x + (*sim).e[i].y*u.y;

			GH(ix,iy,i) = GH(ix,iy,i) - flux_h + ((*sim).dt/tauh)*(heq[i] - GH1(ix,iy,i)) - (((*sim).dt*edotu)/tauhf)*(feq[i] - GF1(ix,iy,i));
		}
	}
}

__global__ void
	RK3_STEP1_KERNEL(double* f_d, double* f1_d, double* h_d, double* h1_d, int* solid_d, double* TW, double2* uW, double* rho_d, double2* u_d, double* T_d, simData* sim)
{
	//perform RK3 stepping

	//global index
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	int Ny = (*sim).Ny;

	double rho = GRHO(ix,iy);
	double2 u = GU(ix,iy);
	double T = GT(ix,iy);

	double feq[13];
	double heq[13];

	int sld = GSLD(ix,iy);

	// --- RK STEP ONE----
	// calculate the first stage updated distribution functions 

	if (sld > -1 || sld == -2)
	{
		for (int i = 0; i < 13; i++)
		{
			GF1(ix,iy,i) = GF(ix,iy,i);
			GH1(ix,iy,i) = GH(ix,iy,i);
		}
	}
	else
	{
		// calculate relaxation times from macroscopic properties
		double tauf = (*sim).mu/(rho*(*sim).R*T);
		double tauh = tauf/(*sim).Pr;
		double tauhf = (tauh*tauf)/(tauf - tauh);

		double Tc = (*sim).Tc;
		double R = (*sim).R;
		double K = (*sim).K;

		cuEq2D(rho, u.x, u.y, T, Tc, R, K, feq, heq);

		double f1_, edotu;

		for (int i = 0; i < 13; i++)
		{
			// temp variables
			f1_ = (GF(ix,iy,i) + ((*sim).dt/(2*tauf))*feq[i])/(1.0 + (*sim).dt/(2*tauf));	//calc to temp variable first, for use later

			GF1(ix,iy,i) = f1_;		// save to global

			// temp variables
			edotu = (*sim).e[i].x*u.x + (*sim).e[i].y*u.y;

			GH1(ix,iy,i) = (GH(ix,iy,i) - ((*sim).dt/2.0)*edotu*((feq[i] - f1_)/tauhf) + ((*sim).dt/(2.0*tauh))*heq[i])/(1.0 + (*sim).dt/(2.0*tauh));
		}
	}
}

__global__ void
	RK3_STEP2_KERNEL(double* f_d, double* f1_d, double* f2_d, double* h_d, double* h1_d, double* h2_d, int* solid_d, double* TW, double2* uW, double* rho_d, double2* u_d, double* T_d, simData* sim)
{
	//perform RK3 stepping

	// NOTE: as all equilibrium df are calulated from the same data, feq1 = feq2 etc, this allows for simplification of equations

	//global index
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	int Ny = (*sim).Ny;

	double rho = GRHO(ix,iy);
	double2 u = GU(ix,iy);
	double T = GT(ix,iy);

	double feq[13];
	double heq[13];

	int sld = GSLD(ix,iy);

	// --- RK STEP TWO----
	// calculate the first stage updated distribution functions 

	if (sld > -1 || sld == -2)
	{
		for (int i = 0; i < 13; i++)
		{
			GF2(ix,iy,i) = GF(ix,iy,i);
			GH2(ix,iy,i) = GH(ix,iy,i);
		}
	}
	else
	{
		// calculate relaxation times from macroscopic properties
		double tauf = (*sim).mu/(rho*(*sim).R*T);
		double tauh = tauf/(*sim).Pr;
		double tauhf = (tauh*tauf)/(tauf - tauh);

		double Tc = (*sim).Tc;
		double R = (*sim).R;
		double K = (*sim).K;

		cuEq2D(rho, u.x, u.y, T, Tc, R, K, feq, heq);

		double f1_, f2_, edotu;

		for (int i = 0; i < 13; i++)
		{
			// temp variables
			f1_ = GF1(ix,iy,i);
			f2_ = (GF(ix,iy,i) + ((*sim).dt/(2*tauf))*f1_)/(1.0 + (*sim).dt/(2*tauf));	//calc to temp variable first, for use later

			GF2(ix,iy,i) = f2_;		// save to global

			// temp variables
			edotu = (*sim).e[i].x*u.x + (*sim).e[i].y*u.y;

			GH2(ix,iy,i) = (GH(ix,iy,i) - ((*sim).dt/2.0)*(edotu/tauhf)*(f1_ - f2_) + ((*sim).dt/2.0)*(GH1(ix,iy,i)/tauh))/(1.0 + (*sim).dt/(2.0*tauh));
		}
	}
}

__global__ void
	RK3_MACRO_PROPERTIES(double* f_d, double* f2_d, double2* fluxf2, double* h_d, double* h2_d, double2* fluxh2, simData* sim, double* rho3_d, double2* u3_d, double* T3_d)
{
	//perform RK3 step 3 macroscopic properties calculation

	// thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//global index
	int ix = blockIdx.x*blockDim.x + tx;
	int iy = blockIdx.y*blockDim.y + ty;

	int Ny = (*sim).Ny;

	//macro variables
	__shared__ double rho_s[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ double2 u_s[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ double T_s[BLOCK_SIZE*BLOCK_SIZE];

	// initialise to zero
	SRHO(tx,ty) = 0;
	SU(tx,ty).x = 0;
	SU(tx,ty).y = 0;
	ST(tx,ty) = 0;

	double f, h, flux_f, flux_h;

	for (int i = 0; i < 13; i++)
	{
		// load f, and h, from global memory
		f = GF(ix,iy,i);
		h = GH(ix,iy,i);

		//calculate fluxes
		flux_f = cuCombineFLUX(fluxf2, i, sim);
		flux_h = cuCombineFLUX(fluxh2, i, sim);		

		SRHO(tx,ty) += f - flux_f;
		SU(tx,ty).x += (f - flux_f)*(*sim).e[i].x;
		SU(tx,ty).y += (f - flux_f)*(*sim).e[i].y;
		ST(tx,ty) += h - flux_h;		
	}
	
	SU(tx,ty).x = SU(tx,ty).x / SRHO(tx,ty);
	SU(tx,ty).y = SU(tx,ty).y / SRHO(tx,ty);

	double usq = sqrt(SU(tx,ty).x*SU(tx,ty).x + SU(tx,ty).y*SU(tx,ty).y);

	usq = usq*usq;
	
	ST(tx,ty) = 2.0*(ST(tx,ty)/SRHO(tx,ty) - usq/2.0)/((*sim).b*(*sim).R);

	// save to global memory
	GRHO3(ix,iy) = SRHO(tx,ty);
	GU3(ix,iy) = SU(tx,ty);
	GT3(ix,iy) = ST(tx,ty);
}

__global__ void
RK3_STEP3_KERNEL(double* f_d, double* f2_d, double*f3_d, double2* fluxf2, double* h_d, double* h2_d, double* h3_d, double2* fluxh2, int* solid_d, double* TW, double2* uW, 
double* rho_d, double2* u_d, double* T_d, double* rho3_d, double2* u3_d, double* T3_d, simData* sim)
{
	//perform RK3 stepping

	//global index
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	int Ny = (*sim).Ny;

	double rho3 = GRHO3(ix,iy);
	double2 u3 = GU3(ix,iy);
	double T3 = GT3(ix,iy);
	
	double feq3[13];
	double heq3[13];

	double Tc = (*sim).Tc;
	double R = (*sim).R;
	double K = (*sim).K;

	int sld = GSLD(ix,iy);

	// --- RK STEP THREE ----
	// calculate the third stage updated distribution functions 

	// macroscopic properties
	double rho2 = GRHO(ix,iy);
	double2 u2 = GU(ix,iy);
	double T2 = GT(ix,iy);

	// calculate relaxation times from macroscopic properties
	double tauf2 = (*sim).mu/(rho2*(*sim).R*T2);
	double tauh2 = tauf2/(*sim).Pr;
	double tauhf2 = (tauh2*tauf2)/(tauf2 - tauh2);

	double tauf3 = (*sim).mu/(rho3*(*sim).R*T3);
	double tauh3 = tauf3/(*sim).Pr;
	double tauhf3 = (tauh3*tauf3)/(tauf3 - tauh3);

	double feq2[13];
	double heq2[13];

	cuEq2D(rho2, u2.x, u2.y, T2, Tc, R, K, feq2, heq2);
	cuEq2D(rho3, u3.x, u3.y, T3, Tc, R, K, feq3, heq3);

	double flux_f2, flux_h2, edotu2, edotu3, f2_, f3_;

	for (int i = 0; i < 13; i++)
	{
		// calculate the updated distribution functions
		if (sld == -1)
		{
			flux_f2 = cuCombineFLUX(fluxf2, i, sim);

			f2_ = GF2(ix,iy,i);

			f3_ = (GF(ix,iy,i) - flux_f2 + ((*sim).dt/(2*tauf2))*(feq2[i] - f2_) + ((*sim).dt/(2*tauf3))*feq3[i])/(1 + (*sim).dt/(2*tauf3));

			GF3(ix,iy,i) = f3_;

			flux_h2 = cuCombineFLUX(fluxh2, i, sim);

			edotu2 = (*sim).e[i].x*u2.x + (*sim).e[i].y*u2.y;
			edotu3 = (*sim).e[i].x*u3.x + (*sim).e[i].y*u3.y;

			GH3(ix,iy,i) = (GH(ix,iy,i) - flux_h2 - ((*sim).dt/2.0)*(edotu2*((feq2[i] - f2_)/tauhf2) + edotu3*((feq3[i] - f3_)/tauhf3)) + 
				((*sim).dt/2.0)*((heq2[i] - GH2(ix,iy,i))/tauh2) + ((*sim).dt/2.0)*(heq3[i]/tauh3))/(1 + (*sim).dt/(2*tauh3));
		}
		else
		{
			GF3(ix,iy,i) = GF(ix,iy,i);
			GH3(ix,iy,i) = GH(ix,iy,i);
		}
	}
}

__global__ void
RK3_COMBINE_KERNEL(double* f_d, double* f2_d, double* f3_d, double2* fluxf2, double2* fluxf3, double* h_d, double* h2_d, double* h3_d, 
double2* fluxh2, double2* fluxh3, double* rho_d, double2* u_d, double* T_d, double* rho3_d, double2* u3_d, double* T3_d, simData* sim, int* solid_d)
{
	//perform RK3 combination step

	//global index
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	int Ny = (*sim).Ny;

	// macroscopic properties
	double rho2 = GRHO(ix,iy);
	double2 u2 = GU(ix,iy);
	double T2 = GT(ix,iy);

	double rho3 = GRHO3(ix,iy);
	double2 u3 = GU3(ix,iy);
	double T3 = GT3(ix,iy);

	// calculate relaxation times from macroscopic properties
	double tauf2 = (*sim).mu/(rho2*(*sim).R*T2);
	double tauh2 = tauf2/(*sim).Pr;
	double tauhf2 = (tauh2*tauf2)/(tauf2 - tauh2);

	double tauf3 = (*sim).mu/(rho3*(*sim).R*T3);
	double tauh3 = tauf3/(*sim).Pr;
	double tauhf3 = (tauh3*tauf3)/(tauf3 - tauh3);

	double feq2[13];
	double heq2[13];

	double feq3[13];
	double heq3[13];

	double Tc = (*sim).Tc;
	double R = (*sim).R;
	double K = (*sim).K;

	cuEq2D(rho2, u2.x, u2.y, T2, Tc, R, K, feq2, heq2);
	cuEq2D(rho3, u3.x, u3.y, T3, Tc, R, K, feq3, heq3);

	double flux_f2, flux_h2, flux_f3, flux_h3, edotu2, edotu3;
	double f2_, f3_, h2_, h3_;

	int sld = GSLD(ix,iy);

	for (int i = 0; i < 13; i++)
	{
		// --- RK COMBINATION ----
		// calculate the updated distribution functions
		if (sld == -1)
		{

			f2_ = GF2(ix,iy,i);
			f3_ = GF3(ix,iy,i);

			flux_f2 = cuCombineFLUX(fluxf2, i, sim);
			flux_f3 = cuCombineFLUX(fluxf3, i, sim);

			GF(ix,iy,i) = GF(ix,iy,i) - (1.0/2.0)*(flux_f2 + flux_f3) + ((*sim).dt/2.0)*((feq2[i] - f2_)/tauf2 + (feq3[i] - f3_)/tauf3);

			h2_ = GH2(ix,iy,i);
			h3_ = GH3(ix,iy,i);

			flux_h2 = cuCombineFLUX(fluxh2, i, sim);
			flux_h3 = cuCombineFLUX(fluxh3, i, sim);

			edotu2 = (*sim).e[i].x*u2.x + (*sim).e[i].y*u2.y;
			edotu3 = (*sim).e[i].x*u3.x + (*sim).e[i].y*u3.y;

			GH(ix,iy,i) = GH(ix,iy,i) - 0.5*(flux_h2 + flux_h3) + ((*sim).dt/2.0)*((heq2[i] - h2_)/tauh2 + (heq3[i] - h3_)/tauh3) 
				- ((*sim).dt/2.0)*((edotu2/tauhf2)*(feq2[i] - f2_) + (edotu3/tauhf3)*(feq3[i] - f3_));
		}
	}
	__syncthreads();
}
