////////////////
// MACROS
////////////////

// LBM_CPU

#define BNDARRAY(x,y) bndArray[x*Ny + y]
#define F(x,y,i) f_G[x*Ny*Ni + y*Ni + i]
#define H(x,y,i) h_G[x*Ny*Ni + y*Ni + i]
#define RHO(x,y) rho_G[x*Ny + y]
#define UX(i,j) u_G[i*Ny + j].x
#define UY(i,j) u_G[i*Ny + j].y
#define P(x,y) p_G[x*Ny + y]
#define T(x,y) T_G[x*Ny + y]

// RK1
#define FR1(x,y,i) fr1[x*Ny*Ni + y*Ni + i]
#define HR1(x,y,i) hr1[x*Ny*Ni + y*Ni + i]
#define TAUF(x,y) tauf[x*Ny + y]
#define TAUH(x,y) tauh[x*Ny + y]
#define TAUHF(x,y) tauhf[x*Ny + y]
#define FEQ(x,y,i) feq[x*Ny*Ni + y*Ni + i]
#define HEQ(x,y,i) heq[x*Ny*Ni + y*Ni + i]
#define FLUXES_F(x,y,i) fluxes_f[x*Ny*Ni + y*Ni + i]
#define FLUXES_H(x,y,i) fluxes_h[x*Ny*Ni + y*Ni + i]

// RK3
#define FR2(x,y,i) fr2[x*Ny*Ni + y*Ni + i]
#define HR2(x,y,i) hr2[x*Ny*Ni + y*Ni + i]
#define FR3(x,y,i) fr3[x*Ny*Ni + y*Ni + i]
#define HR3(x,y,i) hr3[x*Ny*Ni + y*Ni + i]
#define RHOR1(x,y) rhor1[x*Ny + y]
#define RHOR2(x,y) rhor2[x*Ny + y]
#define RHOR3(x,y) rhor3[x*Ny + y]
#define UR1(i,j) ur1[i*Ny + j]
#define UR1X(i,j) ur1[i*Ny + j].x
#define UR1Y(i,j) ur1[i*Ny + j].y
#define UR2(i,j) ur2[i*Ny + j]
#define UR2X(i,j) ur2[i*Ny + j].x
#define UR2Y(i,j) ur2[i*Ny + j].y
#define UR3(i,j) ur3[i*Ny + j]
#define UR3X(i,j) ur3[i*Ny + j].x
#define UR3Y(i,j) ur3[i*Ny + j].y
#define PR1(x,y) pr1[x*Ny + y]
#define PR2(x,y) pr2[x*Ny + y]
#define PR3(x,y) pr3[x*Ny + y]
#define TR1(x,y) Tr1[x*Ny + y]
#define TR2(x,y) Tr2[x*Ny + y]
#define TR3(x,y) Tr3[x*Ny + y]
#define FEQ1(x,y,i) feq1[x*Ny*Ni + y*Ni + i]
#define HEQ1(x,y,i) heq1[x*Ny*Ni + y*Ni + i]
#define FEQ2(x,y,i) feq2[x*Ny*Ni + y*Ni + i]
#define HEQ2(x,y,i) heq2[x*Ny*Ni + y*Ni + i]
#define FEQ3(x,y,i) feq3[x*Ny*Ni + y*Ni + i]
#define HEQ3(x,y,i) heq3[x*Ny*Ni + y*Ni + i]
#define TAUF1(x,y) tauf1[x*Ny + y]
#define TAUF2(x,y) tauf2[x*Ny + y]
#define TAUF3(x,y) tauf3[x*Ny + y]
#define TAUH1(x,y) tauh1[x*Ny + y]
#define TAUH2(x,y) tauh2[x*Ny + y]
#define TAUH3(x,y) tauh3[x*Ny + y]
#define TAUHF1(x,y) tauhf1[x*Ny + y]
#define TAUHF2(x,y) tauhf2[x*Ny + y]
#define TAUHF3(x,y) tauhf3[x*Ny + y]
#define FLUXES_FR2(x,y,i) fluxes_fr2[x*Ny*Ni + y*Ni + i]
#define FLUXES_HR2(x,y,i) fluxes_hr2[x*Ny*Ni + y*Ni + i]
#define FLUXES_FR3(x,y,i) fluxes_fr3[x*Ny*Ni + y*Ni + i]
#define FLUXES_HR3(x,y,i) fluxes_hr3[x*Ny*Ni + y*Ni + i]
#define FLUX_F2(x,y,i) flux_f2[x*Ny*Ni + y*Ni + i]
#define FLUX_H2(x,y,i) flux_h2[x*Ny*Ni + y*Ni + i]

// cFLUX
#define FLUXOUTX(i,j,k) fluxOut[i*Ny*Ni + j*Ni + k].x
#define FLUXOUTY(i,j,k) fluxOut[i*Ny*Ni + j*Ni + k].y

// posFluxes
#define F_FLUX_X(i,j,k) f_flux[i*Ny*Ni + j*Ni + k].x
#define F_FLUX_Y(i,j,k) f_flux[i*Ny*Ni + j*Ni + k].y
#define H_FLUX_X(i,j,k) h_flux[i*Ny*Ni + j*Ni + k].x
#define H_FLUX_Y(i,j,k) h_flux[i*Ny*Ni + j*Ni + k].y

// GENERIC
#define FF(x,y,i) f[x*Ny*Ni + y*Ni + i]
#define HH(x,y,i) h[x*Ny*Ni + y*Ni + i]

#define RRHO(x,y) rho[x*Ny + y]
#define UUX(i,j) u[i*Ny + j].x
#define UUY(i,j) u[i*Ny + j].y
#define PP(x,y) p[x*Ny + y]
#define TT(x,y) T[x*Ny + y]

//FUNCTIONS
#define MAX(a,b) ((a > b) ? a : b)
#define MIN(a,b) ((a < b) ? a : b)