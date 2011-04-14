// kernel.cl

#pragma OPENCL EXTENSION cl_amd_fp64 : enable 

/////////////////////////////////////////
//MACROS 
/////////////////////////////////////////

#define F_G(x,y,i) f_G[x*Ni*Ny + y*Ni + i] 
#define H_G(x,y,i) h_G[x*Ni*Ny + y*Ni + i] 
#define BND(x,y) bnd[x*Ny + y] 
#define CELL(i) cell[i] 
#define DENSITY(i) density[i] 
#define VELX(i) velX[i] 
#define VELY(i) velY[i] 
#define THERM(i) therm[i] 
#define RHO_G(x,y) rho_G[x*Ny + y] 
#define UX_G(x,y) ux_G[x*Ny + y] 
#define UY_G(x,y) uy_G[x*Ny + y] 
#define T_G(x,y)  T_G[x*Ny + y] 

// macros for general use
#define F(x,y,i) f[x*Ni*Ny + y*Ni + i] 
#define H(x,y,i) h[x*Ni*Ny + y*Ni + i] 
#define F_FLUX_X(x,y,i) f_flux_x[x*Ni*Ny + y*Ni + i]
#define F_FLUX_Y(x,y,i) f_flux_y[x*Ni*Ny + y*Ni + i]
#define H_FLUX_X(x,y,i) h_flux_x[x*Ni*Ny + y*Ni + i]
#define H_FLUX_Y(x,y,i) h_flux_y[x*Ni*Ny + y*Ni + i]

/////////////////////////////////////////
//CONSTANTS 
/////////////////////////////////////////

#define Nx 100
#define Ny 200
#define Ni 13 

__constant double mu = 1.86e-05;
__constant double Pr = 0.71;
__constant double R = 287.0;
__constant double K = 3.0;
__constant double b = 5.0;
__constant double Tc = 606.002602025;

__constant double dt = 3.02758695683e-06;

__constant double dx = 0.00505050505051;

__constant double dy = 0.00505050505051;

__constant int periodicX = 0;
__constant int periodicY = 0;
__constant int mirrorN = 1;
__constant int mirrorS = 1;
__constant int mirrorE = 1;
__constant int mirrorW = 1;

__constant double ex[13] = {0.0, 417.040461803, 0.0, -417.040461803, 0.0, 417.040461803, -417.040461803, -417.040461803, 417.040461803, 834.080923607, 0.0, -834.080923607, 0.0};
__constant double ey[13] = {0.0, 0.0, 417.040461803, 0.0, -417.040461803, 417.040461803, 417.040461803, -417.040461803, -417.040461803, 0.0, 834.080923607, 0.0, -834.080923607};

__constant unsigned int mirrorNS[13] = {0,1,4,3,2,8,7,6,5,9,12,11,10};
__constant unsigned int mirrorEW[13] = {0,3,2,1,4,6,5,8,7,11,10,9,12};

__constant int xx[4] = {-1,1,0,0};
__constant int yy[4] = {0,0,-1,1};
__constant unsigned    int ii[4][4] = {{1,5,8,9},{3,6,7,11},{2,5,6,10},{4,7,8,12}};
__constant unsigned    int inv[4][4] = {{3,6,7,11},{1,5,8,9},{4,7,8,12},{2,5,6,10}};
/////////////////////////////////////////
// DEVICE FUNCTIONS
/////////////////////////////////////////

// equilibrium functions for f

double f0(double rho, double u, double v, double T)
{
   //rest particle equilibrium function
    return (rho/4.0)*(4 + 10*pow(T,2) + pow(u,4) - 5*pow(v,2) + pow(v,4) + 10*T*(-1 + pow(u,2) + pow(v,2)) + pow(u,2)*(-5 + 4*pow(v,2)));
}

double f1(double rho, double u, double v, double T)
{ 
    return (rho/6.0)*(-6*pow(T,2) - u*(1 + u)*(-4 + pow(u,2) + 3*pow(v,2)) - T*(-4 + 6*u + 9*pow(u,2) + 3*pow(v,2)));
}

double f5(double rho, double u, double v, double T)
{
    return (rho/4.0)*((T + u + pow(u,2))*(T + v + pow(v,2)));
}

double f9(double rho, double u, double v, double T)
{
    return (rho/24.0)*(3*pow(T,2) + (-1 + u)*u*(1 + u)*(2 + u) + T*(-1 + 6*u*(1 + u)));
}

// equilibrium functions for h

double h0(double T, double u, double v)
{
return (10*(16 + 3*K)*pow(T,3) + 3*T*(8 + 4*K - 40*pow(u,2) - 5*K*pow(u,2) + 20*pow(u,4)
    + K*pow(u,4) + (-5*(8 + K) + 4*(15 + K)*pow(u,2))*pow(v,2) + (20 + K)*pow(v,4))
    + 30*pow(T,2)*(-4 + 9*pow(u,2) + 9*pow(v,2) + K*(-1 + pow(u,2) + pow(v,2)))
    + 3*(pow(u,2) + pow(v,2))*(4 + pow(u,4) - 5*pow(v,2) + pow(v,4) + pow(u,2)*(-5 + 4*pow(v,2))))/24.0;
}

double h1(double T, double u, double v)
{
    return (-2*(16 + 3*K)*pow(T,3) - u*(1 + u)*(pow(u,2) + pow(v,2))*(-4 + pow(u,2) + 3*pow(v,2))
           - T*(u*(-4*(4 + K) - 4*(7 + K)*u + (14 + K)*pow(u,2) + (19 + K)*pow(u,3)) + (-4 + 3*u*(10
            + K + (14 + K)*u))*pow(v,2) + 3*pow(v,4)) - pow(T,2)*(-16 + 6*u*(6 + 13*u) + 30*pow(v,2)
            + K*(-4 + 6*u + 9*pow(u,2) + 3*pow(v,2))))/12.0;
}

double h5(double T, double u, double v)
{
    return ((16 + 3*K)*pow(T,3) + 3*u*(1 + u)*v*(1 + v)*(pow(u,2) + pow(v,2)) + 3*pow(T,2)*((6 + K)*u
            + (9 + K)*pow(u,2) + v*(6 + K + (9 + K)*v)) + 3*T*(pow(u,3) + pow(u,4) + pow(v,3)*(1 + v)
            + u*v*(6 + K + (9 + K)*v) + pow(u,2)*v*(9 + K + (12 + K)*v)))/24.0;
}

double h9(double T, double u, double v)
{
    return ((16 + 3*K)*pow(T,3) + T*u*(-8 + K*(-1 + u)*(1 + u)*(2 + u) + u*(-7 + 2*u*(11 + 8*u)))
            + T*(-1 + 6*u*(1 + u))*pow(v,2) + (-1 + u)*u*(1 + u)*(2 + u)*(pow(u,2) + pow(v,2))
            + pow(T,2)*(-4 + 36*u + 51*pow(u,2) + K*(-1 + 6*u*(1 + u)) + 3*pow(v,2)))/48.0;
}

////////////////////////////////////////////////////////////////////////////////
// clEq2D
////////////////////////////////////////////////////////////////////////////////

int clEq2D(double rho, double u_, double v_, double T_, double* eqf, double* eqh)
{
    //returns the equilibrium values for each velocity vector given the current
    // macroscopic values and the corresponding reference quantities

    double u = u_/417.040461803;
    double v = v_/417.040461803;
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

    double rRTc = rho*173922.746781;

    eqh[0] = rRTc*h0(T,u,v);
    eqh[1] = rRTc*h1(T,u,v);
    eqh[2] = rRTc*h1(T,v,u);
    eqh[3] = rRTc*h1(T,-u,v);
    eqh[4] = rRTc*h1(T,-v,u);
    eqh[5] = rRTc*h5(T,u,v);
    eqh[6] = rRTc*h5(T,-u,v);
    eqh[7] = rRTc*h5(T,-u,-v);
    eqh[8] = rRTc*h5(T,u,-v);
    eqh[9] = rRTc*h9(T,u,v);
    eqh[10] =rRTc*h9(T,v,u);
    eqh[11] =rRTc*h9(T,-u,v);
    eqh[12] =rRTc*h9(T,-v,u);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// clMirIndex
////////////////////////////////////////////////////////////////////////////////

void
clMirIndex(int i, int mir_x, int mir_y, int* i_x, int* i_y){
    // gives the mirrored index about the axes defined by inputs mir_x, and mir_y

     // initialise with no mirror
     (*i_x) = i;
     (*i_y) = i;

     // mirror about x
     if (mir_x == 1) {
        (*i_x) = mirrorNS[i];
     }

     // mirror about y
     if(mir_y == 1) {
        (*i_y) = mirrorEW[i];
     }
}
////////////////////////////////////////////////////////////////////////////////
// clSign
////////////////////////////////////////////////////////////////////////////////

int
clSign(double a)
{
    //returns: a = 0 -> 0, a = neg -> -1, a = pos -> 1
     if (a > 0) {
    return 1;
    }
    else if (a < 0) {
     return -1;
    }
    else {
         return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////
// clIndex
////////////////////////////////////////////////////////////////////////////////

int
clIndex(int i, int pm, int d, int* mir)
{
    // calculate index of next item given current item i and direction pm
    // d = 0 -> x
    // d = 1 -> y

    i = i + pm;
    (*mir) = 0;

    if (d == 0) {
        if (i < 0) {
               i = -i - 1;
               (*mir) = 1;
        }
        else if (i > Nx - 1) {
                i = 2*Nx - i - 1;
                (*mir) = 1;
        }
    }
    else if (d == 1) {
        if (i < 0) {
                i = -i - 1;
                (*mir) = 1;
        }
        else if (i > Ny - 1) {
                i = 2*Ny - i - 1;
                (*mir) = 1;
        }
    }
    return i;
}

////////////////////////////////////////////////////////////////////////////////
// clStencil
////////////////////////////////////////////////////////////////////////////////

void
clStencil(__global double* f,
    double* Sx,
    double* Sy,
    int i,
    __global int* cell,
    __global int* bnd)
{
    // create a stencil of given length for the Flux method specified

    if (i == 0) {
        return;
    }

    // global index
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    // velocity vector, in integer values for indexing
    int exi = clSign(ex[i]);
    int eyi = clSign(ey[i]);

    int low = 2;    // max/min num of stencil from centre

    // calculate stencil along x and y
    int jx1 = ix - exi*low;        //start of stencil index
    int jy1 = iy - eyi*low;

    int jx, jy, mir_x, mir_y, i_x, i_y;

    int sld_x = 0;
    int sld_y = 0;
    int sldx[5];
    int sldy[5];

    int id, type;

    for (int j = 0; j < 5; j++) {
        // put together stencil along axis lines, start on low side, go to hi side: [j-low <-> j+low]
        jx = clIndex(jx1,exi*j,0,&mir_y);
        jy = clIndex(jy1,eyi*j,1,&mir_x);

        clMirIndex(i,mir_x,mir_y,&i_x,&i_y);    // mirror indexes if required

        id = BND(jx,iy);
        type = CELL(id);
        // x
        Sx[j] = F(jx,iy,i_y);

        if (type == 2) { // if solid
            sldx[j] = 1;
            sld_x = 1;
        }
        else {
            sldx[j] = 0;
        }

        // y
        Sy[j] = F(ix,jy,i_x);

        if (type == 2) {
            sldy[j] = 1;
            sld_y = 1;
        }
        else { 
            sldy[j] = 0;
        }
    }

    int diff;    // difference between indexes of sldx
    int x_;    // location of edge solid node, outside coords
    int y_;    // location of edge solid node, outside coords
    int j_;    // location of edge solid node, stencil coords

    if (sld_x == 1) {
        // find where edge solid node is
        for (int j = 0; j < 4; j++) {
            diff = sldx[j] - sldx[j+1];
            if (diff > 0) {
                j_ = j;
                x_ = clIndex(jx1,exi*j,0,&mir_y);
                break;
            }
            else if (diff < 0) {
                j_ = j + 1;
                x_ = clIndex(jx1,exi*(j + 1),0,&mir_y);
                break;
            }
        }

        // update solid part of stencil with linear extrapolation of actual stencil

        int xm1 = clIndex(x_,exi*diff,0,&mir_y);    // closest fluid node
        int xm2 = clIndex(x_,exi*2*diff,0,&mir_y);    // next closest fluid node

        Sx[j_] = 2*F(xm1,iy,i) - F(xm2,iy,i);

        int j__ = j_ - diff;

        if (j__ >= 0 && j__ < 5) { // update next one along if stencil is long enough
            Sx[j__] = Sx[j_];
        }
    }

    if (sld_y == 1) {

        // find where edge solid node is
        for (int j = 0; j < 4; j++) {
            diff = sldy[j] - sldy[j+1];
            if (diff > 0) {
                j_ = j;
                y_ = clIndex(jy1,eyi*j,1,&mir_x);
                break;
            }
            else if (diff < 0) {
                j_ = j + 1;
                y_ = clIndex(jy1,eyi*(j + 1),1,&mir_x);
                break;
            }
        }

        // update solid part of stencil with linear extrapolation of actual stencil

        int ym1 = clIndex(y_,eyi*diff,0,&mir_x);    // closest fluid node
        int ym2 = clIndex(y_,eyi*2*diff,0,&mir_x);    // next closest fluid node

        Sy[j_] = 2*F(ix,ym1,i) - F(ix,ym2,i);

        int j__ = j_ - diff;

        if (j__ >= 0 && j__ < 5) {    // update next one along if stencil is long enough
            Sy[j__] = Sy[j_];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// clWENO5
////////////////////////////////////////////////////////////////////////////////

double
clWENO5(double* S, double e)
{
    // calculate the flux term of the WENO5 scheme for one flow direction

    double epsilon = (double) 1e-6;
    double posFlow = fabs(e);
    double F_Ip12;

    if (posFlow == 0) {
        return 0;
    }
    else {
        double Sp[5];
        for (int i = 0; i < 5; i++) {
            Sp[i] = S[i]*posFlow;
        }

        double B0p, B1p, B2p, alpha0p, alpha1p, alpha2p;
        double omega0p, omega1p, omega2p, f0p, f1p, f2p;

        B0p = 1.08333333333*pow(Sp[0] - 2.0*Sp[1] + Sp[2],2) + 0.25*pow(Sp[0] - 4.0*Sp[1] + 3.0*Sp[2],2);
        B1p = 1.08333333333*pow(Sp[1] - 2.0*Sp[2] + Sp[3],2) + 0.25*pow(Sp[1] - Sp[3],2);
        B2p = 1.08333333333*pow(Sp[2] - 2.0*Sp[3] + Sp[3],2) + 0.25*pow(3.0*Sp[2] - 4.0*Sp[3] + Sp[4],2);

        alpha0p = 0.1*pow(1.0/(epsilon + B0p),2);
        alpha1p = 0.6*pow(1.0/(epsilon + B1p),2);
        alpha2p = 0.3*pow(1.0/(epsilon + B2p),2);

        omega0p = alpha0p/(alpha0p + alpha1p + alpha2p);
        omega1p = alpha1p/(alpha0p + alpha1p + alpha2p);
        omega2p = alpha2p/(alpha0p + alpha1p + alpha2p);

        f0p = 0.333333333333*Sp[0] - 1.16666666667*Sp[1] + 1.83333333333*Sp[2];
        f1p = -0.166666666667*Sp[1] + 0.833333333333*Sp[2] + 0.333333333333*Sp[3];
        f2p = 0.333333333333*Sp[2] + 0.833333333333*Sp[3] - 0.166666666667*Sp[4];

        F_Ip12 = omega0p*f0p + omega1p*f1p + omega2p*f2p;

    }

    return F_Ip12;
}

////////////////////////////////////////////////////////////////////////////////
// clFLUX
////////////////////////////////////////////////////////////////////////////////

void
clFLUX(__global double* f, int i, __global int* cell, __global int* bnd, double* flux_x, double* flux_y)
{
    // choose flux method

    double exi, eyi;

    exi = ex[i];
    eyi = ey[i];

    // WENO5
    double Sx[5];
    double Sy[5];

    clStencil(f, Sx, Sy, i, cell, bnd);

    (*flux_x) = clWENO5(Sx, exi);
    (*flux_y) = clWENO5(Sy, eyi);

}

////////////////////////////////////////////////////////////////////////////////
// clCombineFLUX
////////////////////////////////////////////////////////////////////////////////

#define FLUX_X(x,y,i) flux_x[x*Ni*Ny + y*Ni + i]
#define FLUX_Y(x,y,i) flux_y[x*Ni*Ny + y*Ni + i]
double
clCombineFLUX(__global double* flux_x, __global double* flux_y, int i)
{
    // calculates the combined flux given all the fluxes out of each node

    // global index
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    // velocity vector, in integer values for indexing
    int exi = clSign(ex[i]);
    int eyi = clSign(ey[i]);

    int mir_x, mir_y;

    int x_ = clIndex(ix,-exi,0,&mir_y);
    int y_ = clIndex(iy,-eyi,1,&mir_x);

    if (i == 0) {

        return 0;
    }
    else {
        int i_x, i_y;

        clMirIndex(i,mir_x,mir_y,&i_x,&i_y);    // mirror indexes if required

        double flux_out = dt*(FLUX_X(ix,iy,i)/dx + FLUX_Y(ix,iy,i)/dy);

        double flux_in =  dt*(FLUX_X(x_,iy,i_y)/dx + FLUX_Y(ix,y_,i_x)/dy);

        return flux_out - flux_in;
    }
}

////////////////////////////////////////////////////////////////////////////////
// clPosFlux
////////////////////////////////////////////////////////////////////////////////

void
clPosFlux(__global double* f,
    __global double* h,
    int i,
    __global int* cell,
    __global int* bnd,
    __global double* f_flux_x,
    __global double* f_flux_y,
    __global double* h_flux_x,
    __global double* h_flux_y)
{
    // function to calculate all positive fluxes along axis (x & y) lines for fluid nodes
    // store fluxes in global memory

    // global index
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    // temporary flux storage

    double flux_fx, flux_fy, flux_hx, flux_hy;

    int id = BND(ix,iy);
    int type = CELL(id);
    if (type != 2) {    // only perform fluxes on fluid and permanent nodes
        clFLUX(f,i,cell,bnd,&flux_fx,&flux_fy);
        clFLUX(h,i,cell,bnd,&flux_hx,&flux_hy);

        // store fluxes in global memory
        F_FLUX_X(ix,iy,i) = flux_fx;
        F_FLUX_Y(ix,iy,i) = flux_fy;

        H_FLUX_X(ix,iy,i) = flux_hx;
        H_FLUX_Y(ix,iy,i) = flux_hy;
    }
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: clMacroProp
////////////////////////////////////////////////////////////////////////////////

#define RHO(x,y) rho[x*Ny + y] 
#define UX(x,y) ux[x*Ny + y] 
#define UY(x,y) uy[x*Ny + y] 
#define T(x,y)  T[x*Ny + y] 
__kernel void
clMacroProp(__global double* f,
    __global double* h,
    __global double* rho,
    __global double* ux,
    __global double* uy,
    __global double* T)
{
    // calculate macroscopic properties

    // global index
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    double rho_ = 0.0;
    double rho_ux = 0.0;
    double rho_uy = 0.0;
    double sum_h = 0.0;
    double ux_, uy_, T_;

    for (int i = 0; i < Ni; i++) {
        rho_    += F(ix,iy,i);
        rho_ux += F(ix,iy,i)*ex[i];
        rho_uy += F(ix,iy,i)*ey[i];
        sum_h  += H(ix,iy,i);
    }
    ux_ = rho_ux/rho_;
    uy_ = rho_uy/rho_;

    double ux_abs, uy_abs, usq;

    ux_abs = fabs(ux_);
    uy_abs = fabs(uy_);
    usq = ux_abs*ux_abs + uy_abs*uy_abs;
    T_ = 2.0*(sum_h/rho_ - usq/2.0)/(b*R);

    // assign variables to arrays
    RHO(ix,iy) = rho_;
    UX(ix,iy) = ux_;
    UY(ix,iy) = uy_;
    T(ix,iy) = T_;
}

////////////////////////////////////////////////////////////////////////////////
// clMacroProp3
////////////////////////////////////////////////////////////////////////////////

#define RHO_3(x,y) rho_3[x*Ny + y] 
#define UX_3(x,y) ux_3[x*Ny + y] 
#define UY_3(x,y) uy_3[x*Ny + y] 
#define T_3(x,y)  T_3[x*Ny + y] 

__kernel void
clMacroProp3(__global double* f,
    __global double* h,
    __global double* rho_3,
    __global double* ux_3,
    __global double* uy_3,
    __global double* T_3,    __global double* f_flux_x2,
    __global double* f_flux_y2,
    __global double* h_flux_x2,
    __global double* h_flux_y2)
{
    //perform RK3 step 3 macroscopic properties calculation

    // global index
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    double rho_ = 0.0;
    double rho_ux = 0.0;
    double rho_uy = 0.0;
    double sum_h = 0.0;
    double ux_, uy_, T_, flux_f, flux_h;

    for (int i = 0; i < Ni; i++) {
        //calculate fluxes
        flux_f = clCombineFLUX(f_flux_x2, f_flux_y2, i);
        flux_h = clCombineFLUX(h_flux_x2, h_flux_y2, i);

        rho_    += F(ix,iy,i) - flux_f;
        rho_ux  += (F(ix,iy,i) - flux_f)*ex[i];
        rho_uy  += (F(ix,iy,i) - flux_f)*ey[i];
        sum_h   += H(ix,iy,i) - flux_h;
    }
    ux_ = rho_ux/rho_;
    uy_ = rho_uy/rho_;

    double ux_abs, uy_abs, usq;

    ux_abs = fabs(ux_);
    uy_abs = fabs(uy_);
    usq = ux_abs*ux_abs + uy_abs*uy_abs;
    T_ = 2.0*(sum_h/rho_ - usq/2.0)/(b*R);

    // assign variables to arrays
    RHO_3(ix,iy) = rho_;
    UX_3(ix,iy) = ux_;
    UY_3(ix,iy) = uy_;
    T_3(ix,iy) = T_;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: initFunctions
////////////////////////////////////////////////////////////////////////////////

// KERNEL FUNCTION
__kernel void
initFunctions(__global double* f_G,
    __global double* h_G,
    __global int* bnd,
    __global double* density,
    __global double* velX,
    __global double* velY,
    __global double* therm,
    __global double* rho_G,
    __global double* ux_G,
    __global double* uy_G,
    __global double* T_G) {
    // initialise functions

    // global index
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    int id = BND(ix,iy);

    double rho = DENSITY(id);
    double ux = VELX(id);
    double uy = VELY(id);
    double T = THERM(id);

    double eqf[Ni];
    double eqh[Ni];

    // compute equilibrium functions
    clEq2D(rho, ux, uy, T, eqf, eqh);
    for (int i = 0; i < Ni; i++) {
        F_G(ix,iy,i) = eqf[i];
        H_G(ix,iy,i) = eqh[i];
    }

    clMacroProp(f_G, h_G, rho_G, ux_G, uy_G, T_G);
}////////////////////////////////////////////////////////////////////////////////
//KERNEL: GLOBAL_FLUXES
////////////////////////////////////////////////////////////////////////////////

__kernel void
GLOBAL_FLUXES(__global double* f,
    __global double* h,
    __global int* cell,
    __global int* bnd,
    __global double* f_flux_x,
    __global double* f_flux_y,
    __global double* h_flux_x,
    __global double* h_flux_y)
{
    // calculate all outgoing fluxes for the given distribution functions and save to global memory.

    for (int i = 0; i < Ni; i++) {

        // calc fluxes, save to global
        clPosFlux(f, h, i, cell, bnd, f_flux_x, f_flux_y, h_flux_x, h_flux_y);
    }
}
////////////////////////////////////////////////////////////////////////////////
//KERNEL: WALL_FLUXES
////////////////////////////////////////////////////////////////////////////////

__kernel void 
WALL_FLUXES(__global double* f_flux_x,
    __global double* f_flux_y,
    __global double* h_flux_x,
    __global double* h_flux_y,
    __global int* cell,
    __global int* bnd,
    __global double* therm,
    __global double* velX,
    __global double* velY)
{
    // calculate fluxes into solid nodes and set fluxes out of solid to cancel them out

    // global index
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    int xm1, ym1, i, inv_i;
    int mir_x, mir_y;

    double flux_in, flux_out, alpha;

    double feqW[Ni];
    double heqW[Ni];

    double ux, uy;
    int inx,iny;
    int id = BND(ix,iy);
    int type = CELL(id);

    if ( type == 2) {    //wall nodes

        flux_in = 0;
        flux_out = 0;

        clEq2D(1.0, VELX(id), VELY(id), THERM(id), feqW, heqW);

        // check surrounds for fluid node
        for (int j = 0; j < 4; j++) {
            xm1 = clIndex(ix,xx[j],0,&mir_y);
            ym1 = clIndex(iy,yy[j],1,&mir_x);

            id = BND(xm1,ym1);
            type = CELL(id);

            if (type != 2) { // if fluid

                for (int k = 0; k < 4; k++) {
                    i = ii[j][k];    // index of velocities into solid
                    inv_i = inv[j][k];    // inverted velocities, out of solid

                    // switches for turning off velocities that don't impinge on solid, or leave solid, through the cell pointed to by xx & yy
                    inx = abs(xx[j]);    
                    iny = abs(yy[j]);

                    flux_in += inx*F_FLUX_X(xm1,ym1,i)/dx + iny*F_FLUX_Y(xm1,ym1,i)/dy;        // flux into solid

                    // absolute value of velocities
                    ux = fabs(ex[inv_i]);    
                    uy = fabs(ey[inv_i]);

                    flux_out += inx*feqW[inv_i]*ux/dx + iny*feqW[inv_i]*uy/dy;                // flux out of solid, back along inverse velocity
                }
            }
        }

        alpha = flux_in/flux_out;    // correction factor to equalise flux in to flux out

        // load required fluxes into flux array
        for (int i = 0; i < 13; i++) {
            ux = fabs(ex[i]);    
            uy = fabs(ey[i]);

             F_FLUX_X(ix,iy,i) = alpha*ux*feqW[i];
             F_FLUX_Y(ix,iy,i) = alpha*uy*feqW[i];

             H_FLUX_X(ix,iy,i) = alpha*ux*heqW[i];
             H_FLUX_Y(ix,iy,i) = alpha*uy*heqW[i];
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK3_STEP1
////////////////////////////////////////////////////////////////////////////////

// macros for RK3
#define FR1(x,y,i) fr1[x*Ny*Ni + y*Ni + i]
#define HR1(x,y,i) hr1[x*Ny*Ni + y*Ni + i]
__kernel void
RK3_STEP1(__global double* f_G,
    __global double* fr1,
    __global double* h_G,
    __global double* hr1,
    __global int* cell,
    __global int* bnd,
    __global double* therm,
    __global double* velX,
    __global double* velY,
    __global double* rho_G,
    __global double* ux_G,
    __global double* uy_G,
    __global double* T_G)
{
    //perform RK3 stepping

    // global index
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    // macroscopic properties
    double rho = RHO_G(ix,iy);
    double ux = UX_G(ix,iy);
    double uy = UY_G(ix,iy);
    double T = T_G(ix,iy);

    // calculate relaxation times from macroscopic properties
    double tauf = mu/(rho*R*T);
    double tauh = tauf/Pr;
    double tauhf = (tauh*tauf)/(tauf - tauh);

    double feq[Ni];
    double heq[Ni];

    clEq2D(rho, ux, uy, T, feq, heq);

    int id = BND(ix,iy);
    int type = CELL(id);

    // --- RK STEP ONE----
    // calculate the first stage updated distribution functions 
    for (int i = 0; i < Ni; i++) {
        if (type > 0) {
            FR1(ix,iy,i) = F_G(ix,iy,i);
            HR1(ix,iy,i) = H_G(ix,iy,i);
        }
        else {
            double f1_, edotu;
            // temp variables
            f1_ = (F_G(ix,iy,i) + (dt/(2*tauf))*feq[i])/(1.0 + dt/(2*tauf));    //calc to temp variable first, for use later

            FR1(ix,iy,i) = f1_;        // save to global

            // temp variables
            edotu = ex[i]*ux + ey[i]*uy;

            HR1(ix,iy,i) = (H_G(ix,iy,i) - (dt/2.0)*edotu*((feq[i] - f1_)/tauhf) + (dt/(2.0*tauh))*heq[i])/(1.0 + dt/(2.0*tauh));
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK3_STEP2
////////////////////////////////////////////////////////////////////////////////

// macros for RK3
#define FR2(x,y,i) fr2[x*Ny*Ni + y*Ni + i]
#define HR2(x,y,i) hr2[x*Ny*Ni + y*Ni + i]
__kernel void
RK3_STEP2(__global double* f_G,
    __global double* fr1,
    __global double* fr2,
    __global double* h_G,
    __global double* hr1,
    __global double* hr2,
    __global int* cell,
    __global int* bnd,
    __global double* therm,
    __global double* velX,
    __global double* velY,
    __global double* rho_G,
    __global double* ux_G,
    __global double* uy_G,
    __global double* T_G)
{
    //perform RK3 stepping

    // NOTE: as all equilibrium df are calulated from the same data, feq1 = feq2 etc, this allows for simplification of equations

    // global index
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    // macroscopic properties
    double rho = RHO_G(ix,iy);
    double ux = UX_G(ix,iy);
    double uy = UY_G(ix,iy);
    double T = T_G(ix,iy);

    // calculate relaxation times from macroscopic properties
    double tauf = mu/(rho*R*T);
    double tauh = tauf/Pr;
    double tauhf = (tauh*tauf)/(tauf - tauh);

    double feq[Ni];
    double heq[Ni];

    clEq2D(rho, ux, uy, T, feq, heq);

    int id = BND(ix,iy);
    int type = CELL(id);

    // --- RK STEP TWO----
    // calculate the first stage updated distribution functions 

    if (type > 0) {
        for (int i = 0; i < Ni; i++) {
            FR2(ix,iy,i) = F_G(ix,iy,i);
            HR2(ix,iy,i) = H_G(ix,iy,i);
        }
    }
    else {

        double f1_, f2_, edotu;

        for (int i = 0; i < Ni; i++) {
            // temp variables
            f1_ = FR1(ix,iy,i);
            f2_ = (F_G(ix,iy,i) + (dt/(2*tauf))*f1_)/(1.0 + dt/(2*tauf));    //calc to temp variable first, for use later

            FR2(ix,iy,i) = f2_;        // save to global

            // temp variables
            edotu = ex[i]*ux + ey[i]*uy;

            HR2(ix,iy,i) = (H_G(ix,iy,i) - (dt/2.0)*(edotu/tauhf)*(f1_ - f2_) + (dt/2.0)*(HR1(ix,iy,i)/tauh))/(1.0 + dt/(2.0*tauh));
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK3_STEP3
////////////////////////////////////////////////////////////////////////////////

// macros for RK3
#define FR3(x,y,i) fr3[x*Ny*Ni + y*Ni + i]
#define HR3(x,y,i) hr3[x*Ny*Ni + y*Ni + i]
__kernel void
RK3_STEP3(__global double* f_G,
    __global double* fr2,
    __global double* fr3,
    __global double* h_G,
    __global double* hr2,
    __global double* hr3,
    __global int* cell,
    __global int* bnd,
    __global double* therm,
    __global double* velX,
    __global double* velY,
    __global double* rho_G,
    __global double* rho_3,
    __global double* ux_G,
    __global double* ux_3,
    __global double* uy_G,
    __global double* uy_3,
    __global double* T_G,
    __global double* T_3,
    __global double* fr2_flux_x,
    __global double* fr2_flux_y,
    __global double* hr2_flux_x,
    __global double* hr2_flux_y)
{
    //perform RK3 stepping

    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    // macroscopic properties
    double rho2 = RHO_G(ix,iy);
    double ux2 = UX_G(ix,iy);
    double uy2 = UY_G(ix,iy);
    double T2 = T_G(ix,iy);

    clMacroProp3(f_G, h_G, rho_3, ux_3, uy_3, T_3, fr2_flux_x, fr2_flux_y, hr2_flux_x, hr2_flux_y);

    double rho3 = RHO_3(ix,iy);
    double ux3 =  UX_3(ix,iy);
    double uy3 =  UY_3(ix,iy);
    double T3 =   T_3(ix,iy);

    // calculate relaxation times from macroscopic properties
    double tauf2 = mu/(rho2*R*T2);
    double tauh2 = tauf2/Pr;
    double tauhf2 = (tauh2*tauf2)/(tauf2 - tauh2);

    double tauf3 = mu/(rho3*R*T3);
    double tauh3 = tauf3/Pr;
    double tauhf3 = (tauh3*tauf3)/(tauf3 - tauh3);
    double feq2[Ni];
    double heq2[Ni];

    double feq3[13];
    double heq3[13];

    clEq2D(rho2, ux2, uy2, T2, feq2, heq2);
    clEq2D(rho3, ux3, uy3, T3, feq3, heq3);

    int id = BND(ix,iy);
    int type = CELL(id);

    double flux_f2, flux_h2, edotu2, edotu3, f2_, f3_;

    for (int i = 0; i < Ni; i++) {
        // calculate the updated distribution functions
        if (type > 0) {
            FR3(ix,iy,i) = F_G(ix,iy,i);
            HR3(ix,iy,i) = H_G(ix,iy,i);
        }
        else {
            flux_f2 = clCombineFLUX(fr2_flux_x, fr2_flux_y, i);

            f2_ = FR2(ix,iy,i);

            f3_ = (F_G(ix,iy,i) - flux_f2 + (dt/(2*tauf2))*(feq2[i] - f2_) + (dt/(2.0*tauf3))*feq3[i])/(1 + dt/(2*tauf3));

            FR3(ix,iy,i) = f3_;

            flux_h2 = clCombineFLUX(hr2_flux_x, hr2_flux_y, i);

            edotu2 = ex[i]*ux2 + ey[i]*uy2;
            edotu3 = ex[i]*ux3 + ey[i]*uy3;

            HR3(ix,iy,i) = (H_G(ix,iy,i) - flux_h2 - (dt/2.0)*(edotu2*((feq2[i] - f2_)/tauhf2) + edotu3*((feq3[i] - f3_)/tauhf3)) + 
                (dt/2.0)*((heq2[i] - HR2(ix,iy,i))/tauh2) + (dt/2.0)*(heq3[i]/tauh3))/(1.0 + dt/(2.0*tauh3));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK3_COMBINE
////////////////////////////////////////////////////////////////////////////////

__kernel void
RK3_COMBINE(__global double* f_G,
    __global double* fr2,
    __global double* fr3,
    __global double* h_G,
    __global double* hr2,
    __global double* hr3,
    __global int* cell,
    __global int* bnd,
    __global double* therm,
    __global double* velX,
    __global double* velY,
    __global double* rho_G,
    __global double* rho_3,
    __global double* ux_G,
    __global double* ux_3,
    __global double* uy_G,
    __global double* uy_3,
    __global double* T_G,
    __global double* T_3,
    __global double* fr2_flux_x,
    __global double* fr3_flux_x,
    __global double* fr2_flux_y,
    __global double* fr3_flux_y,
    __global double* hr2_flux_x,
    __global double* hr3_flux_x,
    __global double* hr2_flux_y,
    __global double* hr3_flux_y)
{
    //perform RK3 combination step

    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    // macroscopic properties
    double rho2 = RHO_G(ix,iy);
    double ux2 = UX_G(ix,iy);
    double uy2 = UY_G(ix,iy);
    double T2 = T_G(ix,iy);

    double rho3 = RHO_3(ix,iy);
    double ux3 =  UX_3(ix,iy);
    double uy3 =  UY_3(ix,iy);
    double T3 =   T_3(ix,iy);

    // calculate relaxation times from macroscopic properties
    double tauf2 = mu/(rho2*R*T2);
    double tauh2 = tauf2/Pr;
    double tauhf2 = (tauh2*tauf2)/(tauf2 - tauh2);

    double tauf3 = mu/(rho3*R*T3);
    double tauh3 = tauf3/Pr;
    double tauhf3 = (tauh3*tauf3)/(tauf3 - tauh3);
    double feq2[Ni];
    double heq2[Ni];

    double feq3[13];
    double heq3[13];

    clEq2D(rho2, ux2, uy2, T2, feq2, heq2);
    clEq2D(rho3, ux3, uy3, T3, feq3, heq3);

    int id = BND(ix,iy);
    int type = CELL(id);

    double flux_f2, flux_h2, flux_f3, flux_h3, edotu2, edotu3;
    double f2_, f3_, h2_, h3_;

    for (int i = 0; i < Ni; i++) {
        // --- RK COMBINATION ----
        // calculate the updated distribution functions
        if (type > 0) {
            continue;
        }
        else {
            f2_ = FR2(ix,iy,i);
            f3_ = FR3(ix,iy,i);

            flux_f2 = clCombineFLUX(fr2_flux_x, fr2_flux_y, i);
            flux_f3 = clCombineFLUX(fr3_flux_x, fr3_flux_y, i);

            F_G(ix,iy,i) = F_G(ix,iy,i) - (1.0/2.0)*(flux_f2 + flux_f3) + (dt/2.0)*((feq2[i] - f2_)/tauf2 + (feq3[i] - f3_)/tauf3);

            h2_ = HR2(ix,iy,i);
            h3_ = HR3(ix,iy,i);

            flux_h2 = clCombineFLUX(hr2_flux_x, hr2_flux_y, i);
            flux_h3 = clCombineFLUX(hr3_flux_x, hr3_flux_y, i);

            edotu2 = ex[i]*ux2 + ey[i]*uy2;
            edotu3 = ex[i]*ux3 + ey[i]*uy3;

            H_G(ix,iy,i) = H_G(ix,iy,i) - 0.5*(flux_h2 + flux_h3) + (dt/2.0)*((heq2[i] - h2_)/tauh2 + (heq3[i] - h3_)/tauh3) 
                - (dt/2.0)*((edotu2/tauhf2)*(feq2[i] - f2_) + (edotu3/tauhf3)*(feq3[i] - f3_));
        }
    }
    // update macro properties
    clMacroProp(f_G, h_G, rho_G, ux_G, uy_G, T_G);
}
