//MACROS 
#define F(x,y,i) f_D[x*Ni*Ny + y*Ni + i] 
#define H(x,y,i) h_D[x*Ni*Ny + y*Ni + i] 
#define BRAY(x,y) bndArray_D[x*Ny + y] 
#define DENSITY(i) density_D[i] 
#define VELX(i) velocityX_D[i] 
#define VELY(i) velocityY_D[i] 
#define TEMP(i) temp_D[i] 
#define RHO(x,y) rho_D[x*Ny + y] 
#define UX(x,y) ux_D[x*Ny + y] 
#define UY(x,y) uy_D[x*Ny + y] 
#define T(x,y) T_D[x*Ny + y] 

//CONSTANTS 
#define Nx 50
#define Ny 50
#define Ni 13 

__constant float R = 287.0
__constant float K = 3.0
__constant float Tc = 303.0

__constant float ex[13] = {0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0, 2.0, 0.0, -2.0, 0.0}
__constant float ey[13] = {0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 2.0, 0.0, -2.0}

// DEVICE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
// INLINE
////////////////////////////////////////////////////////////////////////////////
// equilibrium functions for f

float f0(float rho, float u, float v, float T)
{
   //rest particle equilibrium function
    return (rho/4.0)*(4 + 10*pow(T,2) + pow(u,4) - 5*pow(v,2) + pow(v,4) + 10*T*(-1 + pow(u,2) + pow(v,2)) + pow(u,2)*(-5 + 4*pow(v,2)));
}

float f1(float rho, float u, float v, float T)
{ 
    return (rho/6.0)*(-6*pow(T,2) - u*(1 + u)*(-4 + pow(u,2) + 3*pow(v,2)) - T*(-4 + 6*u + 9*pow(u,2) + 3*pow(v,2)));
}

float f5(float rho, float u, float v, float T)
{
    return (rho/4.0)*((T + u + pow(u,2))*(T + v + pow(v,2)));
}

float f9(float rho, float u, float v, float T)
{
    return (rho/24.0)*(3*pow(T,2) + (-1 + u)*u*(1 + u)*(2 + u) + T*(-1 + 6*u*(1 + u)));
}

// equilibrium functions for h

float h0(float K, float T, float u, float v)
{
return (10*(16 + 3*K)*pow(T,3) + 3*T*(8 + 4*K - 40*pow(u,2) - 5*K*pow(u,2) + 20*pow(u,4)
    + K*pow(u,4) + (-5*(8 + K) + 4*(15 + K)*pow(u,2))*pow(v,2) + (20 + K)*pow(v,4))
    + 30*pow(T,2)*(-4 + 9*pow(u,2) + 9*pow(v,2) + K*(-1 + pow(u,2) + pow(v,2)))
    + 3*(pow(u,2) + pow(v,2))*(4 + pow(u,4) - 5*pow(v,2) + pow(v,4) + pow(u,2)*(-5 + 4*pow(v,2))))/24.0;
}

float h1(float K, float T, float u, float v)
{
    return (-2*(16 + 3*K)*pow(T,3) - u*(1 + u)*(pow(u,2) + pow(v,2))*(-4 + pow(u,2) + 3*pow(v,2))
           - T*(u*(-4*(4 + K) - 4*(7 + K)*u + (14 + K)*pow(u,2) + (19 + K)*pow(u,3)) + (-4 + 3*u*(10
            + K + (14 + K)*u))*pow(v,2) + 3*pow(v,4)) - pow(T,2)*(-16 + 6*u*(6 + 13*u) + 30*pow(v,2)
            + K*(-4 + 6*u + 9*pow(u,2) + 3*pow(v,2))))/12.0;
}

float h5(float K, float T, float u, float v)
{
    return ((16 + 3*K)*pow(T,3) + 3*u*(1 + u)*v*(1 + v)*(pow(u,2) + pow(v,2)) + 3*pow(T,2)*((6 + K)*u
            + (9 + K)*pow(u,2) + v*(6 + K + (9 + K)*v)) + 3*T*(pow(u,3) + pow(u,4) + pow(v,3)*(1 + v)
            + u*v*(6 + K + (9 + K)*v) + pow(u,2)*v*(9 + K + (12 + K)*v)))/24.0;
}

float h9(float K, float T, float u, float v)
{
    return ((16 + 3*K)*pow(T,3) + T*u*(-8 + K*(-1 + u)*(1 + u)*(2 + u) + u*(-7 + 2*u*(11 + 8*u)))
            + T*(-1 + 6*u*(1 + u))*pow(v,2) + (-1 + u)*u*(1 + u)*(2 + u)*(pow(u,2) + pow(v,2))
            + pow(T,2)*(-4 + 36*u + 51*pow(u,2) + K*(-1 + 6*u*(1 + u)) + 3*pow(v,2)))/48.0;
}

////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////
int clEq2D(float rho, float u_, float v_, float T_, float* eqf, float* eqh)
{
    //returns the equilibrium values for each velocity vector given the current
    // macroscopic values and the corresponding reference quantities

    float sRTc = sqrt(R*Tc);
    float u = u_/sRTc;
    float v = v_/sRTc;
    float T = T_/Tc;

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

    float rRTc = rho*R*Tc;

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
    float sum_f = 0;
    float sum_h = 0;
    for (int i = 0; i < 13; i++) {
        sum_f += eqf[i];
        sum_h += eqh[i];
    }
    float U = sqrt(u_*u_ + v_*v_);
    float E = (U*U + (K+2.0)*R*T_)/2.0;

    float diff1 = sum_h/rho - E;
    float diff2 = sum_f - rho;

    int errEq;

    if (diff1 > 1 || diff2 > 1) {
            errEq = 1;
    }
    else {
        errEq = 0;
    }
    return errEq;
}

void 
 macroPropShort(__global float* f_D,
    __global float* h_D,
    __global float* rho_D,
    __global float* ux_D,
    __global float* uy_D,
    __global float* T_D)
{
    // calculate macroscopic properties

    // global index
    int ix = get_global_id(0);
    int iy = get_global_id(1);

    float rho = 0.0;
    float rho_ux = 0.0;
    float rho_uy = 0.0;
    float sum_h = 0.0;
    float ux, uy, T_;

    float RTc = sqrt(R*Tc);

    for (int i = 0; i < Ni; i++) {
        rho += F(ix,iy,i);
        rho_ux += F(ix,iy,i)*RTc*ex[i];
        rho_uy += F(ix,iy,i)*RTc*ey[i];
        sum_h += H(ix,iy,i);
    }
    ux = rho_ux/rho;
    uy = rho_uy/rho;

    float ux_abs, uy_abs, usq;

    ux_abs = fabs(ux);
    uy_abs = fabs(uy);
    usq = ux_abs*ux_abs + uy_abs*uy_abs;
    T_ = 2.0*(sum_h/rho - usq/2.0)/(input->b*input->R);
    // assign variables to array
    RHO(ix,iy) = rho;
    UX(ix,iy) = ux;
    UY(ix,iy) = uy;
    T(ix,iy) = T_;
}

// KERNEL FUNCTION
__kernel void
initFunctions(__global float* f_D,
    __global float* h_D,
    __global int* bndArray_D,
    __global float* density_D,
    __global float* velocityX_D,
    __global float* velocityY_D,
    __global float* temp_D,
    __global float* rho_D,
    __global float* ux_D,
    __global float* uy_D,
    __global float* T_D) {
    // initialise functions

    // global index
    int ix = get_global_id(0);
    int iy = get_global_id(1);

    int id = BRAY(ix,iy);

    float rho = DENSITY(id);
    float ux = VELX(id);
    float uy = VELY(id);
    float T_ = TEMP(id);

    float eqf[Ni];
    float eqh[Ni];

    // compute equilibrium functions
    //clEq2D(rho, ux, uy, T_, input->Tc, input->R, input->K, eqf, eqh);
    //clEq2D(rho, ux, uy, T_, 606.0f, 287.0f, 3.0f, eqf, eqh);

    for (int i = 0; i < Ni; i++) {
        F(ix,iy,i) = ix;//eqf[i];
        H(ix,iy,i) = iy;//eqh[i];
    }

    //macroPropShort(f_D, h_D, rho_D, ux_D, uy_D, T_D, input);
}