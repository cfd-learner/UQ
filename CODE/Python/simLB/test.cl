// test openCL program

#define Ni 13

#define F(x,y,i) f_D[x*Ni*Ny + y*Ni + i]
#define H(x,y,i) h_D[x*Ni*Ny + y*Ni + i]

__kernel void
test(__global float* f_D, __global float* h_D, int const Nx, int const Ny)
{
    
    // global index
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    
    for (int i = 0; i < Ni; i++) {
        F(ix,iy,i) = (float)(iy + ix);
        H(ix,iy,i) = (float)(i);
    }
}