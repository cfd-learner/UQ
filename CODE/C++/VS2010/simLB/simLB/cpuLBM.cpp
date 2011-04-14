#include "cpuLBM.h"


using namespace std;


cpuLBM::cpuLBM(folderSystem* files)
{

	saveToMAT = files->resultsMatPath;
	//lbm->saveToTEC = files->resultsTecPath;

	loadInput(files);

	// various visualistion initialisations
	display = "rho";	// set visualiser to density
	pause = 0;
	stop = 0;
	residual = 1;
}

cpuLBM::~cpuLBM()
{
	// destructor
	delete density;
	delete pressure;
	delete temp;
	delete velocityX;
	delete velocityY;
	delete cellType;
	delete bndArray;

	delete f_G;
	delete h_G;

	delete rho_G;
	delete u_G;
	delete p_G;
	delete T_G;
}

//////////////////////////////
// LOAD DATA & INITIALIZE
//////////////////////////////

int cpuLBM::loadInput (folderSystem* files)
{
	// load data from a specially formatted .mat file

	// .mat variables
	MATFile *pmat;
	const char **dir;
	const char *name;
	int	  ndir;
	int	  i;
	mxArray *pa;

	const char * file = files->inputFilePath.c_str();

	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		printf("Error opening file %s\n", file);
		return 1;
	}

	//get directory of MAT-file
	dir = (const char **)matGetDir(pmat, &ndir);
	if (dir == NULL) {
		printf("Error reading directory of file %s\n", file);
		return(1);
	} else {
		printf("Directory of %s:\n", file);
		for (i=0; i < ndir; i++)
			printf("%s\n",dir[i]);
	}
	mxFree(dir);

	/* In order to use matGetNextXXX correctly, reopen file to read in headers. */
	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n",file);
		return(1);
	}
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		printf("Error reopening file %s\n", file);
		return(1);
	}

	/* Read in each array. */
	printf("\nReading in the array contents:\n");
	for (i=0; i<ndir; i++) {
		pa = matGetNextVariable(pmat, &name);
		if (pa == NULL) {
			printf("Error reading in file %s\n", file);
			return(1);
		} 

		// get number of dimensions and size of each dimension
		size_t nDims = mxGetNumberOfDimensions(pa);
		const mwSize* dims = mxGetDimensions(pa);

		printf("According to its contents, array %s has %d dimensions\n",
			name, nDims);

		// go through list and save to correct variable
		double* ptr = mxGetPr(pa);

		string name_str;

		name_str.append(name);

		// overall simulation data items
		if (name_str == "RKMETHOD") {
			RKMETHOD = (int) ptr[0];
		}
		else if ( name_str == "FMETHOD") {
			FMETHOD = (int) ptr[0];
		}
		else if (name_str == "CFL") {
			CFL = (double) ptr[0];
		}
		else if (name_str == "dtau") {
			dtau = (double) ptr[0];
		}
		else if (name_str == "PROCESSOR") {
			PROCESSOR = (int) ptr[0];
		}
		else if (name_str == "tt_tref") {
			tt_tref = (double) ptr[0];
		}
		else if (name_str == "tt_time") {
			tt_time = (double) ptr[0];
		}
		else if (name_str == "steps") {
			steps = (int) ptr[0];
		}
		else if (name_str == "tol") {
			tol = (double) ptr[0];
		}
		else if (name_str == "periodicX") {
			periodicX = (int) ptr[0];
		}
		else if (name_str == "periodicY") {
			periodicY = (int) ptr[0];
		}
		else if (name_str == "mirrorNorth") {
			mirrorNorth = (int) ptr[0];
		}
		else if (name_str == "mirrorSouth") {
			mirrorSouth = (int) ptr[0];
		}
		else if (name_str == "mirrorEast") {
			mirrorEast = (int) ptr[0];
		}
		else if (name_str == "mirrorWest") {
			mirrorWest = (int) ptr[0];
		}
		else if (name_str == "gamma") {
			gamma = (double) ptr[0];
		}
		else if (name_str == "R") {
			R = (double) ptr[0];
		}
		else if (name_str == "mu") {
			mu = (double) ptr[0];
		}
		else if (name_str == "Pr") {
			Pr = (double) ptr[0];
		}
		else if (name_str == "dx") {
			dx = (double) ptr[0];
		}
		else if (name_str == "dy") {
			dy = (double) ptr[0];
		}
		else if (name_str == "Lx") {
			Lx = (double) ptr[0];
		}
		else if (name_str == "Ly") {
			Ly = (double) ptr[0];
		}
		else if (name_str == "rho_ref") {
			rho_ref = (double) ptr[0];
		}
		else if (name_str == "p_ref") {
			p_ref = (double) ptr[0];
		}
		else if (name_str == "T_ref") {
			T_ref = (double) ptr[0];
		}
		else if (name_str == "Tc_Tref") {
			Tc_Tref = (double) ptr[0];
		}
		else if (name_str == "nPrintOut") {
			nPrintOut = (int) ptr[0];
		}
		else if (name_str == "MAT") {
			MAT = (int) ptr[0];
		}

		// solution domain variables
		else if (name_str == "numProps") {
			numProps = (int) ptr[0];
		}

		else if (name_str == "density") {
			size_t nItems = max(dims[0],dims[1]);
			density = new double[nItems];
			for (int j = 0; j < nItems; j++) {
				density[j] = (double) ptr[j];
			}
		}

		else if (name_str == "pressure") {
			size_t nItems = max(dims[0],dims[1]);
			pressure = new double[nItems];
			for (int j = 0; j < nItems; j++) {
				pressure[j] = (double) ptr[j];
			}
		}

		else if (name_str == "temp") {
			size_t nItems = max(dims[0],dims[1]);
			temp = new double[nItems];
			for (int j = 0; j < nItems; j++) {
				temp[j] = (double) ptr[j];
			}
		}

		else if (name_str == "velocityX") {
			size_t nItems = max(dims[0],dims[1]);
			velocityX = new double[nItems];
			for (int j = 0; j < nItems; j++) {
				velocityX[j] = (double) ptr[j];
			}
		}

		else if (name_str == "velocityY") {
			size_t nItems = max(dims[0],dims[1]);
			velocityY = new double[nItems];
			for (int j = 0; j < nItems; j++) {
				velocityY[j] = (double) ptr[j];
			}
		}

		else if (name_str == "cellType") {
			size_t nItems = max(dims[0],dims[1]);
			cellType = new int[nItems];
			for (int j = 0; j < nItems; j++) {
				cellType[j] = (int) ptr[j];
			}
		}

		else if (name_str == "bndArray") {
			Nx = (int) dims[1];
			Ny = (int) dims[0];
			bndArray = new int[Nx*Ny];
			int y_ = Ny;
			for (int y = 0; y < Ny; y++) {
				y_--;
				for (int x = 0; x < Nx; x++) {
					int val = (int) ptr[x*Ny + y_];
					BNDARRAY(x,y) = val;
				}
			}
		}
		mxDestroyArray(pa);
		}

	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n",file);
		return(1);
	}

	
	refineInput();
	initSim();

	printf("Done\n");
	return(0);
}

void cpuLBM::refineInput()
{
	// make sure all data values are as they should be
	// update all items in the tables as well

	//////////////////////////
	// define dx & dy based upon what has been supplied

	if (dx == 0)
	{
		dx = Lx/((double)Nx - 1);
	}
	else
	{
		Lx = dx*(Nx - 1);
	}

	if (dy == 0 && Ly != 0)
	{
		dy = Ly/((double)Ny - 1);
	}
	else if (dy == 0 && Ly == 0)
	{
		dy = dx;
		Ly = dy*(Ny - 1);
	}
	else
	{
		Ly = dy*(Ny - 1);
	}

	//////////////////////////
	// define reference values

	if (rho_ref == 0)
	{
		rho_ref = p_ref/(R*T_ref);
	}
	else if (p_ref == 0)
	{
		p_ref = rho_ref*R*T_ref;
	}
	else if (T_ref == 0)
	{
		T_ref = p_ref/(rho_ref*R);
	}

	//////////////////////////
	// update flow properties

	double rho, p, T;

	for (int r = 0; r < numProps; r++){
		rho = density[r];
		p = pressure[r];
		T = temp[r];


		// set unknown values
		if (rho == 0)
		{
			rho = p/(R*T);
			density[r] = rho;
		}
		else if (p == 0)
		{
			p = rho*R*T;
			pressure[r] = p;
		}
		else if (T == 0)
		{
			T = p/(rho*R);
			temp[r] = T;
		}
	}
}

void cpuLBM::setupVars()
{
	// setup the variables as required

	f_G = new double[Nx*Ny*Ni];
	h_G = new double[Nx*Ny*Ni];

	rho_G = new double[Nx*Ny];
	u_G = new double2[Nx*Ny];
	p_G = new double[Nx*Ny];
	T_G = new double[Nx*Ny];

	res_copy = new double[Nx*Ny];


	//lattice variables - velocity vectors, 2D
	e[0].x = 0;		e[0].y = 0;
	e[1].x = 1;		e[1].y = 0;
	e[2].x = 0;		e[2].y = 1;
	e[3].x = -1;	e[3].y = 0;
	e[4].x = 0;		e[4].y = -1;
	e[5].x = 1;		e[5].y = 1;
	e[6].x = -1;	e[6].y = 1;
	e[7].x = -1;	e[7].y = -1;
	e[8].x = 1;		e[8].y = -1;
	e[9].x = 2;		e[9].y = 0;
	e[10].x = 0;	e[10].y = 2;
	e[11].x = -2;	e[11].y = 0;
	e[12].x = 0;	e[12].y = -2;

	//compressible gas variables
	D = 2;					//dimensionality of simulation
	b = 2/(gamma - 1);		//constant to give correct gamma value
	K = b - D;				// internal degrees of freedom

	//reference values
	Tc = Tc_Tref*T_ref;		// characteristic temperature, K
	tau_ref = mu/p_ref;		// relaxation time, s
	u_ref = sqrt(R*T_ref);	// reference velocity
	t_ref = MAX(Lx,Ly)/u_ref;		// reference time

	//simulation variables
	c = sqrt(R*Tc);	// sound speed, m/s

	for (int i = 0; i < Ni; i++)
	{
		e[i].x *= c;

		e[i].y *= c;
	}

	//stability requirements

	if (dtau == 0)
	{
		dt = (CFL*MIN(dx,dy)) / e[9].x;
	}
	else
	{
		dt = dtau*tau_ref;
	}

	// simulation parameters
	ttotal = 0;	// total time in seconds that the sim is to run

	// rely on user to only intput one variable to determine number of steps
	int need_steps = 0;
	if (tt_tref > 0)
	{
		ttotal = tt_tref*t_ref;
		need_steps = 1;
	}
	else if (tt_time > 0)
	{
		ttotal = tt_time;
		need_steps = 1;
	}
	else if (steps > 0)
	{
		nsteps = steps;
	}
	else if (tol > 0)
	{
		nsteps = 0;
	}

	if (need_steps == 1) {
		nsteps = int(ceil(ttotal / dt));
	}

}

void cpuLBM::initSim()
{
	setupVars();
	initialiseFunctions();

	step = 0;
	tstep = 0;

	// write initial conditions to file
	createMAT("initCond");
}
//////////////////////////////
// SAVE DATA
//////////////////////////////

int cpuLBM::createMAT(string name, int DF)
{
	// create a .mat file of the current state of the simulation

	string *tempFile = new string;

	tempFile->append(saveToMAT);
	tempFile->append("/");
	tempFile->append(name);
	tempFile->append(".mat");

	const char * file    = tempFile->c_str();


	MATFile *pmat;

	//printf("Creating file %s...\n\n", file);
	pmat = matOpen(file, "w");
	if (pmat == NULL) {
		printf("Error creating file %s\n", file);
		printf("(Do you have write permission in this directory?)\n");
		return(EXIT_FAILURE);
	}

	// convert u into ux & uy
	double* ux = new double[Nx*Ny];
	double* uy = new double[Nx*Ny];

	for (int x = 0; x < Nx; x++)
	{
		for (int y= 0; y < Ny; y++)
		{
			ux[Ny*x + y] = UX(x,y);
			uy[Ny*x + y] = UY(x,y);
		}
	}

	// calc time stamp
	double dtt[1] = {tstep};

	double DX[1] = {dx};
	double DY[1] = {dy};

	// data stuff
	mxArray *m_rho, *m_ux, *m_uy, *m_p, *m_T, *time, *dx_, *dy_;
	int status;
	size_t size_data = Nx*Ny*sizeof(double);

	const mwSize dims[2] = {Ny,Nx};
	const mwSize dims2[2] = {1,1};

	// output of distribution functions if called for
	if (DF != 0) {
		mxArray *m_f, *m_h, *m_ex, *m_ey;

		const mwSize dims3[3] = {13,Ny,Nx};
		const mwSize dims13[2] = {13,1};
		size_t size_df = Nx*Ny*13*sizeof(double);
		size_t size_e = 13*sizeof(double);

		// f
		m_f = mxCreateNumericArray(3,dims3,mxDOUBLE_CLASS,mxREAL);
		if (m_f == NULL) 
		{
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void *)(mxGetPr(m_f)), (void *)f_G, size_df);

		status = matPutVariable(pmat, "f", m_f);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}

		// h
		m_h = mxCreateNumericArray(3,dims3,mxDOUBLE_CLASS,mxREAL);
		if (m_h == NULL) 
		{
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void *)(mxGetPr(m_h)), (void *)h_G, size_df);

		status = matPutVariable(pmat, "h", m_h);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}

		// e

		// convert e into ex & ey
		double ex[13];
		double ey[13];

		for (int i = 0; i < 13; i++) {
			ex[i] = e[i].x;
			ey[i] = e[i].y;
		}

		// ex
		m_ex = mxCreateNumericArray(2,dims13,mxDOUBLE_CLASS,mxREAL);
		if (m_ex == NULL) 
		{
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void *)(mxGetPr(m_ex)), (void *)ex, size_e);

		status = matPutVariable(pmat, "ex", m_ex);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}

		// ey
		m_ey = mxCreateNumericArray(2,dims13,mxDOUBLE_CLASS,mxREAL);
		if (m_ey == NULL) 
		{
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void *)(mxGetPr(m_ey)), (void *)ey, size_e);

		status = matPutVariable(pmat, "ey", m_ey);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}

		mxDestroyArray(m_f);
		mxDestroyArray(m_h);
		mxDestroyArray(m_ex);
		mxDestroyArray(m_ey);
	}

	// rho
	m_rho = mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
	if (m_rho == NULL) 
	{
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}

	memcpy((void *)(mxGetPr(m_rho)), (void *)rho_G, size_data);

	status = matPutVariable(pmat, "rho", m_rho);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}

	// u_x
	m_ux = mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
	if (m_ux == NULL) {
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}

	memcpy((void *)(mxGetPr(m_ux)), (void *)ux, size_data);

	status = matPutVariable(pmat, "ux", m_ux);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}

	// u_y

	m_uy = mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
	if (m_uy == NULL) {
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}

	memcpy((void *)(mxGetPr(m_uy)), (void *)uy, size_data);

	status = matPutVariable(pmat, "uy", m_uy);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}

	// p

	m_p = mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
	if (m_p == NULL) {
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}

	memcpy((void *)(mxGetPr(m_p)), (void *)p_G, size_data);

	status = matPutVariable(pmat, "p", m_p);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}

	// T

	m_T = mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
	if (m_T == NULL) {
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}

	memcpy((void *)(mxGetPr(m_T)), (void *)T_G, size_data);

	status = matPutVariable(pmat, "T", m_T);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}

	// time

	time = mxCreateNumericArray(2,dims2,mxDOUBLE_CLASS,mxREAL);
	if (time == NULL) {
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}

	memcpy((void *)(mxGetPr(time)), (void *)dtt, sizeof(double));

	status = matPutVariable(pmat, "t", time);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}

	// dx

	dx_ = mxCreateNumericArray(2,dims2,mxDOUBLE_CLASS,mxREAL);
	if (dx_ == NULL) {
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}

	memcpy((void *)(mxGetPr(dx_)), (void *)DX, sizeof(double));

	status = matPutVariable(pmat, "dx", dx_);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}

	// dy

	dy_ = mxCreateNumericArray(2,dims2,mxDOUBLE_CLASS,mxREAL);
	if (dy_ == NULL) {
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__); 
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}

	memcpy((void *)(mxGetPr(dy_)), (void *)DY, sizeof(double));

	status = matPutVariable(pmat, "dy", dy_);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}
	// clean up

	delete [] ux;
	delete [] uy;
	delete tempFile;

	mxDestroyArray(m_rho);
	mxDestroyArray(m_ux);
	mxDestroyArray(m_uy);
	mxDestroyArray(m_p);
	mxDestroyArray(m_T);
	mxDestroyArray(time);
	mxDestroyArray(dx_);
	mxDestroyArray(dy_);


	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n",file);
		return(EXIT_FAILURE);
	}
	return 0;
}
//////////////////////////////
// GENERAL FUNCTIONS
//////////////////////////////
void cpuLBM::updateStepCount()
{
	step += 1;
	tstep = step*dt;
}
double cpuLBM::dot(double a1, double a2, double b1, double b2)
{
	// perform the dot product between two (2,1) vectors
	return a1*b1 + a2*b2;
}
//////////////////////////////
// SIMULATION FUNCTIONS
//////////////////////////////

void cpuLBM::equilibrium2D(double rho, double u_, double v_, double T_, double eqf[], double eqh[])
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
	//double sum_h = 0;
	for (int i = 0; i < Ni; i++)
	{
		sum_f += eqf[i];
		//sum_h += eqh[i];
	}
	//double U = sqrt(u_*u_ + v_*v_);
	//double E = (U*U + b*R*T_)/2.0;

	//double diff1 = sum_h/rho - E;
	double diff2 = sum_f - rho;

	if (abs(diff2) > 1e-6)
	{
		printf("\n error: equilibrium stuffed up");
		hasFailed = 1;
	}
}

void cpuLBM::initialiseFunctions()
{
	// initialise all distribution functions
	double eqf[Ni];
	double eqh[Ni];

	for (int y = 0; y < Ny; y++)
	{
		for (int x = 0; x < Nx; x++)
		{
			int id = BNDARRAY(x,y);

			double rho = density[id];
			double ux = velocityX[id];
			double uy = velocityY[id];
			double T = temp[id];

			equilibrium2D(rho,ux,uy,T,eqf,eqh);

			for (int i = 0; i < Ni; i++)
			{
				F(x,y,i) = eqf[i];
				H(x,y,i) = eqh[i];
			}

			// calculate macroscopic properties from the distribution functions
			macroPropShort(x,y);
		}
	}
}

void cpuLBM::macroPropShort(int x, int y)
{
	//calculate the macroscopic properties of a node (x,y)
	double rho = 0;
	double rho_ux = 0;
	double rho_uy = 0;
	double sum_h = 0;
	double ux, uy, T_, p;

	for (int i = 0; i < Ni; i++)
	{
		rho += F(x,y,i);
		rho_ux += F(x,y,i)*e[i].x;
		rho_uy += F(x,y,i)*e[i].y;
		sum_h += H(x,y,i);
	}

	ux = rho_ux/rho;
	uy = rho_uy/rho;

	double ux_abs, uy_abs, usq;

	ux_abs = abs(ux);
	uy_abs = abs(uy);

	usq = ux_abs*ux_abs + uy_abs*uy_abs;

	T_ = 2.0*(sum_h/rho - usq/2.0)/(b*R);

	p = rho*R*T_;

	// assign variables to array
	RHO(x,y) = rho;
	UX(x,y) = ux;
	UY(x,y) = uy;
	T(x,y) = T_;
	P(x,y) = p;
}

void cpuLBM::macroProp(int x, int y, double* rho, double2* u, double* T, 
						double* p, double* tauf, double* tauh, double* tauhf)
{
	//calculate the macroscopic properties of a node (x,y)
	double rho_ = 0;
	double rho_ux = 0;
	double rho_uy = 0;
	double sum_h = 0;
	double ux, uy, T_, p_;

	for (int i = 0; i < Ni; i++)
	{
		rho_ += F(x,y,i);
		rho_ux += F(x,y,i)*e[i].x;
		rho_uy += F(x,y,i)*e[i].y;
		sum_h += H(x,y,i);
	}

	ux = rho_ux/rho_;
	uy = rho_uy/rho_;

	double ux_abs, uy_abs, usq;

	ux_abs = abs(ux);
	uy_abs = abs(uy);

	usq = ux_abs*ux_abs + uy_abs*uy_abs;

	T_ = 2.0*(sum_h/rho_ - usq/2.0)/(b*R);

	p_ = rho_*R*T_;

	double tauf_;
	double tauh_;
	double tauhf_;

	tauf_ = mu/p_;
	tauh_ = tauf_/Pr;

	tauhf_ = (tauh_*tauf_)/(tauf_ - tauh_);

	// assign variables to array
	RRHO(x,y) = rho_;
	UUX(x,y) = ux;
	UUY(x,y) = uy;
	TT(x,y) = T_;
	PP(x,y) = p_;
	TAUF(x,y) = tauf_;
	TAUH(x,y) = tauh_;
	TAUHF(x,y) = tauhf_;

	TAUF(x,y) = tauf_;
	TAUH(x,y) = tauh_;
	TAUHF(x,y) = tauhf_;
}


void cpuLBM::macroProp3(double2* fr2_flux, double2* hr2_flux, int x, int y, double* rho, double2* u, 
						 double* T, double* p, double* tauf, double* tauh, double* tauhf)
{
	//calculate the macroscopic properties of a node

	double rho_ = 0;
	double u_x;
	double u_y;
	double T_;
	double p_;
	double tauf_;
	double tauh_;
	double tauhf_;

	double rho_ux = 0;
	double rho_uy = 0;
	double sum_h = 0;

	double flux_f, flux_h;

	for (int i = 0; i < Ni; i++)
	{
		flux_f = cFLUX(fr2_flux, x, y, i);
		flux_h = cFLUX(hr2_flux, x, y, i);

		rho_ += F(x,y,i) - flux_f;
		rho_ux += F(x,y,i)*e[i].x - flux_f*e[i].x;
		rho_uy += F(x,y,i)*e[i].y - flux_f*e[i].y;
		sum_h += H(x,y,i) - flux_h;
	}

	u_x = rho_ux / rho_;
	u_y = rho_uy / rho_;

	double ux_abs, uy_abs;

	ux_abs = abs(u_x);
	uy_abs = abs(u_y);

	double usq = ux_abs*ux_abs + uy_abs*uy_abs;

	T_ = 2.0*(sum_h/rho_ - usq/2.0)/(b*R);

	p_ = rho_*R*T_;

	tauf_ = mu/p_;
	tauh_ = tauf_/Pr;

	tauhf_ = (tauh_*tauf_)/(tauf_ - tauh_);

	// assign variables to array
	RRHO(x,y) = rho_;
	UUX(x,y) = u_x;
	UUY(x,y) = u_y;
	TT(x,y) = T_;
	PP(x,y) = p_;
	TAUF(x,y) = tauf_;
	TAUH(x,y) = tauh_;
	TAUHF(x,y) = tauhf_;
}
int cpuLBM::index(int i, int pm, int d, int& mir)
{
	// d = 0 -> x
	// d = 1 -> y
	i = i + pm;
	mir = 0;

	if (d == 0)
	{
		if (i < 0)
		{
			if (periodicX > 0)
			{
				i = Nx + i;
			}
			else if (mirrorWest > 0)
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
			if (periodicX > 0)
			{
				i = i - Nx;
			}
			else if (mirrorEast > 0)
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
			if (periodicY > 0)
			{
				i = Ny + i;
			}
			else if (mirrorSouth > 0)
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
			if (periodicY > 0)
			{
				i = i - Ny;
			}
			else if (mirrorNorth > 0)
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
int cpuLBM::sign(double a)
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
double cpuLBM::minmod(double a, double b)
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

void cpuLBM::mirIndex(int i, int mir_x, int mir_y, int& i_x, int& i_y)
{
	// gives the mirrored index about the axes defined by inputs mir_x, and mir_y

	// initialise with no mirror
	i_x = i;
	i_y = i;

	// mirror about x
	if (mir_x == 1)	
	{
		int mi[13] = {0,1,4,3,2,8,7,6,5,9,12,11,10};
		i_x = mi[i];
	}
	
	// mirror about y
	if(mir_y == 1)
	{
		int mi[13] = {0,3,2,1,4,6,5,8,7,11,10,9,12};
		i_y = mi[i];
	}

}

void cpuLBM::stencil(double* f, int x, int y, int i, int lengthS, double* Sx, double* Sy)
{
	// given an array of distribution functions and the coordinate, find a stencil along axis directions given by the chosen velocity

	if ( i == 0)
	{
		return;
	}

	// velocity vector, in integer values for indexing
	int ex = sign(e[i].x);
	int ey = sign(e[i].y);

	int low = (lengthS - 1) / 2;	// max/min num of stencil from centre

	// calculate stencil along x and y

	int jx1 = x - ex*low;	//start of stencil index
	int jy1 = y - ey*low;

	int jx, jy;
	int mir_x, mir_y;
	int i_x, i_y;

	int sld_x = 0;
	int sld_y = 0;
	int* sldx = new int[lengthS];
	int* sldy = new int[lengthS];

	int id, type;

	for (int j = 0; j < lengthS; j++)	// put together stencil along axis lines, start on low side, go to hi side: [j-low <-> j+low]
	{
		jx = index(jx1,ex*j,0,mir_y);
		jy = index(jy1,ey*j,1,mir_x);

		mirIndex(i,mir_x,mir_y,i_x,i_y);	// mirror indexes if required

		id = BNDARRAY(jx,y);
		type = cellType[id];

		// x
		Sx[j] = FF(jx,y,i_y);

		if (type == 2)	// if solid
		{
			sldx[j] = 1;
			sld_x = 1;
		}
		else
		{
			sldx[j] = 0;
		}

		id = BNDARRAY(x,jy);
		type = cellType[id];

		// y
		Sy[j] = FF(x,jy,i_x);

		if (type == 2)
		{
			sldy[j] = 1;
			sld_y = 1;
		}
		else
		{
			sldy[j] = 0;
		}
	}

	// update solid nodes if required

	if (sld_x == 1)
	{
		int diff;	// difference between indexes of sldx
		int x_;		// location of edge solid node, outside coords
		int j_;		// location of edge solid node, stencil coords
		
		// find where edge solid node is
		for (int j = 0; j < lengthS - 1; j++)
		{
			diff = sldx[j] - sldx[j+1];
			if (diff > 0)
			{
				j_ = j;
				x_ = index(jx1,ex*j,0,mir_y);
				break;
			}
			else if (diff < 0)
			{
				j_ = j + 1;
				x_ = index(jx1,ex*(j + 1),0,mir_y);
				break;
			}
		}

		// update solid part of stencil with linear extrapolation of actual stencil

		int xm1 = index(x_,ex*diff,0,mir_y);		// closest fluid node
		int xm2 = index(x_,ex*2*diff,0,mir_y);		// next closest fluid node

		Sx[j_] = 2*FF(xm1,y,i) - FF(xm2,y,i);
		//Sx[j_] = f[i3(xm1,y,i,Ny)];

		int j__ = j_ - diff;

		if (j__ >= 0 && j__ < lengthS) // update next one along if stencil is long enough
		{
			Sx[j__] = Sx[j_];
		}
	}

	if (sld_y == 1)
	{
		int diff;	// difference between indexes of sldy
		int y_;		// location of edge solid node, outside coords
		int j_;		// location of edge solid node, stencil coords
		
		// find where edge solid node is
		for (int j = 0; j < lengthS - 1; j++)
		{
			diff = sldy[j] - sldy[j+1];
			if (diff > 0)
			{
				j_ = j;
				y_ = index(jy1,ey*j,1,mir_x);
				break;
			}
			else if (diff < 0)
			{
				j_ = j + 1;
				y_ = index(jy1,ey*(j + 1),1,mir_x);
				break;
			}
		}

		// update solid part of stencil with linear extrapolation of actual stencil

		int ym1 = index(y_,ey*diff,1,mir_x);		// closest fluid node
		int ym2 = index(y_,ey*2*diff,1,mir_x);		// next closest fluid node

		Sy[j_] = 2*FF(x,ym1,i) - FF(x,ym2,i);
		//Sy[j_] = f[i3(x,ym1,i,Ny)];

		int j__ = j_ - diff;

		if (j__ >= 0 && j__ < lengthS)	// update next one along if stencil is long enough
		{
			Sy[j__] = Sy[j_];
		}
	}

	delete [] sldx;
	delete [] sldy;
}

double cpuLBM::NND(double ee, double* S)
{
	// find positive flux for given stencil, F_(j+1/2), flux going OUT of cell, along x and y
	int n = 1;

	double Fp_I, Fp_Ip1, Fp_Im1;

	double vP = abs(ee);

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

	F_Ip12 = Fp_I + 0.5*minmod(dFp_Ip12,dFp_Im12);

	return F_Ip12;
}

double cpuLBM::WENO5(double ee, double* S)
{
	// calculate the flux term of the WENO5 scheme for one flow direction

	double epsilon = 1e-6;
	double posFlow = abs(ee);
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

		B0p = (13.0/12.0)*pow(Sp[n-2] - 2*Sp[n-1] + Sp[n],2) + (1.0/4.0)*pow(Sp[n-2] - 4*Sp[n-1] + 3*Sp[n],2);
		B1p = (13.0/12.0)*pow(Sp[n-1] - 2*Sp[n] + Sp[n+1],2) + (1.0/4.0)*pow(Sp[n-1] - Sp[n+1],2);
		B2p = (13.0/12.0)*pow(Sp[n] - 2*Sp[n+1] + Sp[n+2],2) + (1.0/4.0)*pow(3*Sp[n] - 4*Sp[n+1] + Sp[n+2],2);

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

void cpuLBM::computeFlux(double* f, int x, int y, int i, double2& flux)
{
	double F_Ip12x,F_Ip12y;

	// choose flux method
	if (FMETHOD == 0)
	{
		double Sx[3];
		double Sy[3];

		stencil(f,x,y,i,3,Sx,Sy);

		F_Ip12x = NND(e[i].x,Sx);
		F_Ip12y = NND(e[i].y,Sy);
	}
	else if (FMETHOD == 1)
	{
		double Sx[5];
		double Sy[5];

		stencil(f,x,y,i,5,Sx,Sy);

		F_Ip12x = WENO5(e[i].x,Sx);
		F_Ip12y = WENO5(e[i].y,Sy);
	}

	// fluxes in component direction
	flux.x = F_Ip12x;
	flux.y = F_Ip12y;
}

double cpuLBM::cFLUX(double2* fluxOut, int x, int y, int i)
{
	// calculates the combined flux given all the fluxes out of each node

	// velocity vector, in integer values for indexing
	int ex = sign(e[i].x);
	int ey = sign(e[i].y);

	int mir_x, mir_y;

	int x_ = index(x,-ex,0,mir_y);
	int y_ = index(y,-ey,1,mir_x);

	double flux;
	double flux_out, flux_in;

	if (i == 0)
	{
		return 0;
	}
	else
	{
		int i_x, i_y;

		mirIndex(i,mir_x,mir_y,i_x,i_y);	// mirror indexes if required

		flux_out = (dt/dx)*FLUXOUTX(x,y,i) + (dt/dy)*FLUXOUTY(x,y,i);

		flux_in = (dt/dx)*FLUXOUTX(x_,y,i_y) + (dt/dy)*FLUXOUTY(x,y_,i_x);

		flux = flux_out - flux_in;
	}

	return flux;
}
void cpuLBM::posFluxes(double* f, double* h, double2* f_flux, double2* h_flux)
{
	// function to calculate all positive fluxes along axis (x & y) lines for fluid nodes

	double2 flux_f, flux_h;
	int id, type;

	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{
			for (int i = 0; i < Ni; i++)
			{
				id = BNDARRAY(x,y);
				type = cellType[id];
				if (type != 2)	// only perform fluxes on fluid nodes
				{
					computeFlux(f,x,y,i,flux_f);
					computeFlux(h,x,y,i,flux_h);

					F_FLUX_X(x,y,i) = flux_f.x;
					F_FLUX_Y(x,y,i) = flux_f.y;

					H_FLUX_X(x,y,i) = flux_h.x;
					H_FLUX_Y(x,y,i) = flux_h.y;
				}
			}
		}
	}
}

void cpuLBM::wallFluxes(double* f, double2* f_flux, double* h, double2* h_flux)
{
	// calculate fluxes into solid nodes and set fluxes out of solid to cancel them out

	int id, type;

	int xm1, ym1, i, inv_i;
	int mir_x, mir_y;

	double flux_inf, flux_outf, alphaf;
	double flux_inh, flux_outh;

	double ux, uy;

	int inx,iny;

	int xx[4] = {-1,1,0,0};
	int yy[4] = {0,0,-1,1};
	int ii[4][4] = {{1,5,8,9},{3,6,7,11},{2,5,6,10},{4,7,8,12}};
	int inv[4][4] = {{3,6,7,11},{1,5,8,9},{4,8,7,12},{2,6,5,10}};
	//int inv[4][4] = {{3,7,6,11},{1,8,5,9},{4,7,8,12},{2,5,6,10}};

	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{
			id = BNDARRAY(x,y);
			type = cellType[id];

			if ( type == 2)	//wall nodes
			{
				flux_inf = 0;
				flux_outf = 0;
				flux_inh = 0;
				flux_outh = 0;

				// check surrounds for fluid node
				for (int j = 0; j < 4; j++)
				{
					xm1 = index(x,xx[j],0,mir_y);
					ym1 = index(y,yy[j],1,mir_x);

					id = BNDARRAY(xm1,ym1);
					type = cellType[id];

					if (type < 2) // if fluid (permanent or initial)
					{
						for (int k = 0; k < 4; k++)
						{
							i = ii[j][k];	// index of velocities into solid
							inv_i = inv[j][k];	// inverted velocities, out of solid

							// switches for turning off velocities that don't impinge on solid, or leave solid, through the cell pointed to by xx & yy
							inx = abs(xx[j]);	
							iny = abs(yy[j]);

							flux_inf += inx*F_FLUX_X(xm1,ym1,i)/dx + iny*F_FLUX_Y(xm1,ym1,i)/dy;		// flux into solid
							
							flux_inh += inx*H_FLUX_X(xm1,ym1,i)/dx + iny*H_FLUX_Y(xm1,ym1,i)/dy;		// energy flux into solid

							// absolute value of velocities
							ux = abs(e[inv_i].x);	
							uy = abs(e[inv_i].y);

							flux_outf += inx*FF(x,y,inv_i)*ux/dx + iny*FF(x,y,inv_i)*uy/dy;				// flux out of solid, back along inverse velocity
							
							flux_outh += inx*HH(x,y,inv_i)*ux/dx + iny*HH(x,y,inv_i)*uy/dy;				// energy flux out of solid, back along inverse velocity
						}
					}
				}

				alphaf = flux_inf/flux_outf;	// correction factor to equalise flux in to flux out

				// load required fluxes into flux array
				for (int i = 0; i < 13; i++)
				{
					ux = abs(e[i].x);	
					uy = abs(e[i].y);

					F_FLUX_X(x,y,i) = alphaf*ux*FF(x,y,i);
					F_FLUX_Y(x,y,i) = alphaf*uy*FF(x,y,i);

					H_FLUX_X(x,y,i) = alphaf*ux*HH(x,y,i);
					H_FLUX_Y(x,y,i) = alphaf*uy*HH(x,y,i);
				}
			}
		}
	}
}
void cpuLBM::boundaryFunctions(double* f, double* h, double2* f_flux, double2* h_flux)
{
	// call series of functions to handle boundary conditions
	posFluxes(f,h,f_flux,h_flux);			// calculate all outgoing fluxes of fluid nodes
	wallFluxes(f,f_flux,h,h_flux);	// update solid fluxes to maintain conservation of mass
}
void cpuLBM::RK1()
{
	// perform Runge-Kutta 1 time stepping for update

	double* fr1 = new double[Nx*Ny*Ni];
	double* hr1 = new double[Nx*Ny*Ni];

	double* feq = new double[Nx*Ny*Ni];
	double* heq = new double[Nx*Ny*Ni];

	double* tauf = new double[Nx*Ny];
	double* tauh = new double[Nx*Ny];
	double* tauhf = new double[Nx*Ny];

	double tempEqf[Ni];
	double tempEqh[Ni];

	double f_, tauf_, feq_, h_, f1_, tauh_, tauhf_, heq_, edotu;
	int id, type;

	// STEP 1

	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type > 0)
			{
				for (int i = 0; i < Ni; i++)
				{
					FR1(x,y,i) = F(x,y,i);
					HR1(x,y,i) = H(x,y,i);
				}
				continue;
			}

			macroProp(x,y,rho_G,u_G,T_G,p_G,tauf,tauh,tauhf);

			equilibrium2D(RHO(x,y),UX(x,y),UY(x,y),T(x,y),tempEqf,tempEqh);

			for (int i = 0; i < Ni; i++)
			{
				//save equilibrium function for later use
				FEQ(x,y,i) = tempEqf[i];
				HEQ(x,y,i) = tempEqh[i];

				//temp variables for calc
				f_ = F(x,y,i);
				tauf_ = TAUF(x,y);
				feq_ = FEQ(x,y,i);

				FR1(x,y,i) = (f_ + (dt/tauf_)*feq_)/(1.0 + dt/tauf_);

				h_ = H(x,y,i);
				f1_ = FR1(x,y,i);
				tauh_ = TAUH(x,y);
				tauhf_ = TAUHF(x,y);
				heq_ = HEQ(x,y,i);
				edotu = dot(e[i].x,e[i].y,UX(x,y),UY(x,y));


				HR1(x,y,i) = (h_ - dt*edotu*((feq_ - f1_)/tauhf_) + (dt/tauh_)*heq_)/(1 + dt/tauh_);
			}
		}
	}

	double flux_f, h1_, flux_h;

	// calculate all outgoing fluxes
	double2* fluxes_f = new double2[Nx*Ny*Ni];
	double2* fluxes_h = new double2[Nx*Ny*Ni];

	boundaryFunctions(fr1,hr1,fluxes_f,fluxes_h);

	//calculate recombination stage of RK method
	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type > 0)
			{
				continue;
			}

			for (int i = 0; i < Ni; i++)
			{

				//temp variables for calc
				f_ = F(x,y,i);
				f1_ = FR1(x,y,i);
				tauf_ = TAUF(x,y);
				feq_ = FEQ(x,y,i);

				flux_f = cFLUX(fluxes_f,x,y,i);

				F(x,y,i) = f_ - flux_f + (dt/tauf_)*(feq_ - f1_);

				h_ = H(x,y,i);
				h1_ = HR1(x,y,i);
				tauh_ = TAUH(x,y);
				tauhf_ = TAUHF(x,y);
				heq_ = HEQ(x,y,i);
				edotu = dot(e[i].x,e[i].y,UX(x,y),UY(x,y));

				flux_h = cFLUX(fluxes_h,x,y,i);

				H(x,y,i) = h_ - flux_h + (dt/tauh_)*(heq_ - h1_) - ((dt*edotu)/tauhf_)*(feq_ - f1_);
			}
		}
	}

	// update macro proprties
	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{		
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type == 2)	// if solid
			{
				continue;
			}
			macroPropShort(x,y);
		}
	}

	delete [] fr1;
	delete [] hr1;
	delete [] feq;
	delete [] heq;
	delete [] tauf;
	delete [] tauh;
	delete [] tauhf;

	delete [] fluxes_f;
	delete [] fluxes_h;
}
void cpuLBM::RK3()
{
	//performs the 3 step, 2nd order runge kutta method

	double* fr1 = new double[Nx*Ny*Ni];
	double* fr2 = new double[Nx*Ny*Ni];
	double* fr3 = new double[Nx*Ny*Ni];

	double* hr1 = new double[Nx*Ny*Ni];
	double* hr2 = new double[Nx*Ny*Ni];
	double* hr3 = new double[Nx*Ny*Ni];

	double* rhor1 = new double[Nx*Ny];
	double* rhor2;
	double* rhor3 = new double[Nx*Ny];

	double2* ur1 = new double2[Nx*Ny];
	double2* ur2;
	double2* ur3 = new double2[Nx*Ny];

	double* pr1 = new double[Nx*Ny];
	double* pr2;
	double* pr3 = new double[Nx*Ny];

	double* Tr1 = new double[Nx*Ny];
	double* Tr2;
	double* Tr3 = new double[Nx*Ny];

	double* feq1 = new double[Nx*Ny*Ni];
	double* feq2;
	double* feq3 = new double[Nx*Ny*Ni];

	double* heq1 = new double[Nx*Ny*Ni];
	double* heq2;
	double* heq3 = new double[Nx*Ny*Ni];

	double* tauf1 = new double[Nx*Ny];
	double* tauf2;
	double* tauf3 = new double[Nx*Ny];

	double* tauh1 = new double[Nx*Ny];
	double* tauh2;
	double* tauh3 = new double[Nx*Ny];

	double* tauhf1 = new double[Nx*Ny];
	double* tauhf2;
	double* tauhf3 = new double[Nx*Ny];

	double2* fluxes_fr2 = new double2[Nx*Ny*Ni];
	double2* fluxes_hr2 = new double2[Nx*Ny*Ni];

	double2* fluxes_fr3 = new double2[Nx*Ny*Ni];
	double2* fluxes_hr3 = new double2[Nx*Ny*Ni];

	double* flux_f2 = new double[Nx*Ny*Ni];
	double* flux_h2 = new double[Nx*Ny*Ni];



	//double net_h1, net_h2;

	double tempEqf[Ni];
	double tempEqh[Ni];

	double rho_, T_;
	double2 u_;
	double f_, f1_, f2_, f3_;
	double h_, h1_, h2_, h3_;
	double feq1_, feq2_, feq3_;
	double heq1_, heq2_, heq3_;
	double tauf1_, tauf2_, tauf3_;
	double tauh1_, tauh2_, tauh3_;
	double tauhf1_, tauhf2_, tauhf3_;
	double flux_f2_, flux_f3_;
	double flux_h2_, flux_h3_;
	double edotu1, edotu2, edotu3;

	int type, id;

	// STEP 1

	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type > 0) {					
				for (int i = 0; i < Ni; i++)
				{
					FR1(x,y,i) = F(x,y,i);
					HR1(x,y,i) = H(x,y,i);
				}
				continue;
			}

			macroProp(x,y,rhor1,ur1,Tr1,pr1,tauf1,tauh1,tauhf1);

			rho_ = RHOR1(x,y);
			u_ = UR1(x,y);
			T_ = TR1(x,y);

			equilibrium2D(rho_,u_.x,u_.y,T_,tempEqf,tempEqh);

			for (int i = 0; i < Ni; i++)
			{
				//save equilibrium function for later use
				FEQ1(x,y,i) = tempEqf[i];
				HEQ1(x,y,i) = tempEqh[i];

				//temp variables for calc
				f_ = F(x,y,i);
				tauf1_ = TAUF1(x,y);
				feq1_ = FEQ1(x,y,i);

				FR1(x,y,i) = (f_ + (dt/(2*tauf1_))*feq1_)/(1 + dt/(2*tauf1_));

				h_ = H(x,y,i);
				f1_ = FR1(x,y,i);
				tauh1_ = TAUH1(x,y);
				tauhf1_ = TAUHF1(x,y);
				heq1_ = HEQ1(x,y,i);
				edotu1 = dot(e[i].x,e[i].y,UR1X(x,y),UR1Y(x,y));

				HR1(x,y,i) = (h_ - (dt/(2*tauhf1_))*edotu1*(feq1_ - f1_) + (dt/(2*tauh1_))*heq1_)/(1 + dt/(2*tauh1_));
			}
		}
	}

	// STEP 2

	rhor2 = rhor1;
	ur2 = ur1;
	pr2 = pr1;
	Tr2 = Tr1;

	feq2 = feq1;
	heq2 = heq1;

	tauf2 = tauf1;
	tauh2 = tauh1;
	tauhf2 = tauhf1;

	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type > 0) {
				for (int i = 0; i < Ni; i++)
				{
					FR2(x,y,i) = FR1(x,y,i);
					HR2(x,y,i) = HR1(x,y,i);
				}
				continue;
			}

			for (int i = 0; i < Ni; i++)
			{
				f_ = F(x,y,i);
				tauf1_ = TAUF1(x,y);
				tauf2_ = TAUF2(x,y);
				feq1_ = FEQ1(x,y,i);
				feq2_ = FEQ2(x,y,i);
				f1_ = FR1(x,y,i);

				FR2(x,y,i) = (f_ - (dt/(2*tauf1_))*(feq1_ - f1_) + (dt/(2*tauf2_))*feq2_)/(1 + dt/(2*tauf2_));

				h_ = H(x,y,i);
				tauh1_ = TAUH1(x,y);
				tauh2_ = TAUH2(x,y);
				tauhf1_ = TAUHF1(x,y);
				tauhf2_ = TAUHF2(x,y);
				heq1_ = HEQ1(x,y,i);
				heq2_ = HEQ2(x,y,i);
				h1_ = HR1(x,y,i);
				edotu1 = dot(e[i].x,e[i].y,UR1X(x,y),UR1Y(x,y));
				edotu2 = dot(e[i].x,e[i].y,UR2X(x,y),UR2Y(x,y));
				f2_ = FR2(x,y,i);


				HR2(x,y,i) = (h_ - (dt/2)*((edotu2/tauhf2_)*(feq2_ - f2_) - (edotu1/tauhf1_)*(feq1_ - f1_)) - (dt/(2*tauh1_))*(heq1_ - h1_) + (dt/(2*tauh2_))*heq2_)/(1 + dt/(2*tauh2_));

			}
		}
	}

	// STEP 3

	boundaryFunctions(fr2,hr2,fluxes_fr2,fluxes_hr2);

	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type > 0) {
				for (int i = 0; i < Ni; i++)
				{
					FR3(x,y,i) = FR2(x,y,i);
					HR3(x,y,i) = HR2(x,y,i);
				}
				continue;
			}

			macroProp3(fluxes_fr2,fluxes_hr2,x,y,rhor3,ur3,Tr3,pr3,tauf3,tauh3,tauhf3);


			rho_ = RHOR3(x,y);
			u_ = UR3(x,y);
			T_ = TR3(x,y);

			equilibrium2D(rho_,u_.x,u_.y,T_,tempEqf,tempEqh);

			for (int i = 0; i < Ni; i++)
			{
				//save equilibrium function for later use
				FEQ3(x,y,i) = tempEqf[i];
				HEQ3(x,y,i) = tempEqh[i];

				//temp variables for calc
				f_ = F(x,y,i);
				f2_ = FR2(x,y,i);
				tauf2_ = TAUF2(x,y);
				tauf3_ = TAUF3(x,y);
				feq2_ = FEQ2(x,y,i);
				feq3_ = FEQ3(x,y,i);

				flux_f2_ = cFLUX(fluxes_fr2,x,y,i);

				FR3(x,y,i) = (f_ - flux_f2_ + (dt/(2*tauf2_))*(feq2_ - f2_) + (dt/(2*tauf3_))*feq3_)/(1 + dt/(2*tauf3_));

				h_ = H(x,y,i);
				h2_ = HR2(x,y,i);
				heq2_ = HEQ2(x,y,i);
				heq3_ = HEQ3(x,y,i);
				f3_ = FR3(x,y,i);
				tauh2_ = TAUH2(x,y);
				tauh3_ = TAUH3(x,y);
				tauhf2_ = TAUHF2(x,y);
				tauhf3_ = TAUHF3(x,y);
				edotu2 = dot(e[i].x,e[i].y,UR2X(x,y),UR2Y(x,y));
				edotu3 = dot(e[i].x,e[i].y,UR3X(x,y),UR3Y(x,y));

				flux_h2_ = cFLUX(fluxes_hr2,x,y,i);

				HR3(x,y,i) = (h_ - flux_h2_ - (dt/2)*((edotu2/tauhf2_)*(feq2_ - f2_) + (edotu3/tauhf3_)*(feq3_ - f3_)) + (dt/(2*tauh2_))*(heq2_ - h2_) + (dt/(2*tauh3_))*heq3_)/(1 + dt/(2*tauh3_));

				//save fluxes
				FLUX_F2(x,y,i) = flux_f2_;
				FLUX_H2(x,y,i) = flux_h2_;
			}
		}
	}

	// UPDATE

	boundaryFunctions(fr3,hr3,fluxes_fr3,fluxes_hr3);

	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type > 0) {
				continue;
			}

			for (int i = 0; i < Ni; i++)
			{
				f_ = F(x,y,i);
				flux_f2_ = FLUX_F2(x,y,i);
				feq2_ = FEQ2(x,y,i);
				f2_ = FR2(x,y,i);
				tauf2_ = TAUF2(x,y);
				feq3_ = FEQ3(x,y,i);
				f3_ = FR3(x,y,i);
				tauf3_ = TAUF3(x,y);

				flux_f3_ = cFLUX(fluxes_fr3,x,y,i);

				F(x,y,i) = f_ - 0.5*(flux_f2_ + flux_f3_) + (dt/2.0)*((feq2_ - f2_)/tauf2_ + (feq3_ - f3_)/tauf3_);

				h_ = H(x,y,i);
				flux_h2_ = FLUX_H2(x,y,i);
				heq2_ = HEQ2(x,y,i);
				h2_ = HR2(x,y,i);
				tauh2_ = TAUH2(x,y);
				heq3_ = HEQ3(x,y,i);
				h3_ = HR3(x,y,i);
				tauh3_ = TAUH3(x,y);
				edotu2 = dot(e[i].x,e[i].y,UR2X(x,y),UR2Y(x,y));
				edotu3 = dot(e[i].x,e[i].y,UR3X(x,y),UR3Y(x,y));
				tauhf2_ = TAUHF2(x,y);
				tauhf3_ = TAUHF3(x,y);

				flux_h3_ = cFLUX(fluxes_hr3,x,y,i);

				H(x,y,i) = h_ - 0.5*(flux_h2_ + flux_h3_) + (dt/2.0)*((heq2_ - h2_)/tauh2_ + (heq3_ - h3_)/tauh3_) - (dt/2.0)*((edotu2/tauhf2_)*(feq2_ - f2_) + (edotu3/tauhf3_)*(feq3_ - f3_));			
			}
		}
	}

	// update macro proprties
	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{			
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type == 2)	// if solid
			{
				continue;
			}

			macroPropShort(x,y);
		}
	}

	delete [] fr1;
	delete [] fr2;
	delete [] fr3;

	delete [] hr1;
	delete [] hr2;
	delete [] hr3;

	delete [] rhor1;
	delete [] rhor3;

	delete [] ur1;
	delete [] ur3;

	delete [] pr1;
	delete [] pr3;

	delete [] Tr1;
	delete [] Tr3;

	delete [] feq1;
	delete [] feq3;

	delete [] heq1;
	delete [] heq3;

	delete [] tauf1;
	delete [] tauf3;

	delete [] tauh1;
	delete [] tauh3;

	delete [] tauhf1;
	delete [] tauhf3;

	delete [] flux_f2;
	delete [] flux_h2;

	delete [] fluxes_fr2;
	delete [] fluxes_hr2;

	delete [] fluxes_fr3;
	delete [] fluxes_hr3;
}
void cpuLBM::EULER()
{
	// perform Euler time stepping for update

	double* feq = new double[Nx*Ny*Ni];
	double* heq = new double[Nx*Ny*Ni];

	double* tauf = new double[Nx*Ny];
	double* tauh = new double[Nx*Ny];
	double* tauhf = new double[Nx*Ny];

	double tempEqf[Ni];
	double tempEqh[Ni];

	double f_, tauf_, feq_, h_, tauh_, tauhf_, heq_, edotu;
	int id, type;

	double flux_f, flux_h;

	// calculate all outgoing fluxes
	double2* fluxes_f = new double2[Nx*Ny*Ni];
	double2* fluxes_h = new double2[Nx*Ny*Ni];

	boundaryFunctions(f_G,h_G,fluxes_f,fluxes_h);

	//calculate Euler step
	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type > 0)
			{
				continue;
			}

			macroProp(x,y,rho_G,u_G,T_G,p_G,tauf,tauh,tauhf);

			equilibrium2D(RHO(x,y),UX(x,y),UY(x,y),T(x,y),tempEqf,tempEqh);

			for (int i = 0; i < Ni; i++)
			{

				//temp variables for calc
				f_ = F(x,y,i);
				tauf_ = TAUF(x,y);
				feq_ = tempEqf[i];

				flux_f = cFLUX(fluxes_f,x,y,i);

				F(x,y,i) = f_ - flux_f + (dt/tauf_)*(feq_ - f_);

				h_ = H(x,y,i);
				tauh_ = TAUH(x,y);
				tauhf_ = TAUHF(x,y);
				heq_ = tempEqh[i];
				edotu = dot(e[i].x,e[i].y,UX(x,y),UY(x,y));

				flux_h = cFLUX(fluxes_h,x,y,i);

				H(x,y,i) = h_ - flux_h + (dt/tauh_)*(heq_ - h_) - ((dt*edotu)/tauhf_)*(feq_ - f_);
			}
		}
	}

	// update macro proprties
	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{		
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type == 2)	// if solid
			{
				continue;
			}
			macroPropShort(x,y);
		}
	}

	delete [] feq;
	delete [] heq;
	delete [] tauf;
	delete [] tauh;
	delete [] tauhf;

	delete [] fluxes_f;
	delete [] fluxes_h;
}
void cpuLBM::EULER_EQ()
{
	// perform Euler time stepping for update

	double* feq = new double[Nx*Ny*Ni];
	double* heq = new double[Nx*Ny*Ni];

	double* tauf = new double[Nx*Ny];
	double* tauh = new double[Nx*Ny];
	double* tauhf = new double[Nx*Ny];

	double tempEqf[Ni];
	double tempEqh[Ni];

	double f_, tauf_, feq_, h_, tauh_, tauhf_, heq_, edotu;
	int id, type;

	double flux_f, flux_h;

	// calculate all outgoing fluxes
	double2* fluxes_f = new double2[Nx*Ny*Ni];
	double2* fluxes_h = new double2[Nx*Ny*Ni];

	boundaryFunctions(f_G,h_G,fluxes_f,fluxes_h);

	//calculate Euler step
	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type > 0)
			{
				continue;
			}

			macroProp(x,y,rho_G,u_G,T_G,p_G,tauf,tauh,tauhf);

			equilibrium2D(RHO(x,y),UX(x,y),UY(x,y),T(x,y),tempEqf,tempEqh);

			for (int i = 0; i < Ni; i++)
			{

				//temp variables for calc
				f_ = F(x,y,i);
				tauf_ = TAUF(x,y);
				feq_ = tempEqf[i];

				flux_f = cFLUX(fluxes_f,x,y,i);

				F(x,y,i) = feq_ - flux_f;

				h_ = H(x,y,i);
				tauh_ = TAUH(x,y);
				tauhf_ = TAUHF(x,y);
				heq_ = tempEqh[i];
				edotu = dot(e[i].x,e[i].y,UX(x,y),UY(x,y));

				flux_h = cFLUX(fluxes_h,x,y,i);

				H(x,y,i) = heq_ - flux_h;
			}
		}
	}

	// update macro proprties
	for (int x = 0; x < Nx; x++)
	{
		for (int y = 0; y < Ny; y++)
		{		
			id = BNDARRAY(x,y);
			type = cellType[id];
			if (type == 2)	// if solid
			{
				continue;
			}
			macroPropShort(x,y);
		}
	}

	delete [] feq;
	delete [] heq;
	delete [] tauf;
	delete [] tauh;
	delete [] tauhf;

	delete [] fluxes_f;
	delete [] fluxes_h;
}
int cpuLBM::runOne(int res)
{
	// perform one step of the appropriate RK method

	hasFailed = 0;		// initialise failure flag

	double* flash;		// data that is being written to by method

	flash = T_G;

	// take a copy of the data
	if (res > 0) {
		for (int i = 0; i < Nx*Ny; i++) {
			res_copy[i] = flash[i];
		}
	}

	// PERFORM RK ////////////////

	if (RKMETHOD == 0)
	{
		RK1();
	}
	else if (RKMETHOD == 1)
	{
		RK3();
	}
	else if (RKMETHOD == 2)
	{
		EULER();
	}
	else if (RKMETHOD == 3)
	{
		EULER_EQ();
	}

	// ///////////////////////////

	if (res > 0) {
		// calculate residual
		double sum1 = 0;

		for (int i = 0; i < Nx*Ny; i++) {
			sum1 += (flash[i] - res_copy[i])*(flash[i] - res_copy[i]);
		}

		residual = sqrt(sum1/(Nx*Ny));
	}


	return hasFailed;
}