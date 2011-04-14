
#include "main.h"

using namespace std;

int main(int argc, char *argv[])
{
	/////////////////////////////////
	// INITIALISATION + IMPORT
	/////////////////////////////////

	folderSystem* files = new folderSystem;

	lbm = new cpuLBM(files);

	/////////////////////////////////
	// MAIN FUNCTIONALITY
	/////////////////////////////////

	// MAIN LOOP
	while (lbm->stop == 0)
	{
		runSimStep();
	}

	// clean up
	cleanUp();

	return 0;
}

void runSimStep()
{
	int numRes = 20;	// number of iterations of residual collection before normalising, must be less than 100 - see cpuLBM definition of initRes
	if (lbm->stop != 0)		// check if should be stopped
	{
		// save data to .MAT
		if (lbm->MAT == 1) {
			char buffer[50];
			int n = sprintf(buffer,"step %d",lbm->step);
			
			string fileName;
			fileName.append(buffer,n);

			lbm->createMAT(fileName,1);
		}
		printf("\n simulation has stopped - user command\n");
		lbm->pause = 1;	// stop openGL re-render
	}
	else {
		lbm->updateStepCount();

		// perform RK step
		if (lbm->tol != 0) {

			lbm->stop = lbm->runOne(1);

			if (lbm->step >= 2 && lbm->step <= numRes) {
				lbm->initRes[lbm->step - 2] = lbm->residual;
				lbm->residual = 1;
			}
			else if (lbm->step == numRes+1) {
				for (int i = 0; i < numRes-1; i++) {
					lbm->resNorm = max(lbm->initRes[i],lbm->resNorm);
				}
				lbm->residual = lbm->residual/lbm->resNorm;
			}
			else if (lbm->step >= numRes+1) {
				lbm->residual = lbm->residual/lbm->resNorm;
			}
		}
		else {
			lbm->stop = lbm->runOne();
		}

		printf("\niteration %d/%d\n",lbm->step,lbm->nsteps);

		// write stuff out
		if (lbm->step % lbm->nPrintOut == 0 || lbm->stop != 0)
		{

			// save data to .MAT
			if (lbm->MAT == 1) {
				char buffer[50];
				int n = sprintf(buffer,"step %d",lbm->step);

				string fileName;
				fileName.append(buffer,n);

				lbm->createMAT(fileName);
			}

			if (lbm->tol != 0) {
				printf("\n residual = %e -> %e\n", lbm->residual, lbm->tol);
			}
		}


		// end of iterations condition
		if (lbm->step >= lbm->nsteps && lbm->nsteps != 0)	{
			lbm->stop = 2;
			printf("\n simulation has stopped - end of iterations\n");
		}

		// below tolerance condition
		if (lbm->tol != 0 && lbm->residual <= lbm->tol && lbm->step >= numRes+1)	{
			lbm->stop = 2;
			printf("\n residual = %e -> %e\n", lbm->residual, lbm->tol);
			printf("\n simulation has stopped - residual less than tolerance\n");
		}
	}
}

void cleanUp()
{
	// delete global variables
	delete lbm;
}