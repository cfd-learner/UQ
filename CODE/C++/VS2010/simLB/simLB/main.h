#ifndef MAIN_H_
#define MAIN_H_

////////////////////////////////////////////////////////////////////////////////
// includes

// includes, local
#include "fileHandling.h"
#include "cpuLBM.h"
#include "macros.h"
#include "vector_types.h"
#include "engine.h"

// includes, system
#include <string>

// global variables
cpuLBM* lbm;

// forward declaration
void runSimStep();
void cleanUp();

#endif