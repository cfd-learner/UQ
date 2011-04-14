#ifndef FILEHANDLING_H_
#define FILEHANDLING_H_

//include
#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <time.h>

using namespace std;

// definitions
#define BUFSIZE MAX_PATH

class folderSystem {
	// class containing all folders that are used in a run and their paths
public:
	// Constructor
	folderSystem();
	// Paths
	string homePath;
	string resultsPath;
	string runPath;
	string resultsMatPath;
	string resultsTecPath;
	string inputsPath;
	string inputFilePath;

	// functions
	void getDir(string& cDir);
	void makeFolder(string szFile);
	void copyFile(char * from, char * to);
	string getFileName(string file);

private:
	string splitFilename (const string& str);
};




#endif