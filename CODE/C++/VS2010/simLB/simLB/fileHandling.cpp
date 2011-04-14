// all file handling stuff goes here

#include "fileHandling.h"

using namespace std;

folderSystem::folderSystem()
{
	// constructor

	// find the current directory path
	getDir(homePath);

	// check for results directory, create if not already there
	resultsPath = homePath;
	resultsPath.append("\\Results");

	int err;

	err = CreateDirectory(resultsPath.c_str(),NULL);

	// generate current results folder with date and time stamp
	time_t rawtime;
	struct tm * timeinfo;
	string dateNtime;

	time(&rawtime);
	timeinfo = localtime(&rawtime);
	dateNtime = string(asctime(timeinfo));

	size_t pos;

	// replace colons
	while (dateNtime.find(':') != -1)
	{
		pos = dateNtime.find(':');
		dateNtime.replace(pos,1,"-");
	}

	// replace spaces
	while (dateNtime.find(' ') != -1)
	{
		pos = dateNtime.find(' ');
		dateNtime.replace(pos,1,"_");
	}

	// delete last item in string
	dateNtime.erase(24);

	string runFolder;
	runFolder = resultsPath;
	
	runFolder.append("\\");
	runFolder.append(dateNtime.c_str());

	if (!CreateDirectory(runFolder.c_str(),NULL)) 
   { 
      printf("CreateDirectory failed (%d)\n", GetLastError()); 
   } 
	

	// make subfolders

	// MATLAB
	resultsMatPath.append(runFolder);
	resultsMatPath.append("\\MATLAB");

	err = CreateDirectory(resultsMatPath.c_str(),NULL);

	//INPUTS
	inputsPath.append(runFolder);
	inputsPath.append("\\INPUTS");

	err = CreateDirectory(inputsPath.c_str(),NULL);

	// get input file path from .txt file
	inputFilePath = getFileName("input.txt");

	// copy file to inputs
	string save_to;
	save_to.append(inputsPath);
	save_to.append("\\");
	save_to.append(splitFilename(inputFilePath));
	copyFile((char*)inputFilePath.c_str(),(char*)save_to.c_str());

}

void folderSystem::copyFile(char * from, char * to)
{
	// save a file from location "from" to location "to"

	ifstream input_file;
	input_file.open(from);

	ofstream save_file(to);

	if (save_file.is_open() && input_file.is_open())
	{
		save_file << input_file.rdbuf();
		save_file.close();
		input_file.close();
	}
	else cout << "Unable to open file";
}

void folderSystem::makeFolder(string szFile)
{
	// makes a folder at the location specified by szFile relative to current directory
	// eg: /bin/x64.Debug

	string fullPath;

	getDir(fullPath);

	fullPath.append(szFile);

	CreateDirectory(fullPath.c_str(),NULL);
}

void folderSystem::getDir(string& cDir)
{
	// returns current directory as string to passed in value
	char buffer[BUFSIZE];
	int dwret;

	dwret = GetCurrentDirectory(BUFSIZE,buffer);

	cDir = buffer;
}

string folderSystem::getFileName(string file)
{
	string filePath;

	std::ifstream inf(file.c_str());

	string delimit = "end";
	string sub, tmp;

	while(!getline(inf, tmp).eof())
	{
		size_t i = tmp.find(delimit);
		sub = tmp.substr(0,i);
	}

	return sub;
}

string folderSystem::splitFilename (const string& str)
{
  size_t found;
  found=str.find_last_of("/\\");
  return str.substr(found+1);
}
