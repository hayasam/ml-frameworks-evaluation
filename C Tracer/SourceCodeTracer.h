#include <iostream>
#include <string>
#include <fstream>
#include <list>
#include <map>
#include <iomanip>
#include <ios>
#include <cstdlib>


#ifndef __SOURCE_CODE_TRACER_H__
#define __SOURCE_CODE_TRACER_H__

using namespace std;
class SourceCodeTracer {
	public:
	~SourceCodeTracer();

	static SourceCodeTracer& getInstance();
	void trace(const string& tr);
private:
	SourceCodeTracer();
	// https://stackoverflow.com/q/631664/9876427
	std::string get_env_var(std::string const & key) const;
	static SourceCodeTracer* _singleton;
	// Traceur() = delete;
	std::map<string, long> _map;
	string _filePath;
};

#endif // __SOURCE_CODE_TRACER_H__