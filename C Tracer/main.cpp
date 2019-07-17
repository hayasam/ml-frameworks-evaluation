#include "stdafx.h"
#include "SourceCodeTracer.h"
#include "otherFunctions.h"

int main() {
	extern SourceCodeTracer SOURCE_CODE_TRACER;
	SOURCE_CODE_TRACER.trace("main!");
	SOURCE_CODE_TRACER.trace("main!");
	someMachineLearningBuggedCode();
	SOURCE_CODE_TRACER.trace("main!");
	SOURCE_CODE_TRACER.trace("main!");
	std::cout << "Trace function executed" << std::endl;
	return 0;
}
