#include "stdafx.h"
#include "utils.h"

using namespace std;

string lookupClError(int errc)
{
	// from cl.h : Error code definitions into -> awk '{printf "case %d: return \"%s\";\n", $3, substr($2,4)}' errorcodes.txt
	switch (errc) {
		case 0: return "SUCCESS";
		case -1: return "DEVICE_NOT_FOUND";
		case -2: return "DEVICE_NOT_AVAILABLE";
		case -3: return "COMPILER_NOT_AVAILABLE";
		case -4: return "MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "OUT_OF_RESOURCES";
		case -6: return "OUT_OF_HOST_MEMORY";
		case -7: return "PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "MEM_COPY_OVERLAP";
		case -9: return "IMAGE_FORMAT_MISMATCH";
		case -10: return "IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "BUILD_PROGRAM_FAILURE";
		case -12: return "MAP_FAILURE";
		case -13: return "MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "COMPILE_PROGRAM_FAILURE";
		case -16: return "LINKER_NOT_AVAILABLE";
		case -17: return "LINK_PROGRAM_FAILURE";
		case -18: return "DEVICE_PARTITION_FAILED";
		case -19: return "KERNEL_ARG_INFO_NOT_AVAILABLE";
		case -30: return "INVALID_VALUE";
		case -31: return "INVALID_DEVICE_TYPE";
		case -32: return "INVALID_PLATFORM";
		case -33: return "INVALID_DEVICE";
		case -34: return "INVALID_CONTEXT";
		case -35: return "INVALID_QUEUE_PROPERTIES";
		case -36: return "INVALID_COMMAND_QUEUE";
		case -37: return "INVALID_HOST_PTR";
		case -38: return "INVALID_MEM_OBJECT";
		case -39: return "INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "INVALID_IMAGE_SIZE";
		case -41: return "INVALID_SAMPLER";
		case -42: return "INVALID_BINARY";
		case -43: return "INVALID_BUILD_OPTIONS";
		case -44: return "INVALID_PROGRAM";
		case -45: return "INVALID_PROGRAM_EXECUTABLE";
		case -46: return "INVALID_KERNEL_NAME";
		case -47: return "INVALID_KERNEL_DEFINITION";
		case -48: return "INVALID_KERNEL";
		case -49: return "INVALID_ARG_INDEX";
		case -50: return "INVALID_ARG_VALUE";
		case -51: return "INVALID_ARG_SIZE";
		case -52: return "INVALID_KERNEL_ARGS";
		case -53: return "INVALID_WORK_DIMENSION";
		case -54: return "INVALID_WORK_GROUP_SIZE";
		case -55: return "INVALID_WORK_ITEM_SIZE";
		case -56: return "INVALID_GLOBAL_OFFSET";
		case -57: return "INVALID_EVENT_WAIT_LIST";
		case -58: return "INVALID_EVENT";
		case -59: return "INVALID_OPERATION";
		case -60: return "INVALID_GL_OBJECT";
		case -61: return "INVALID_BUFFER_SIZE";
		case -62: return "INVALID_MIP_LEVEL";
		case -63: return "INVALID_GLOBAL_WORK_SIZE";
		case -64: return "INVALID_PROPERTY";
		case -65: return "INVALID_IMAGE_DESCRIPTOR";
		case -66: return "INVALID_COMPILER_OPTIONS";
		case -67: return "INVALID_LINKER_OPTIONS";
		case -68: return "INVALID_DEVICE_PARTITION_COUNT";
		case -69: return "INVALID_PIPE_SIZE";
		case -70: return "INVALID_DEVICE_QUEUE";
		case -71: return "INVALID_SPEC_ID";
		case -72: return "MAX_SIZE_RESTRICTION_EXCEEDED";
	}
	return "Unknown OpenCl error.";
}

string clErrInfo(cl::Error e)
{
	return "OpenCL error " + to_string(e.err()) + ": " + lookupClError(e.err());
}