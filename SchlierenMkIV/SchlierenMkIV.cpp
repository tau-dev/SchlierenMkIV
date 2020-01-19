
#include "stdafx.h"
#include "utils.h"

//#define STD_EXPORT
#define CSV_EXPORT
#undef CSV_APPEND

using namespace std;

cl::Device device;
cl::Context context;
cl::Program program;
cl::CommandQueue queue;

const int log2res = 12;
const int64_t Resolution = (1 << log2res);
const double Scale = 6.0;
const int Iteration = 2670;
const double Viewport_x = 0.0;
const double Viewport_y = 0.0;

void printDevice(int i, cl::Device &d)
{
	cout << "Device #" << i << endl;
	cout << "Name: " << d.getInfo<CL_DEVICE_NAME>() << endl;
	cout << "Type: " << d.getInfo<CL_DEVICE_TYPE>();
	cout << " (GPU = " << CL_DEVICE_TYPE_GPU << ", CPU = " << CL_DEVICE_TYPE_CPU << ")" << endl;
	cout << "Vendor: " << d.getInfo<CL_DEVICE_VENDOR>() << endl;
	cout << "Max Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
	cout << "Global Memory: " << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << " MByte" << endl;
	cout << "Max Clock Frequency: " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << endl;
	cout << "Max Allocateable Memory: " << d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / (1024 * 1024) << " MByte" << endl;
	cout << "Local Memory: " << d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KByte" << endl;
	cout << "Available: " << d.getInfo< CL_DEVICE_AVAILABLE>() << endl;
}

void print2D(uint8_t *out, int res)
{
	for (int i = 0; i < res; i++) {
		for (int j = 0; j < res; j++) {
			//cout << (int)buffers[i * Resolution + j];
			cout << ((out[i * res + j] == 1) ? '#' : ' ') << " ";
		}
		cout << endl;
	}
}

bool initOpenCL(cl::Device &device, cl::Context &context, cl::Program &prog, cl::CommandQueue &q)
{
	cl::Platform platform = cl::Platform::getDefault();
	cout << "OpenCL version: " << platform.getInfo<CL_PLATFORM_VERSION>() << endl;

	if (platform() == 0)
		throw string("No OpenCL 2.0 platform found");

	vector<cl::Device> gpus;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
	if (gpus.size() == 0)
		throw string("No devices found");
	for (int i = 0; i < gpus.size(); i++)
		printDevice(i, gpus[i]);

	unsigned int deviceId = 0;
	cout << "Device choice: ";
	cin >> deviceId;
	if (deviceId < 0 || deviceId >= gpus.size())
		throw string("Invalid device choice");

	device = gpus[deviceId];

	context = cl::Context({ device });

	cl::Program::Sources sources;
	ifstream sourcefile("kernel.cl");
	string sourcecode(istreambuf_iterator<char>(sourcefile), (istreambuf_iterator<char>()));
	sources.push_back(sourcecode);

	prog = cl::Program(context, sources);
	if (prog.build({ device }) != CL_SUCCESS)
		throw string("OpenCL build error:\n") + prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

	q = cl::CommandQueue(context, device);

	return true;
}


void calculate(uint8_t *schlieren, int32_t res = 32768, int32_t iter = 1000, double scale = 6.0, double vx = 0.0, double vy = 0.0)
{
	cl::Buffer schlierenbuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * res * res);

	cl::Kernel kernel = cl::Kernel(program, "schlieren");
	kernel.setArg(0, schlierenbuffer);
	kernel.setArg(1, scale);
	kernel.setArg(2, res);
	kernel.setArg(3, iter);
	kernel.setArg(4, vx);
	kernel.setArg(5, vy);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(res * res), cl::NullRange);
	queue.finish();
	cout << " computed...";
	queue.enqueueReadBuffer(schlierenbuffer, CL_TRUE, 0, sizeof(uint8_t) * res * res, schlieren);
}

void scaledown(uint8_t *schlieren_old, uint8_t *schlieren_new, int oldres)
{
	int newres = oldres / 2;
	cl::Buffer oldbuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * oldres * oldres);
	cl::Buffer newbuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * newres * newres);

	cl::Kernel kernel = cl::Kernel(program, "scaledown");

	kernel.setArg(0, oldbuffer);
	kernel.setArg(1, newbuffer);
	kernel.setArg(2, oldres);

	queue.enqueueWriteBuffer(oldbuffer, CL_FALSE, 0, sizeof(uint8_t) * oldres * oldres, schlieren_old);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(newres * newres), cl::NullRange/*cl::NDRange(newres)*/);
	queue.finish();
	queue.enqueueReadBuffer(newbuffer, CL_TRUE, 0, sizeof(uint8_t) * newres * newres, schlieren_new);
}

int sumup(uint8_t *schlieren, int res)
{
	int count = 0;
	for (int i = 0; i < res; i++)
		for (int j = 0; j < res; j++)
			if (schlieren[j * res + i] == 1)
				count++;
	return count;
}



int main(int argc, char *argv[])
{
#ifdef CSV_EXPORT
#ifdef CSV_APPEND
	ofstream outfile("dim.csv", ios::out | ios::app);
#else
	ofstream outfile("dim.csv", ios::out);
#endif
#endif
	try {
		initOpenCL(device, context, program, queue);
	} catch (string s) {
		cout << s << endl << "Could not init OpenCL." << endl;
		return -1;
	}

	cout << "Calculating null sets...";

	uint8_t *schlierenBufferA = new uint8_t[Resolution * Resolution];
	uint8_t *schlierenBufferB = new uint8_t[Resolution / 2L * Resolution / 2L];

	uint8_t *buffers[] = { schlierenBufferA, schlierenBufferB };
	try {
		calculate(schlierenBufferA, Resolution, Iteration, Scale, Viewport_x, Viewport_y);
	} catch (cl::Error e) {
		cout << clErrInfo(e) << endl;
		return -1;
	}

	cout << " finished." << endl;

	int res = Resolution;
	int count = 0;

	outfile << "S;k;r;N;log r;log N" << endl;

	for (int i = 0; i < log2res; i++) {
		cout << "Downscale from " << res << " to " << res / 2 << "... ";
		count = sumup(buffers[i % 2], res);
		outfile << Scale << ";" << Iteration << ";" << res / Scale << ";" << count << ";" << log10(res / Scale) << ";" << log10(count) << endl;
		scaledown(buffers[i % 2], buffers[(i + 1) % 2], res);
		cout << "finished." << endl;
		res /= 2;

	}

#ifdef CSV_EXPORT
	outfile.close();
#endif
	delete[] schlierenBufferA, schlierenBufferB;

	cout << "Done." << endl;
	system("PAUSE");
	return 0;
}

