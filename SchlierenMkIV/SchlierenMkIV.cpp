
#include "stdafx.h"
#include "utils.h"

//#define STD_EXPORT
#define CSV_EXPORT
#undef CSV_APPEND

using namespace std;
using namespace std::chrono;

cl::Device device;
cl::Context context;
cl::Program program;
cl::CommandQueue queue;

const int log2res = 11;
const int64_t Resolution = (1 << log2res);
const double Scale = 6.0;
const int Iteration = 1000; // WTF.
const double Viewport_x = 0.0;
const double Viewport_y = 0.0;

void printDevice(int i, cl::Device &d)
{
	cout << "Device #" << i << ": \"" << d.getInfo<CL_DEVICE_NAME>() << "\"" << endl;
	cl_device_type typ = d.getInfo<CL_DEVICE_TYPE>();
	cout << "Type ";
	switch (typ) {
	case CL_DEVICE_TYPE_GPU: 
		cout << "GPU"; break;
	case CL_DEVICE_TYPE_CPU:
		cout << "GPU"; break;
	default:
		cout << typ; break;
	}
	cout << endl;
	cout << "Vendor: " << d.getInfo<CL_DEVICE_VENDOR>() << endl;
	cout << "Max Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
	cout << "Global Memory: " << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << " MByte" << endl;
	cout << "Max Clock Frequency: " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << endl;
	cout << "Max Allocateable Memory: " << d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / (1024 * 1024) << " MByte" << endl;
	cout << "Local Memory: " << d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KByte" << endl;
	cout <<  (d.getInfo< CL_DEVICE_AVAILABLE>() == 1 ? "Available" : "Not available") << endl << endl;
}

void print2D(uint8_t *buffer, int res)
{
	for (int i = 0; i < res; i++) {
		for (int j = 0; j < res; j++) {
			//cout << (int)buffers[i * Resolution + j];
			cout << ((buffer[i * res + j] == 1) ? '#' : ' ') << " ";
		}
		cout << endl;
	}
}

struct Color
{
  uint8_t R, G, B, A;
};

Color white{ 255, 255, 255, 255 };
Color black{ 0, 0, 0, 255 };

void drawPNG(uint8_t * buffer, int res, string filename, Color yes = black, Color no = white)
{
	vector<uint8_t> Image(sizeof(Color) * res * res);
	Color c;

	for (int i = 0; i < res * res; i++) {
		c = (buffer[i] == 1) ? yes : no;

		Image[4 * i + 0] = c.R;
		Image[4 * i + 1] = c.G;
		Image[4 * i + 2] = c.B;
		Image[4 * i + 3] = c.A;
	}

	unsigned error = lodepng::encode(filename, Image, res, res);

	if (error)
		std::cout << "LodePNG error: " << error << ": " << lodepng_error_text(error) << std::endl;
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
	if (gpus.size() != 1) {
		cout << "Device choice (0 - " << gpus.size() - 1 << "): ";
		cin >> deviceId;
	}
	else {
		cout << "Choosing the only GPU" << endl;
	}
	if (deviceId < 0 || deviceId >= gpus.size())
		throw string("Invalid device choice");

	device = gpus[deviceId];

	context = cl::Context({ device });

	cl::Program::Sources sources;
	ifstream sourcefile("kernel.cl");
	string sourcecode(istreambuf_iterator<char>(sourcefile), (istreambuf_iterator<char>()));
	sources.push_back(sourcecode);

	prog = cl::Program(context, sources);
	try {
		prog.build({ device });
	}
	catch (cl::Error e) {
		throw string("OpenCL build error:\n") + prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	}

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

void testscaledown()
{
	uint8_t * A = new uint8_t[4096* 4096];
	uint8_t * B = new uint8_t[8192 * 8192];

	cout << "Calculating 128 x 128...";

	try {
		calculate(A, 128, Iteration, Scale, Viewport_x, Viewport_y);
	}
	catch (cl::Error e) {
		cout << clErrInfo(e) << endl;
	}

	cout << " finished." << endl;

	cout << "Exporting PNG...";
	drawPNG(A, 128, "128.png");
	cout << " finished." << endl;

	cout << "Calculating 8192 x 8192...";

	try {
		calculate(B, 8192, Iteration, Scale, Viewport_x, Viewport_y);
	}
	catch (cl::Error e) {
		cout << clErrInfo(e) << endl;
	}

	cout << " finished." << endl;

	cout << "Scaling down n times...";
	scaledown(B, A, 8192);
	scaledown(A, B, 4096);
	scaledown(B, A, 2048);
	scaledown(A, B, 1024);
	scaledown(B, A, 512);
	scaledown(A, B, 256);

	cout << " finished." << endl;

	cout << "Exporting PNG...";
	drawPNG(B, 128, "8192scaleddown.png");
	cout << " finished." << endl;

	delete[] A, B;
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

	testscaledown();

//	
//	cout << "Calculating null sets...";
//
//	uint8_t *schlierenBufferA = new uint8_t[Resolution * Resolution];
//	uint8_t *schlierenBufferB = new uint8_t[Resolution / 2L * Resolution / 2L];
//
//	uint8_t *buffers[] = { schlierenBufferA, schlierenBufferB };
//	auto start = steady_clock::now();
//	try {
//		calculate(schlierenBufferA, Resolution, Iteration, Scale, Viewport_x, Viewport_y);
//	} catch (cl::Error e) {
//		cout << clErrInfo(e) << endl;
//		return -1;
//	}
//
//	cout << " finished." << endl;
//
//	int res = Resolution;
//	int count = 0;
//
//#ifdef CSV_EXPORT
//#ifndef CSV_APPEND
//	outfile << "S;k;r;N;log r;log N" << endl;
//#endif // !CSV_APPEND
//#endif // CSV_EXPORT
//
//	for (int i = 0; i < log2res; i++) {
//		cout << "Downscale from " << res << " to " << res / 2 << "... ";
//		count = sumup(buffers[i % 2], res);
//		outfile << Scale << ";" << Iteration << ";" << res / Scale << ";" << count << ";" << log10(res / Scale) << ";" << log10(count) << endl;
//		try {
//			scaledown(buffers[i % 2], buffers[(i + 1) % 2], res);
//		}
//		catch (cl::Error e) {
//			cout << clErrInfo(e) << endl;
//			return -1;
//		}
//		
//		cout << "finished." << endl;
//		res /= 2;
//
//	}
//
//#ifdef CSV_EXPORT
//	outfile.close();
//#endif
//
//	//calculate(schlierenBufferA, 16, 10);
//	//drawPNG(schlierenBufferA, 16, "test.png");
//
//#ifdef CSV_EXPORT
//	outfile.close();
//#endif
//	delete[] schlierenBufferA, schlierenBufferB;
//
//	std::chrono::duration<float> delta = steady_clock::now() - start;
//	cout << "Done in " << delta.count() << " seconds." << endl;
	system("PAUSE");
	return 0;
}

