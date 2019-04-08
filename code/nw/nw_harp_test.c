/*
 * File: 					nw_harp_test.c
 * Author: 				Anthony Cabrera
 * Desrcription:	HR Zohouri OpenCL FPGA implementation for Intel HARP
 * 								Also based on mem_bandwidth example for OpenCL on TACC
 *
 */

// Kernel Specific
#include "work_group_size.h"
#define LIMIT -999
#define BLOCK_SIZE 16

// Both HARP and nw
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "CL/opencl.h"

// Just nw
#include <iostream>
#include <string>
#include <assert.h>
#include <CL/cl.h> // don't know if necessary just yet
#include "../common/timer.h"
#include "../common/opencl_util.h"

// Just HARP
#include <math.h>
#include <unistd.h>
#include "CL/opencl.h" //ACL specific includes
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

// global variables

int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

// local variables from nw
static cl_context				context;
static cl_command_queue	cmd_queue1;
static cl_command_queue	cmd_queue2;
static cl_device_type		device_type;
static cl_uint						num_devices;

// local variable from harp
static cl_platform_id platform;
static cl_device_id		device;
static cl_int 				status;
static cl_program			program;

// local variable added as a result of HARP
static cl_kernel			kernel_nw1;
static cl_kernel			kernel_nw2;


static void dump_error(const char *str, cl_int status)
{
	printf("%s\n", str);
	printf("Error code: %d\n", status);
}

static void freeResources()
{
	// free kernel resources
	if (kernel_nw1)
		clReleaseKernel(kernel_nw1);
	if (kernel_nw2)
		clReleaseKernel(kernel_nw2);

	if (program)
		clReleaseProgram(program);
	if (cmd_queue1)
		clReleaseCommandQueue(cmd_queue1);
	if (cmd_queue2)
		clReleaseCommandQueue(cmd_queue2);

}

int maximum(int a, int b, int c)
{
	int k;
	if(a <= b)
		k = b;
	else
		k = a;
	if(k <=c )
		return(c);
	else
		return(k);
}

// build the context and command
// queue from like the OpenCL TACC example
// TODO: factor out all of the "if status checks"
static int initialize()
{

	cl_uint num_platforms;
	//cl_uint num_devices;

	// get the platform ID
	status = clGetPlatformIDs(1, &platform, &num_platforms);
	if (status != CL_SUCCESS)
	{
		dump_error("Failed clGetPlatformIDs.", status);
		freeResources();
		return 1;
	}
	if (num_platforms != 1) // remember, this is a HARP specific constraint
													// for reasons that aren't immediately clear.
	{
		printf("Found %d platforms!\n", num_platforms);
		freeResources();
		return 1;
	}

	// get the device ID
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device,
						&num_devices);
	if (status != CL_SUCCESS)
	{
		dump_error("Failed clGetDeviceIDs.", status);
		freeResources();
		return 1;
	}
	if (num_devices != 1) // another HARP specific constraint...
	{
		printf("Found %d devices!\n", num_devices);
		freeResources();
		return 1;
	}

	//print_device_info(device);

	// create a context
	context = clCreateContext(0, 1, &device, NULL, NULL, &status);
	if (status != CL_SUCCESS)
	{
		dump_error("Failed clCreateContext.", status);
		freeResources();
		return 1;
	}

	// create a command queue
	cmd_queue1 = clCreateCommandQueue(context, device, 0, NULL);
	if(!cmd_queue1)
	{
		printf("ERROR: clCreateCommandQueue() failed\n");
		return -1;
	}

	return 0;

}

void cleanup()
{
	freeResources();
}




void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <kernel_version> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	fprintf(stderr, "\t<enable_traceback> - write traceback to results.txt \n");
	fprintf(stderr, "\t<kernel_version> - version of kernel or bianry file to load\n");
	exit(1);
}


int main(int argc, char **argv)
{

	int max_rows, max_cols, penalty;
	char *version_string;
	int version_number;
	size_t sourcesize;

	TimeStamp input_start, input_end;
	TimeStamp compute_start, compute_end;
	TimeStamp output_start, output_end;
	double inputTime, outputTime, computeTime;

  // this just gets the version number. no OpenCL specific calls here.
  init_fpga2(&argc, &argv, &version_string, &version_number);

	int enable_traceback = 0;

	//Anthony note:
	//init_fpga2 decrements argc... WTF
  if (argc >= 3)
  {
      max_rows = atoi(argv[1]);
      max_cols = atoi(argv[1]);
      penalty = atoi(argv[2]);
  }
  else
  {
      usage(argc, argv);
      exit(1);
  }

	// If enable traceback is to be specified, it should be argv[3]
	// and arg[4] is version number.
	// else, enable_traceback remains at 0.
	printf("argc= %d\n", argc);
	if (argc+1 == 5)
	{
		enable_traceback = atoi(argv[3]);
		printf("enable_traceback = %d\n", enable_traceback);
	}
  
  if (is_ndrange_kernel(version_number))
  {
      if ( atoi(argv[1]) % 16 != 0)
      {
          fprintf(stderr, "The dimension values must be a multiple of 16\n");
          exit(1);
      }
  }
  
  max_rows = max_rows + 1;
  max_cols = max_cols + 1;
  int num_rows = max_rows;
  int num_cols = (version_number == 5) ? max_cols - 1 : max_cols;

	int data_size = max_cols * max_rows;
	int ref_size = num_cols * num_rows;

	printf("max_rows = %d, max_cols = %d\n", max_rows, max_cols);
	printf("num_rows = %d, num_cols = %d\n", num_rows, num_cols);
	printf("data_size = %d, ref_size = %d\n", data_size, ref_size);
	printf("BSIZE = %d\n", BSIZE);
	printf("PAR = %d\n", PAR);

	// alignedMalloc is in opencl/common/opencl_util.h
	int *reference = (int *)alignedMalloc(ref_size * sizeof(int));
#ifdef VERIFY
	int *reference_cpu = (int *)alignedMalloc(data_size * sizeof(int));
#endif
	int *input_itemsets = (int *)alignedMalloc(data_size * sizeof(int));
	int *output_itemsets = (int *)alignedMalloc(ref_size * sizeof(int));

	// for v7 and above
	int *buffer_v = NULL, *buffer_h = NULL;
	if (version_number >= 5)
	{
		buffer_h = (int *)alignedMalloc(num_cols * sizeof(int));
		buffer_v = (int *)alignedMalloc(num_rows * sizeof(int));
	}

	srand(7);

	//Initialize matrix to 0's
	for (int i = 0; i < max_cols; i++)
	{
		for (int j = 0; j < max_rows; j++)
		{
			input_itemsets[i * max_cols + j] = 0;
		}
	}

	for (int i = 1; i < max_rows; i++)
	{
		//initialize the first column
		input_itemsets[i * max_cols] = rand() % 10 + 1;
	}

	for (int j = 1; j < max_cols; j++)
	{
		//initialize the first row
		input_itemsets[j] = rand() % 10 + 1;
	}

	for (int i = 1; i < max_cols; i++)
	{
		for (int j = 1; j < max_rows; j++)
		{
			int ref_offset = (version_number == 5) ? i * num_cols + (j - 1) :
				i * num_cols + j;
			reference[ref_offset] =
				blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
			//printf("ref[%d][%d] = %d\n", i, j-1, reference[ref_offset]);
		}
	}

	for (int i = 1; i < max_rows; i++)
	{
		input_itemsets[i * max_cols] = -i * penalty;
		if (version_number == 5)
		{
			buffer_v[i] = -i * penalty;
		}
	}

	if (version_number == 5)
		buffer_v[0] = 0;

	for (int j = 1; j < max_cols; j++)
	{
		input_itemsets[j] = -j * penalty;
		if (version_number == 5)
		{
			buffer_h[j - 1] = -j * penalty;
		}
	}

	// getVersionedKernelName2 is in opencl/common/opencl_util.h
	// there are no OpenCL API calls in this function.
	// might need to update it to look for path ./bin_fpga/%s_%s.aocx
	char *kernel_file_path  = getVersionedKernelName2("./nw_kernel",
		version_string);

	// read in .aocx file. The loadBinaryFile() (HARP) and read_kernel()
	// (Rodinia_FPGA) are basically the same except HARP uses the cpp "new"
	// operator to allocate memory instead of malloc.	
	// 
	// sourcesize is a local variable defined at beginning of main function
	char *source = read_kernel(kernel_file_path, &sourcesize);

	//read the kernel core source
	char const * kernel_nw1 = "nw_kernel1";
	char const * kernel_nw2 = "nw_kernel2";
	
	int nworkitems, workgroupsize = 0;
	// TODO: investigate BSIZE
	nworkitems = BSIZE;

	if (nworkitems < 1 || workgroupsize < 0)
	{
		printf("ERROR: invalid or missing \
			<num_work_items>[/<work_group_size>]\n");
		return -1;
	}

	//set global and local workitems
	size_t local_work[3] = { (size_t) ( (workgroupsize>0) ? workgroupsize : 1 ),
		1, 1 };
	// for GPU, nworkitems = no. of GPU threads
	size_t global_work[3] = { (size_t) nworkitems, 1, 1 };  

	// OpenCL initialization
	// remember we're going with the Rodinia flavor of this
	// also remember, this function is defined earlier in this source code
	if (initialize())
	{
		return -1;
	}

	// compile kernel
	cl_int err = 0;	
	// pre-build FPGA kernel
	// Anthony note: original Rodinia has device_list, HARP only has one device.
	// changing device_list to device.
	cl_program prog = clCreateProgramWithBinary(context, 1, &device,
		&sourcesize, (const unsigned char **) &source, NULL, &err);
	if (err != CL_SUCCESS)	
	{
		printf("ERROR: clCreateProgramWithSource/Binary() => %d\n", err);
		// display_error_message in opencl/common/opencl_util.h
		display_error_message(err, stderr);
		return -1;
	}

	char clOptions[110];
	sprintf(clOptions, "-I .");

	// clBuildProgram_SAFE in opencl/common/opencl_util.h
	// this is a wrapper around two OpenCL API calls:
	// clBuildProgram is called to build and link host with FPGA binary
	// if clBuildProgram fails, clGetProgramBuildInfo is called to help diagnose
	// what went wrong.
	// original Rodinia has device_list, HARP only has one device.
	// changing device_list to device.
	printf("num_devices=%d, device=%d\n", num_devices, device);
	clBuildProgram_SAFE(prog, num_devices, &device, clOptions, NULL, NULL);

	cl_kernel kernel1;
	cl_kernel kernel2;
	kernel1 = clCreateKernel(prog, kernel_nw1, &err);
	if (is_ndrange_kernel(version_number))
	{
		kernel2 = clCreateKernel(prog, kernel_nw2, &err);
	}
	else
	{
		// use the same kernel in single work-item versions
		kernel2 = kernel1;
	}

	if (err != CL_SUCCESS)
	{
		printf("ERROR: clCreateBuffer buffer_v_d (size:%d) => %d\n", num_rows,
			err);
		return -1;
	}	

	// create buffers
#ifdef NO_INTERLEAVE

	cl_mem reference_d = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_BANK_1_ALTERA, ref_size * sizeof(int), NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("ERROR: clCreateBuffer input_item_set (size:%d) => %d\n", ref_size,
			err);
		return -1;
	}
	
	int device_buff_size = (version_number >= 7) ? num_cols * (num_rows + 1) :
		((version_number == 5) ? ref_size : data_size);
	
	cl_mem input_itemsets_d = clCreateBuffer(context, CL_MEM_READ_WRITE | 
		CL_MEM_BANK_2_ALTERA, device_buff_size * sizeof(int), NULL, &err);

	
	if(err != CL_SUCCESS) 
	{ 
		printf("ERROR: clCreateBuffer input_item_set (size:%d) => %d\n", 
			device_buff_size, err); 
		return -1;
	}

	
	// create extra buffer for v5 and above
	cl_mem buffer_v_d;
	if (version_number >= 5)
	{
		buffer_v_d = clCreateBuffer(context, CL_MEM_READ_ONLY | 
			CL_MEM_BANK_1_ALTERA, num_rows * sizeof(int), NULL, &err);
		if(err != CL_SUCCESS) 
		{ 
			printf("ERROR: clCreateBuffer buffer_v_d \
				(size:%d) => %d\n", num_rows, err); 
			return -1;}
		}

#else

	cl_mem reference_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
		ref_size * sizeof(int), NULL, &err);
	if(err != CL_SUCCESS) 
	{ 
		printf("ERROR: clCreateBuffer reference \
			(size:%d) => %d\n", ref_size, err); 
		return -1;
	}
	int device_buff_size = (version_number >= 7) ? num_cols * (num_rows + 1) 
		: ((version_number == 5) ? ref_size : data_size);
	cl_mem input_itemsets_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 
		device_buff_size * sizeof(int), NULL, &err);
	if(err != CL_SUCCESS) 
	{ 
		printf("ERROR: clCreateBuffer input_item_set (size:%d) => %d\n", 
			device_buff_size, err); 
		return -1;
	}

	// create extra buffer for v5 and above
	cl_mem buffer_v_d;
	if (version_number == 5)
	{
		buffer_v_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
			num_rows * sizeof(int), NULL, &err);
		if(err != CL_SUCCESS) 
		{ 
			printf("ERROR: clCreateBuffer buffer_v_d (size:%d) => %d\n", 
				num_rows, err); 
			return -1;
		}
	}

#endif

	//write buffers
	GetTime(input_start);
	CL_SAFE_CALL(clEnqueueWriteBuffer(cmd_queue1, reference_d, 1, 0, 
		ref_size * sizeof(int), reference, 0, 0, 0));
	if (version_number == 5)
	{
		CL_SAFE_CALL(clEnqueueWriteBuffer(cmd_queue1, input_itemsets_d, 1, 0, 
			num_cols * sizeof(int), buffer_h, 0, 0, 0));
		CL_SAFE_CALL(clEnqueueWriteBuffer(cmd_queue1, buffer_v_d, 1, 0, 
			num_rows * sizeof(int), buffer_v, 0, 0, 0));
	}
	else
	{
		CL_SAFE_CALL(clEnqueueWriteBuffer(cmd_queue1, 
			input_itemsets_d, 1, 0, data_size * sizeof(int), input_itemsets, 
			0, 0, 0));
	}
	GetTime(input_end);
	inputTime = TimeDiff(input_start, input_end);
	printf("\nWrite to device done in %0.3lf ms.\n", inputTime);

	int worksize = max_cols - 1;
	printf("WG size of kernel = %d \n", BSIZE);
	printf("worksize = %d\n", worksize);
	//these two parameters are for extension use, don't worry about it.
	int offset_r = 0, offset_c = 0;
	int block_width = worksize/BSIZE;

	// constant kernel arguments
	if (is_ndrange_kernel(version_number))
	{
		CL_SAFE_CALL( clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &reference_d     ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &input_itemsets_d) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 2, sizeof(cl_int), (void*) &num_rows        ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 3, sizeof(cl_int), (void*) &penalty         ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 5, sizeof(cl_int), (void*) &offset_r        ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*) &offset_c        ) );

		CL_SAFE_CALL( clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &reference_d     ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &input_itemsets_d) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*) &num_cols        ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &penalty         ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 5, sizeof(cl_int), (void*) &block_width     ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 6, sizeof(cl_int), (void*) &offset_r        ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 7, sizeof(cl_int), (void*) &offset_c        ) );
	}
	else if (version_number < 5)
	{
		CL_SAFE_CALL( clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &reference_d     ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &input_itemsets_d) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 2, sizeof(cl_int), (void*) &num_rows        ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 3, sizeof(cl_int), (void*) &penalty         ) );
	}
	else
	{
		int cols = num_cols - 1 + PAR; // -1 since last column is invalid, +PAR to 
																	 // make sure all cells in the last chunk are processed
		int exit_col = (cols % PAR == 0) ? cols : cols + PAR - (cols % PAR);
		int loop_exit = exit_col * (BSIZE / PAR);
		
		/*int startup_cycles = PAR - 1;
		int col_chunk_cycles = BSIZE + (PAR - 1);
		int num_chunks = (num_cols + PAR - (num_cols % PAR)) / PAR;
		int loop_exit = startup_cycles + (col_chunk_cycles * num_chunks);*/

		CL_SAFE_CALL( clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &reference_d     ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &input_itemsets_d) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 2, sizeof(void *), (void*) &buffer_v_d      ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 3, sizeof(cl_int), (void*) &num_cols        ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 4, sizeof(cl_int), (void*) &penalty         ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 5, sizeof(cl_int), (void*) &loop_exit       ) );
	}

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
	#pragma omp parallel num_threads(2) shared(flag)
	{
		if (omp_get_thread_num() == 0)
		{
			#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115
				power = GetPowerFPGA(&flag);
			#else
				power = GetPowerFPGA(&flag, device_list);
			#endif
		}
		else
		{
			#pragma omp barrier
#endif
			// This block happens regardless of the #ifdef. Formatted kinda wonky
			// though.
			// Beginning of timing point
			//GetTime(compute_start);

			// NDRange versions
			if (is_ndrange_kernel(version_number))
			{
				printf("global_work_size = %d\n", BSIZE * worksize/BSIZE);
				printf("local_work_size = %d\n", BSIZE);
				printf("num_blks = %d\n", worksize/BSIZE);
				GetTime(compute_start);
				for(int blk = 1; blk <= worksize/BSIZE; blk++)
				{
					global_work[0] = BSIZE * blk;
					local_work[0]  = BSIZE;

					CL_SAFE_CALL( clSetKernelArg(kernel1, 4, sizeof(cl_int), (void*) &blk) );
					CL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue1, kernel1, 2, NULL, 
						global_work, local_work, 0, 0, NULL) );
					//printf("global_work_size = %d, blk = %d, local_work_size = %d worked\n", global_work[0], blk, local_work[0] );
				}
				clFinish(cmd_queue1);

				for(int blk = worksize/BSIZE - 1; blk >= 1; blk--)
				{
					global_work[0] = BSIZE * blk;
					local_work[0]  = BSIZE;

					CL_SAFE_CALL( clSetKernelArg(kernel2, 4, sizeof(cl_int), (void*) &blk) );
					CL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue1, kernel2, 2, NULL, 
						global_work, local_work, 0, 0, NULL) );
				}
				clFinish(cmd_queue1);
				GetTime(compute_end);
			}
			else if (version_number < 5)
			{
				GetTime(compute_start);
				CL_SAFE_CALL(clEnqueueTask(cmd_queue1, kernel1, 0, NULL, NULL));
				clFinish(cmd_queue1);
				GetTime(compute_end);
			}
			else
			{
				GetTime(compute_start);
				//int num_diags  = max_rows - 1; // -1 since last row is invalid
				int num_diags  = max_rows - 1; // -1 since last row is invalid
				int comp_bsize = BSIZE - 1;
				int last_diag  = (num_diags % comp_bsize == 0) ? num_diags : num_diags + comp_bsize - (num_diags % comp_bsize);
				int num_blocks = last_diag / comp_bsize;

				for (int bx = 0; bx < num_blocks; bx++)
				{
					int block_offset = bx * comp_bsize;

					CL_SAFE_CALL( clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*) &block_offset) );

					CL_SAFE_CALL( clEnqueueTask(cmd_queue1, kernel1, 0, NULL, NULL) );

					clFinish(cmd_queue1);
				}
				GetTime(compute_end);
			}

			// End of timing point
			//GetTime(compute_end);

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
			flag = 1;
		}
	}
#endif

	GetTime(output_start);
	err = clEnqueueReadBuffer(cmd_queue1, input_itemsets_d, 1, 0, 
		ref_size * sizeof(int), output_itemsets, 0, 0, 0);
    GetTime(output_end);
    outputTime = TimeDiff(output_start, output_end);
    printf("\nRead from device done in %0.3lf ms.\n", outputTime);
	clFinish(cmd_queue1);

	computeTime = TimeDiff(compute_start, compute_end);
	printf("\nComputation done in %0.3lf ms.\n", computeTime);

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
	energy = GetEnergyFPGA(power, computeTime);
	if (power != -1) // -1 --> sensor read failure
	{
		printf("Total energy used is %0.3lf jouls.\n", energy);
		printf("Average power consumption is %0.3lf watts.\n", power);
	}
	else
	{
		printf("Failed to read power values from the sensor!\n");
	}
#endif

#ifdef OUTPUT 
	printf("==============================================================\n");

	FILE *fout = fopen("output_itemsets.txt","w");
	int start_j = (version_number >= 5) ? 0 : 1;
	for (int i = 0; i < num_rows; ++i)
	{
		for (int j = 0; j < num_cols ; ++j)
		{
			fprintf(fout, "[%d, %d] = %d (ref: %d)\n", 
				i, j, output_itemsets[i * num_cols + j], reference[i * num_cols + j]);
		}
	}
	fclose(fout);
	//fclose(fout_cpu);
	printf("Output itemsets saved in output_itemsets.txt\n");

	printf("==============================================================\n");

#endif
/* starting here, every max_cols is decremented */
	if (enable_traceback)
	{
	  FILE *fpo = fopen("result.txt","w");
	  fprintf(fpo, "max_cols: %d, penalty: %d\n", max_cols - 1, penalty);
	  //for (int i = max_cols - 2,  j = max_rows - 2; i>1 && j>0;){
	  //for (int i = max_cols - 2,  j = max_rows - 2; i>=0 && j>=0;){ //OG
	  //for (int i = max_cols - 2,  j = max_rows - 2; i>0 && j>=0;){
	  for (int i = max_cols - 2,  j = max_rows - 2; i>=0 && j>=0;){
	    fprintf(fpo, "[%d, %d] ", i, j);
	    int nw = 0, n = 0, w = 0, traceback;
	    if (i == 0 && j == 0) {
	      fprintf(fpo, "(output: %d)\n", output_itemsets[0]);
	      break;
	    }
	    if (i > 0 && j >= 0){
	      /*nw = output_itemsets[(i - 1) * max_cols + j - 1];
	      w  = output_itemsets[ i * max_cols + j - 1 ];
	      n  = output_itemsets[(i - 1) * max_cols + j];*/
	      nw = output_itemsets[(i - 1) * num_cols + j - 1];
	      w  = output_itemsets[ i * num_cols + j - 1 ];
	      n  = output_itemsets[(i - 1) * num_cols + j];
	      fprintf(fpo, "(nw: %d, w: %d, n: %d, ref: %d) ",
		      //nw, w, n, reference[i * max_cols+j]);
		      //nw, w, n, reference[i+1 * num_cols+j+1]); // for ver 5
		      nw, w, n, reference[i * num_cols+j]); // for ver 5
	    }
	    else if (i == 0){
	      nw = n = LIMIT;
	      w  = output_itemsets[ i * num_cols + j - 1 ];
	    }
	    else if (j == 0){
	      nw = w = LIMIT;
	      n = output_itemsets[(i - 1) * num_cols + j];
	    }
	    else{
	    }

	    //traceback = maximum(nw, w, n);
	    int new_nw, new_w, new_n;
	    new_nw = nw + reference[i * num_cols + j];
	    //new_nw = nw + reference[i * num_cols + j-1]; //for ver 5
	    new_w = w - penalty;
	    new_n = n - penalty;

	    traceback = maximum(new_nw, new_w, new_n);
	    if (traceback != output_itemsets[i * num_cols +j]) {
	      fprintf(stderr, "Mismatch at (%d, %d). traceback: %d, \
					output_itemsets: %d\n",
		      i, j, traceback, output_itemsets[i * num_cols+j]);
	      //exit(1);
	    }
	    fprintf(fpo, "(output: %d)", traceback);

	    if(traceback == new_nw) {
	      traceback = nw;
	      fprintf(fpo, "(->nw) ");
	    } else if(traceback == new_w) {
	      traceback = w;
	      fprintf(fpo, "(->w) ");
	    } else if(traceback == new_n) {
	      traceback = n;
	      fprintf(fpo, "(->n) ");
	    } else {
	      fprintf(stderr, "Error: inconsistent traceback at (%d, %d)\n", i, j);
	      abort();
	    }

	    fprintf(fpo, "\n");

	    if(traceback == nw)
	    {i--; j--; continue;}

	    else if(traceback == w)
	    {j--; continue;}

	    else if(traceback == n)
	    {i--; continue;}

	    else
	      ;
	  }
	  fclose(fpo);
	  printf("Traceback saved in result.txt\n");
	}
	
/* ending here, every max_cols is decremented */
#ifdef VERIFY_
	if (enable_traceback)
	{
	  FILE *fpo = fopen("result_cpu.txt","w");
	  fprintf(fpo, "max_cols: %d, penalty: %d\n", max_cols - 1, penalty);
	  for (int i = max_cols - 2,  j = max_rows - 2; i>=0 && j>=0;){
	    fprintf(fpo, "[%d, %d] ", i, j);
	    int nw = 0, n = 0, w = 0, traceback;
	    if (i == 0 && j == 0) {
	      fprintf(fpo, "(output: %d)\n", input_itemsets_cpu[0]);
	      break;
	    }
	    if (i > 0 && j > 0){
	      nw = input_itemsets_cpu[(i - 1) * max_cols + j - 1];
	      w  = input_itemsets_cpu[ i * max_cols + j - 1 ];
	      n  = input_itemsets_cpu[(i - 1) * max_cols + j];
	      fprintf(fpo, "(nw: %d, w: %d, n: %d, ref: %d) ",
		      nw, w, n, reference[i * max_cols+j]);
	    }
	    else if (i == 0){
	      nw = n = LIMIT;
	      w  = input_itemsets_cpu[ i * max_cols + j - 1 ];
	    }
	    else if (j == 0){
	      nw = w = LIMIT;
	      n = input_itemsets_cpu[(i - 1) * max_cols + j];
	    }
	    else{
	    }

	    int new_nw, new_w, new_n;
	    new_nw = nw + reference[i * max_cols + j];
	    new_w = w - penalty;
	    new_n = n - penalty;

	    traceback = maximum(new_nw, new_w, new_n);
	    if (traceback != input_itemsets_cpu[i * max_cols+j]) {
	      fprintf(stderr, "Mismatch at (%d, %d). traceback: %d, \
					input_itemsets_cpu: %d\n",
		      i, j, traceback, input_itemsets_cpu[i * max_cols+j]);
	      //exit(1);
	    }
	    fprintf(fpo, "(output: %d)", traceback);

	    if(traceback == new_nw) {
	      traceback = nw;
	      fprintf(fpo, "(->nw) ");
	    } else if(traceback == new_w) {
	      traceback = w;
	      fprintf(fpo, "(->w) ");
	    } else if(traceback == new_n) {
	      traceback = n;
	      fprintf(fpo, "(->n) ");
	    } else {
	      fprintf(stderr, "Error: inconsistent traceback at (%d, %d)\n", i, j);
	      abort();
	    }

	    fprintf(fpo, "\n");

	    if(traceback == nw)
	    {i--; j--; continue;}

	    else if(traceback == w)
	    {j--; continue;}

	    else if(traceback == n)
	    {i--; continue;}

	    else
	      ;
	  }
	  fclose(fpo);
	  printf("Traceback saved in result_cpu.txt\n");
	} // if enable_traceback 
#endif

	// OpenCL shutdown
	freeResources();

	clReleaseMemObject(reference_d);
	clReleaseMemObject(input_itemsets_d);

	free(reference);
	free(input_itemsets);
	free(output_itemsets);
#ifdef VERIFY_
	free(input_itemsets_cpu);
	free(output_itemsets_cpu);
	free(reference_cpu);
#endif
	free(source);

	return 0;
}
