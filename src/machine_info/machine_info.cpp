#include "machine_info.h"
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef WIN32
    // Used by __cpuid (see: http://msdn.microsoft.com/en-us/library/hskdteyh%28v=vs.80%29.aspx and http://weseetips.com/2009/06/21/how-to-get-the-cpu-name-string/)
    #include <iostream>
    #include <intrin.h>
#endif

//#ifdef UNIX
    #include <fstream>
    #include <string>
    #include <iostream>
//#endif

void MachineInformation::printMachineInformation()
{
    printCPUInformation();
    printGPUInformation();
}

void MachineInformation::printCPUInformation()
{
#ifdef WIN32
    // Get CPU information by __cpuid()
    // Get extended ids.
    int CPUInfo[4] = {-1};
    __cpuid(CPUInfo, 0x80000000);
    unsigned int nExIds = CPUInfo[0];

    // Get the information associated with each extended ID.
    char CPUBrandString[0x40] = { 0 };
    for( unsigned int i=0x80000000; i<=nExIds; ++i)
    {
        __cpuid(CPUInfo, i);

        // Interpret CPU brand string and cache information.
        if  (i == 0x80000002)
        {
            memcpy( CPUBrandString,
            CPUInfo,
            sizeof(CPUInfo));
        }
        else if( i == 0x80000003 )
        {
            memcpy( CPUBrandString + 16,
            CPUInfo,
            sizeof(CPUInfo));
        }
        else if( i == 0x80000004 )
        {
            memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
        }
    }

    printf("Cpu String: %s\n\n", CPUBrandString);
#endif

#ifdef UNIX
    std::ifstream cpuinfo;
    cpuinfo.open("/proc/cpuinfo");
    if (!cpuinfo)
    {
        perror("Can't open /proc/cpuinfo\n");
    }
    while (!cpuinfo.eof()){
        char temp;
        temp = cpuinfo.get();
        std::cout << temp;
    }
    std::cout << std::endl;
#endif
}


void MachineInformation::printGPUInformation(){
#ifndef QUERY_CUDA
    printf("No CUDA GPU found.");
#endif

#ifdef QUERY_CUDA
    cudaDeviceProp prop;
    int count = 0;
	if (cudaGetDeviceCount(&count) == cudaErrorInsufficientDriver)
	{
		printf("CUDA Insufficient driver error!\n");
		return;
	}
	if (cudaGetDeviceCount(&count) == cudaErrorNoDevice)
	{
		printf("CUDA no device error!\n");
		return;
	}
    for (int i=0; i< count; ++i)
    {
        cudaGetDeviceProperties(&prop,i);
        printf("GPU %i:\n", i);
        printf("\tName: %s\n", prop.name);
        printf("\ttotalGlobalMem: %i\n", (int)prop.totalGlobalMem);
        printf("\tCompute Capability: %i.%i\n", prop.major, prop.minor);
        printf("\tmultiProcessorCount: %i\n", prop.multiProcessorCount);
        printf("\tclockRate: %i\n", prop.clockRate);
        printf("\tconcurrent Kernels available: %i\n", (int)prop.concurrentKernels);
        printf("\tmaxGridSize: %i %i %i\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\tmaxThreadsDim: %i %i %i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("\tmaxThreadsPerBlock: %i\n", prop.maxThreadsPerBlock);
    }
    printf("\n");
#endif

}

