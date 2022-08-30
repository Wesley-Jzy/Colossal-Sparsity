#include <cuda_runtime.h>

#include <cusparse.h>

#include <iostream>

// A simple program to make sure CUSPARSE is working from C.

// Compile with: nvcc cusparse_available.cu -o cusparse -lcusparse

int main(int argc, char** argv) {

	cusparseHandle_t context = 0;
	std::cout << "Making context..." << std::endl;

	cusparseStatus_t status = cusparseCreate(&context);

	std::cout << "Status: " << status << std::endl;

	if(status != CUSPARSE_STATUS_SUCCESS) {

		std::cout << "Failed!" << std::endl;

	}

	

	int cusparseVersion = 0;
    int cudartVersion = 0;
    int cudadriVersion = 0;

	std::cout << "Getting version..." << std::endl;

	status = cusparseGetVersion(context, &cusparseVersion);
    cudaRuntimeGetVersion(&cudartVersion);
    cudaDriverGetVersion(&cudadriVersion);

	std::cout << "cusparse Version: " << cusparseVersion << std::endl;
    std::cout << "cudart Version: " << cudartVersion << std::endl;
    std::cout << "cudadriver Version: " << cudadriVersion << std::endl;

	std::cout << "Status: " << status << std::endl;

	if(status != CUSPARSE_STATUS_SUCCESS) {

		std::cout << "Failed!" << std::endl;

	}

	

	std::cout << "Destroying context..." << std::endl;

	status = cusparseDestroy(context);

	std::cout << "Status: " << status << std::endl;

	if(status != CUSPARSE_STATUS_SUCCESS) {

		std::cout << "Failed!" << std::endl;

	}

}