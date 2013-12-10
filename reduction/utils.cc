#include <iostream>
#include <fstream>
#include <cstdlib>
#include "defines.hh"
#include "utils.hh"

const int RAND_RANGE=128;

void
initGenerator( unsigned seed ){
    srand( seed );
}

void
genData( int* &data ){
    data = new int[ DATA_SIZE ];
    for( int i = 0; i != DATA_SIZE; ++i ){
        data[i] = -RAND_RANGE + static_cast<int>( rand() % (2 * RAND_RANGE - 1) );
    }
}

void
deleteData( int* &data ){
    delete [] data;
}

void
buildWithBinary( cl_program &mProgram, cl_context &mContext, const cl_device_id* const mDevice ){

    using std::cout;
    using std::endl;

    int err_code;
    size_t file_size;
    unsigned char* bin_content;
    std::fstream file;

    file.open("OpenCL/addReduce.cl.gpu_64.bc", std::ios::in );
    file.seekg(0, file.end);
    file_size = file.tellg();
    file.seekg(0, file.beg);

    bin_content = new unsigned char[ file_size ];
    file.read( (char*) bin_content, file_size );
    file.close();

    mProgram = clCreateProgramWithBinary( mContext, 1, mDevice, &file_size, (const unsigned char**) &bin_content, NULL, &err_code );
    if( err_code != CL_SUCCESS ){
        std::cout<<"Error building program"<<std::endl;
    }
    if( err_code == CL_INVALID_CONTEXT ){
        cout<<file_size<<endl;
    }

    delete [] bin_content;
    clBuildProgram( mProgram, 1, &mDevice[0], NULL, NULL, NULL );
}
