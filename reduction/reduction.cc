#include <iostream>
#include "utils.hh"
#include "defines.hh"
#include "reduction.hh"
#include <pthread.h>

size_t GROUP_SIZE;
size_t INPUT_SIZE;
/*
 * The following function is only used to check whether certain command is correctly executed
 * For more information, please refers to
 *     http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/
 */

struct ComputeData {
    int* ptr;
    unsigned long length;
    int result;
    ComputeData():ptr(NULL),length(0),result(0){}
};

void* ComputeResult(void* ptr) {
    ComputeData* data=(ComputeData*)ptr;
    for (unsigned i=0; i<data->length; i++) {
        data->result+=data->ptr[i];
    }
    return NULL;
}

int CPUComputed(int* data_ptr) {
    
    const unsigned No_thread=8;
    pthread_t thread[No_thread];
    ComputeData dataSet[No_thread];
    
    unsigned long offset=INPUT_SIZE/8;
    dataSet[0].ptr=&(data_ptr[0]);         dataSet[0].length=offset;
    dataSet[1].ptr=&(data_ptr[offset]);    dataSet[1].length=offset;
    dataSet[2].ptr=&(data_ptr[2*offset]);  dataSet[2].length=offset;
    dataSet[3].ptr=&(data_ptr[3*offset]);  dataSet[3].length=offset;
    dataSet[4].ptr=&(data_ptr[4*offset]);  dataSet[4].length=offset;
    dataSet[5].ptr=&(data_ptr[5*offset]);  dataSet[5].length=offset;
    dataSet[6].ptr=&(data_ptr[6*offset]);  dataSet[6].length=offset;
    dataSet[7].ptr=&(data_ptr[7*offset]);  dataSet[7].length=offset+INPUT_SIZE%8;
    
    for (unsigned i=0; i<No_thread; i++) {
        pthread_create(&thread[i], NULL, ComputeResult,(void*)&(dataSet[i]));
    }
    for (unsigned i=0; i<No_thread; i++) {
        pthread_join(thread[i], NULL);
    }
    
    int result=0;
    
    for (unsigned i=0; i<No_thread; i++) {
        result+=dataSet[i].result;
    }
    return result;
}



void
errVerify( cl_int status ){
    using std::cerr;
    using std::endl;
    if( status != CL_SUCCESS ){
        cerr<<"There is something error"<<endl;
    }
}

void
OclAddReduce::run(){
    INPUT_SIZE=DATA_SIZE;
    result=0;
    IsCPU=false;
    
//    if (INPUT_SIZE<100000000) {
//        IsCPU=true;
//        if (INPUT_SIZE<1000) {
//            for (unsigned i=0; i<INPUT_SIZE; i++) {
//                result+=mHostData[i];
//            }
//            return;
//        }
//        result=CPUComputed(mHostData);
//        return;
//    }
    
    
    /*Step 1: dectect & initialize platform*/
    initPlatform();

    /*Step 2: detect & initialize device*/
    initDevice();

    /*TA's Information Show Function*/
    showInfo();
    
    GROUP_SIZE=8;
    /*Step 3: create a context*/
    initContext();

    /*Step 4: create a command queue*/
    initCommandQ();

    /*Step 5: create device buffers*/
    initDeviceMem();

    /*Step 6: build program, and then create & set kernel*/
    initKernel();

    /*Step 7: run kernel*/
    runKernel();
    
    //storeResult
//    clEnqueueReadBuffer(mCommandQ, dev_B, CL_TRUE, 0, sizeof(int), &result, 0, 0, 0);
//    result+=value_offset;
}

int
OclAddReduce::getResult(){
    return result;
}

void
OclAddReduce::initPlatform(){
    cl_uint numPlatforms = 0;

    clGetPlatformIDs( 0, NULL, &numPlatforms );
    mPlatform = new cl_platform_id[ numPlatforms ];
    clGetPlatformIDs( numPlatforms, mPlatform, NULL );
}

void
OclAddReduce::initDevice(){
    cl_uint numDevices = 0;

    clGetDeviceIDs( mPlatform[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices );
    mDevice = new cl_device_id[ numDevices ];
    clGetDeviceIDs( mPlatform[0], CL_DEVICE_TYPE_GPU, numDevices, mDevice, NULL );
}

void
OclAddReduce::showInfo(){
    using std::cout;
    using std::endl;

    size_t msg_size;
    char* msg;

    cout<<"*********************OpenCL Information*********************"<<endl;
    clGetPlatformInfo( *mPlatform, CL_PLATFORM_VENDOR, 0, NULL, &msg_size );
    msg = new char[ msg_size ];
    clGetPlatformInfo( *mPlatform, CL_PLATFORM_VENDOR, msg_size, (void*) msg, NULL );
    cout<<"Vendor: "<<msg<<endl;
    delete [] msg;

    clGetPlatformInfo( *mPlatform, CL_PLATFORM_VERSION, 0, NULL, &msg_size );
    msg = new char[ msg_size ];
    clGetPlatformInfo( *mPlatform, CL_PLATFORM_VERSION, msg_size, (void*) msg, NULL );
    cout<<"Version info: "<<msg<<endl;
    delete [] msg;

    clGetDeviceInfo( *mDevice, CL_DEVICE_NAME, 0, NULL, &msg_size );
    msg = new char[ msg_size ];
    clGetDeviceInfo( *mDevice, CL_DEVICE_NAME, msg_size, (void*) msg, NULL );
    cout<<"Device: "<<msg<<endl;

    cl_bool ava;
    clGetDeviceInfo( *mDevice, CL_DEVICE_AVAILABLE, sizeof( cl_bool), (void*) &ava, NULL);
    cout<<"Available: ";
    if( ava == CL_TRUE ){
        cout<<"YES"<<endl;
    } else {
        cout<<"NO"<<endl;
    }

    cout<<"************************************************************"<<endl;
    cout<<endl;
}

void
OclAddReduce::initContext(){
    cl_int status;
    mContext = clCreateContext( NULL, 1, mDevice, NULL, NULL, &status );
    errVerify( status );
}

void
OclAddReduce::initCommandQ(){
    mCommandQ = clCreateCommandQueue( mContext, mDevice[0], 0, NULL );
}


void
OclAddReduce::initDeviceMem(){
    int offset=(int)(INPUT_SIZE)%GROUP_SIZE;
    value_offset=0;
    INPUT_SIZE-=offset;
    for (unsigned i=0; i<offset; i++) {
        value_offset+=mHostData[i];
    }
    
    dev_A = clCreateBuffer( mContext, CL_MEM_READ_ONLY, INPUT_SIZE * sizeof(int), NULL, NULL );
    clEnqueueWriteBuffer(mCommandQ, dev_A, CL_FALSE, 0, INPUT_SIZE * sizeof(int), &mHostData[offset], 0, NULL, NULL);
    dev_B = clCreateBuffer( mContext, CL_MEM_READ_WRITE, INPUT_SIZE * sizeof(int), NULL, NULL );
    dev_C = clCreateBuffer( mContext, CL_MEM_READ_WRITE, INPUT_SIZE * sizeof(int), NULL, NULL );
}

void
OclAddReduce::initKernel(){

    // build program with binary
    // please use program "m2c" and do not rename addReduce.cl
    buildWithBinary( mProgram, mContext, mDevice );

    // create kernel
    mKernel = clCreateKernel(mProgram, "addreduce", 0);
    // setting kernel arguments
    clSetKernelArg(mKernel, 0, sizeof(cl_mem), &dev_A);
    clSetKernelArg(mKernel, 1, sizeof(cl_mem), &dev_B);
    clSetKernelArg(mKernel, 2, GROUP_SIZE*sizeof(int), NULL);
}

void
OclAddReduce::runKernel(){
    size_t work_size=INPUT_SIZE;
    size_t local_size=GROUP_SIZE;

    clEnqueueNDRangeKernel(mCommandQ, mKernel, 1, 0, &work_size, &local_size, 0, 0, 0);
    
    work_size=work_size/local_size;
    
    while (work_size>1000) {
        std::swap(dev_B,dev_C);
        size_t offset=work_size%local_size;
        work_size+=local_size-offset;
        clSetKernelArg(mKernel, 0, sizeof(cl_mem), &dev_C);
        clSetKernelArg(mKernel, 1, sizeof(cl_mem), &dev_B);
        clSetKernelArg(mKernel, 2, GROUP_SIZE*sizeof(int), NULL);
        clEnqueueNDRangeKernel(mCommandQ, mKernel, 1, 0, &work_size, &local_size, 0, 0, 0);
        work_size=work_size/local_size;
    }
    
    int *data_back=new int[work_size];
    clEnqueueReadBuffer( mCommandQ, dev_B, CL_TRUE, 0, work_size*sizeof(int), data_back, 0, 0, 0);
    result=value_offset;
    for (unsigned i=0; i<work_size; i++) {
        result+=data_back[i];
    }
    delete [] data_back;
}

void
OclAddReduce::clear(){
    if (IsCPU) {
        return;
    }
    clReleaseMemObject(dev_A);
    clReleaseMemObject(dev_B);
    clReleaseMemObject(dev_C);
    /* Release the memory*/
    clReleaseKernel( mKernel );
    clReleaseProgram( mProgram );
    clReleaseCommandQueue( mCommandQ );
    clReleaseContext( mContext );
    if( mDevice != NULL ) delete [] mDevice;
    if( mPlatform != NULL ) delete [] mPlatform;
}
