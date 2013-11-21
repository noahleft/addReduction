#ifndef __REDUCTION_HH_UQLYSFW6__
#define __REDUCTION_HH_UQLYSFW6__

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

class OclAddReduce{
    public:
        OclAddReduce( int* host_data )
            : mHostData( host_data )
        {};
        ~OclAddReduce(){
            clear();
        };
        void run();
        int getResult();
    private:
        OclAddReduce( const OclAddReduce & );
        const OclAddReduce& operator=( const OclAddReduce & );
        void initHost();
        void endHost();
        void initPlatform();
        void initDevice();
        void showInfo();
        void initContext();
        void initCommandQ();
        void initDeviceMem();
        void initKernel();
        void runKernel();
        void clear();

        /* Data */
        int* mHostData;
        cl_platform_id *mPlatform;
        cl_device_id *mDevice;
        cl_context mContext;
        cl_command_queue mCommandQ;
        cl_program mProgram;
        cl_kernel mKernel;

        /*Custom Data*/
        cl_mem dev_A;
        cl_mem dev_B;
        cl_mem dev_C;
        int value_offset;
        int result;
        bool IsCPU;
};

#endif /* end of include guard: REDUCTION_HH_UQLYSFW6 */

