/*Write your kernel here*/
__kernel void addreduce(__global int* a, __global int* result, __local int* ldata)
{
    int tid = get_global_id(0);
    int tsize = get_global_size(0);
    
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int gid = get_group_id(0);
    int gsize = get_num_groups(0);
    
    ldata[lid]=a[tid];
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    int overflow=gsize%64;
    
    if(tid==0) {
        int data=0;
        for(unsigned i=0;i<lsize;i++) {
            data+=ldata[i];
        }
        result[gid]=data;
        for(unsigned i=0;i<(64-overflow);i++) {
            result[gsize+i]=0;
        }
    }
    else if(lid==0) {
        int data=0;
        for(unsigned i=0;i<lsize;i++) {
            data+=ldata[i];
        }
        result[gid]=data;
    }
    
    
}
