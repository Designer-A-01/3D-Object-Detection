#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// Device function
__global__ void FEV2TP_kernel(float *d_input, float *d_patch, 
                    double tp_center_lat, double tp_center_lon = 0)
{
    int cn = 3;

    const int patch_width = 704;
    // const int patch_height = 396;
    const int patch_height = 256;
    const int input_width = 3840;

    const double PI = 3.1415926535f;

    double o1_x = 960.0f;
    double o1_y = 960.0f;
    double o2_x = 2880.0f;
    double o2_y = 960.0f;

    double fov = (60.0f / 360.0f) * 2.0f * PI;
    double radius = 960.0f;
    double radius_tp = (patch_width / 2.0f) / tan(fov / 2.0f);

    int x, y;
    double lat, lon;
    double origin_x, origin_y;
    double tp_coord_x, tp_coord_y;

    int i = blockIdx.x;
    int j = threadIdx.x;

    const float mean[] = {123.67500305, 116.27999878, 103.52999878};
    const float stdinv[] = {0.01712475, 0.017507, 0.01742919};

    // 1. Get TP_coord
    tp_coord_x = j - patch_width / 2.0f;
    tp_coord_y = i - patch_height / 2.0f;

    lat = atan2(tp_coord_x, radius_tp) + tp_center_lat;
    lon = atan2(tp_coord_y, sqrt(pow(tp_coord_x, 2) + pow(radius_tp, 2))) + tp_center_lon;
    if (lat >= PI * 2.0f)
        lat -= PI * 2.0f;
    if (lat < 0)
        lat += PI * 2.0f;

    // find the center of fisheye
    if (0 <= lat && lat < PI / 2.0f || PI * 1.5f <= lat && lat < PI * 2.0f)
    {
        origin_x = o1_x;
        origin_y = o1_y;
    }
    else 
    {
        origin_x = o2_x;
        origin_y = o2_y;
        lat -= PI;
    }

    x = (int)(radius * sin(lat) * cos(lon) + origin_x);
    y = (int)(radius * sin(lon) + origin_y);


    // 2. Copy the pixels to patch 
    // Host is RGB format, remain into RGB format
    // d_patch[i*patch_width*cn + j*cn + 0] = (float)d_input[y*input_width*cn + x*cn + 2];
    // d_patch[i*patch_width*cn + j*cn + 1] = (float)d_input[y*input_width*cn + x*cn + 1];
    // d_patch[i*patch_width*cn + j*cn + 2] = (float)d_input[y*input_width*cn + x*cn + 0];

    d_patch[i*patch_width*cn + j*cn + 0] = d_input[y*input_width*cn + x*cn + 0];
    d_patch[i*patch_width*cn + j*cn + 1] = d_input[y*input_width*cn + x*cn + 1];
    d_patch[i*patch_width*cn + j*cn + 2] = d_input[y*input_width*cn + x*cn + 2];
    // printf("%f %f %f\n", d_input[y*input_width*cn + x*cn + 0], d_input[y*input_width*cn + x*cn + 1], d_input[y*input_width*cn + x*cn + 2]);

    // 3. Normalize with mean and std

    d_patch[i*patch_width*cn + j*cn + 0] -= mean[0];
    d_patch[i*patch_width*cn + j*cn + 1] -= mean[1];
    d_patch[i*patch_width*cn + j*cn + 2] -= mean[2];

    d_patch[i*patch_width*cn + j*cn + 0] *= stdinv[0];
    d_patch[i*patch_width*cn + j*cn + 1] *= stdinv[1];
    d_patch[i*patch_width*cn + j*cn + 2] *= stdinv[2];
}


__global__ void FEV2TRT_kernel(float *d_input, float *d_patch, int idx,
                    double tp_center_lat, double tp_center_lon = 0)
{
    int cn = 3;

    const int patch_width = 704;
    // const int patch_height = 396;
    const int patch_height = 256;
    const int input_width = 3840;

    const double PI = 3.1415926535f;

    double o1_x = 960.0f;
    double o1_y = 960.0f;
    double o2_x = 2880.0f;
    double o2_y = 960.0f;

    double fov = (60.0f / 360.0f) * 2.0f * PI;
    double radius = 960.0f;
    double radius_tp = (patch_width / 2.0f) / tan(fov / 2.0f);

    int x, y;
    double lat, lon;
    double origin_x, origin_y;
    double tp_coord_x, tp_coord_y;

    int i = blockIdx.x;
    int j = threadIdx.x;

    const float mean[] = {123.67500305, 116.27999878, 103.52999878};
    const float stdinv[] = {0.01712475, 0.017507, 0.01742919};

    // 1. Get TP_coord
    tp_coord_x = j - patch_width / 2.0f;
    tp_coord_y = i - patch_height / 2.0f;

    lat = atan2(tp_coord_x, radius_tp) + tp_center_lat;
    lon = atan2(tp_coord_y, sqrt(pow(tp_coord_x, 2) + pow(radius_tp, 2))) + tp_center_lon;
    if (lat >= PI * 2.0f)
        lat -= PI * 2.0f;
    if (lat < 0)
        lat += PI * 2.0f;

    // find the center of fisheye
    if (0 <= lat && lat < PI / 2.0f || PI * 1.5f <= lat && lat < PI * 2.0f)
    {
        origin_x = o1_x;
        origin_y = o1_y;
    }
    else 
    {
        origin_x = o2_x;
        origin_y = o2_y;
        lat -= PI;
    }

    x = (int)(radius * sin(lat) * cos(lon) + origin_x);
    y = (int)(radius * sin(lon) + origin_y);


    // 2. Copy the pixels to patch 
    // Host is RGB format, remain into RGB format
    // Structure of 1d-array: [n_patch, channel, patch_height, patch_width]
    int offset = idx * patch_width * patch_height * cn;

    d_patch[offset + i*patch_width*cn + j*cn + 0] = d_input[y*input_width*cn + x*cn + 0];
    d_patch[offset + i*patch_width*cn + j*cn + 1] = d_input[y*input_width*cn + x*cn + 1];
    d_patch[offset + i*patch_width*cn + j*cn + 2] = d_input[y*input_width*cn + x*cn + 2];

    // 3. Normalize with mean and std

    d_patch[offset + i*patch_width*cn + j*cn + 0] -= mean[0];
    d_patch[offset + i*patch_width*cn + j*cn + 1] -= mean[1];
    d_patch[offset + i*patch_width*cn + j*cn + 2] -= mean[2];

    d_patch[offset + i*patch_width*cn + j*cn + 0] *= stdinv[0];
    d_patch[offset + i*patch_width*cn + j*cn + 1] *= stdinv[1];
    d_patch[offset + i*patch_width*cn + j*cn + 2] *= stdinv[2];
}



// Host function
void FEV2TP_host(float *d_input, float *d_patch[])
{
    int n_patch = 6;
    int patch_width = 704;
    // int patch_height = 396;
    int patch_height = 256;

    const double PI = 3.1415926535f;

    double tp_center_lat;
    double lat_interval = 2.0f * PI / n_patch;
    
    // multi-thread possible?
    for (int i = 0; i < n_patch; i++) {
        tp_center_lat = i * lat_interval;

        FEV2TP_kernel<<<patch_height, patch_width>>>(d_input, d_patch[i], tp_center_lat);
        // FEV2TP_kernel<<<1089, patch_width>>>(d_input, d_patch[i], tp_center_lat);
        
    }
    printf("Copy finished\n");
    // cudaDeviceSynchronize();
}


void FEV2TRT_host(float *d_input, float *d_patch)
{
    int n_patch = 6;
    int patch_width = 704;
    // int patch_height = 396;
    int patch_height = 256;

    const double PI = 3.1415926535f;

    double tp_center_lat;
    double lat_interval = 2.0f * PI / n_patch;
    
    // multi-thread possible?
    for (int i = 0; i < n_patch; i++) {
        tp_center_lat = i * lat_interval;

        FEV2TRT_kernel<<<patch_height, patch_width>>>(d_input, d_patch, i, tp_center_lat);
        // FEV2TP_kernel<<<1089, patch_width>>>(d_input, d_patch[i], tp_center_lat);
        
    }
    printf("Copy finished\n");
    // cudaDeviceSynchronize();
}