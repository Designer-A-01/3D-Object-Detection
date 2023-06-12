#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <cassert>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <cuda_runtime.h>

#include "FEV2TP.cuh"

using namespace cv;
using namespace std;

typedef struct {
    float *data;
    int width;
    int height;
} Frame_float;

extern "C" {

const int n_patch = 6;
const int channels = 3;

int fisheye_width = 3840;
int fisheye_height = 1920;

int tangent_width = 704;
// int tangent_height = 396;
int tangent_height = 256;

int input_size = fisheye_width * fisheye_height * channels;
int patch_size = tangent_width * tangent_height * channels;

Mat input_image, patches[n_patch];
Frame_float *h_input, h_patch[n_patch];    // host input, patch
float *d_input, *d_patch[n_patch], *d_patch_trt;    // device input, patch

// Frame_uint8 h_patch[n_patch];
// uint8_t *d_patch[n_patch];


void initHostDeviceMemory()
{
    // Initialize the Host memory
    for (int i = 0; i < n_patch; i++) {
        h_patch[i].data = (float*)malloc(patch_size * sizeof(float));
        h_patch[i].width = tangent_width;
        h_patch[i].height = tangent_height;
    }
    cout << "Allocating Host Memory completed" << endl;

    // Allocating Device(GPU) memory for input image and tangent patch
    cudaMalloc((void**)&d_input, input_size*sizeof(float));

    cudaMalloc((void**)&d_patch_trt, patch_size*n_patch*sizeof(float)); // memory for trt patch

    cudaMalloc((void***)&d_patch, n_patch*sizeof(float*));
    for (int i = 0; i < 6; i++) {
        cudaMalloc((void**)&d_patch[i], patch_size*sizeof(float));
    }
    cout << "Allocating Device Memory completed" << endl;

   
}

void convertFEV2TP(Frame_float *frame)
{
    // ============== Convert to tangent patch ==============

    // Check frame passed by python
    assert(frame->height != 0 && frame->width != 0);

    // -------------- Copy input image to GPU memory (3ms) --------------
    cudaMemcpy(d_input, frame->data, input_size*sizeof(float), cudaMemcpyHostToDevice);

    // -------------- Copy input image pixel to patch (0.08ms) --------------
    FEV2TP_host(d_input, d_patch);
}

void convertFEV2TRTInput(Frame_float *frame)
{
    // ============== Convert to tangent patch ==============

    // Check frame passed by python
    assert(frame->height != 0 && frame->width != 0);

    // -------------- Copy input image to GPU memory (3ms) --------------
    cudaMemcpy(d_input, frame->data, input_size*sizeof(float), cudaMemcpyHostToDevice);

    // -------------- Copy input image pixel to patch (0.08ms) --------------
    FEV2TRT_host(d_input, d_patch_trt);
}

float** getDevicePatchPtr(Frame_float *frame)
{
    convertFEV2TRTInput(frame);
    
    return d_patch;
}


Frame_float* getHostPatchPtr(Frame_float *frame)
{
    convertFEV2TP(frame);

    // -------------- Copy patch image to CPU memory (22ms ~ 34ms) --------------
    for (int i = 0; i < n_patch; i++) {
        cudaMemcpy(h_patch[i].data, d_patch[i], patch_size*sizeof(float), cudaMemcpyDeviceToHost);
    }

    return h_patch;
}
}