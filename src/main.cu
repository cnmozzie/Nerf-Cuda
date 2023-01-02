/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   main.cu
 *  @author Hangkun
 */

#ifdef _WIN32
  #include <GL/gl3w.h>
#else
  #include <GL/glew.h>
#endif
#include <GLFW/glfw3.h>
#include <nerf-cuda/common_device.cuh>
#include <nerf-cuda/dlss.h>
#include <nerf-cuda/render_buffer.h>
#include <nerf-cuda/npy.hpp>

#include <cuda.h>
#include <filesystem/path.h>
#include <nerf-cuda/nerf_render.h>
#include <nerf-cuda/common.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <Eigen/Dense>
#include <args/args.hxx>
#include <iostream>
#include <string>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>

using namespace args;
using namespace std;
using namespace ngp;
using namespace tcnn;
namespace fs = ::filesystem;

void simple_glfw_error_callback(int error, const char* description) 
{
    std::cout << "GLFW error #" << error << ": " << description << std::endl;
}

__global__ void dlss_prep_kernel(
	uint32_t sample_index,
  Eigen::Vector2i resolution,
  Eigen::Vector2f focal_length,
  Eigen::Vector2f screen_center,
  Eigen::Vector3f parallax_shift,
	float* depth_buffer,
  Eigen::Matrix<float, 3, 4> camera,
  Eigen::Matrix<float, 3, 4> prev_camera,
	cudaSurfaceObject_t depth_surface,
	cudaSurfaceObject_t mvec_surface,
	cudaSurfaceObject_t exposure_surface
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	uint32_t x_orig = x;
	uint32_t y_orig = y;

	const float depth = depth_buffer[idx];
	Eigen::Vector2f mvec = {0, 0};
  /*
  Eigen::Vector2f mvec = motion_vector_3d(
		sample_index,
    {x, y},
    resolution,
    focal_length,
    camera,
    prev_camera,
    screen_center,
    parallax_shift,
		depth
	);*/

	surf2Dwrite(make_float2(mvec.x(), mvec.y()), mvec_surface, x_orig * sizeof(float2), y_orig);

	// Scale depth buffer to be guaranteed in [0,1].
	surf2Dwrite(std::min(std::max(depth / 128.0f, 0.0f), 1.0f), depth_surface, x_orig * sizeof(float), y_orig);

	// First thread write an exposure factor of 1. Since DLSS will run on tonemapped data,
	// exposure is assumed to already have been applied to DLSS' inputs.
	if (x_orig == 0 && y_orig == 0) {
		surf2Dwrite(1.0f, exposure_surface, 0, 0);
	}
}

Eigen::Vector2f calc_focal_length(const Eigen::Vector2i& resolution, int fov_axis, float zoom) {
	Eigen::Vector2f relative_focal_length = {3550.115, 3554.515};
  return relative_focal_length * resolution[fov_axis] * zoom;
}

void render_frame(const Eigen::Matrix<float, 3, 4>& camera_matrix0, const Eigen::Matrix<float, 3, 4>& prev_camera, ngp::CudaRenderBuffer& render_buffer, unsigned char *depth, unsigned char *rgb, int H, int W) 
{
    std::cout << "render frame begin" << std::endl;
    
    // CUDA stuff
	  tcnn::StreamAndEvent m_stream;
    render_buffer.clear_frame(m_stream.get());
    render_buffer.set_color_space(ngp::EColorSpace::Linear);
	  render_buffer.set_tonemap_curve(ngp::ETonemapCurve::Identity);

    std::cout << "load depth buffer..." << std::endl;
	  std::vector<float> data(H*W);
	  for (int i=0; i<H*W; i++) {
      data[i] = float(depth[i]);
    }
	  render_buffer.host_to_depth_buffer(data);

    // Prepare DLSS data: motion vectors, scaled depth, exposure
    std::cout << "prepare the dlss data..." << std::endl;
    auto res = render_buffer.in_resolution();
    uint32_t fov_axis = 1;
	  float zoom = 1.f;
    Eigen::Vector2f focal_length = calc_focal_length(render_buffer.in_resolution(), fov_axis, zoom);
    Eigen::Vector2f screen_center = Eigen::Vector2f::Constant(0.5f); // center of 2d zoom
    Eigen::Vector3f parallax_shift = {0.0f, 0.0f, 0.0f};
    //bool distortion = false;
    const dim3 threads = { 16, 8, 1 };
	  const dim3 blocks = { tcnn::div_round_up((uint32_t)res.x(), threads.x), tcnn::div_round_up((uint32_t)res.y(), threads.y), 1 };
    float m_dlss_sharpening = 0.0;
    uint32_t spp = 0;
    dlss_prep_kernel<<<blocks, threads, 0, m_stream.get()>>>(
			spp,
      res,
      focal_length,
      screen_center,
      parallax_shift,
			render_buffer.depth_buffer(),
      camera_matrix0,
      prev_camera,
			render_buffer.dlss()->depth(),
			render_buffer.dlss()->mvec(),
			render_buffer.dlss()->exposure()
	);
    render_buffer.set_dlss_sharpening(m_dlss_sharpening);

    std::cout << "run dlss..." << std::endl;
    float m_exposure = 0.0;
    Eigen::Array4f m_background_color = {0.0f, 0.0f, 0.0f, 1.0f};
    render_buffer.accumulate(m_exposure, m_stream.get());

    render_buffer.host_to_accumulate_buffer(rgb, H*W);

    render_buffer.tonemap(m_exposure, m_background_color, ngp::EColorSpace::Linear, m_stream.get());
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
}

int main(int argc, char** argv) {

  cout << "Hello, Metavese!" << endl;

  std::cout << "custom glfw init" << std::endl;
    glfwSetErrorCallback(simple_glfw_error_callback);
    if (!glfwInit()) {
		throw std::runtime_error{"GLFW could not be initialized."};
	}
    std::cout << "custom enable dlss" << std::endl;
    try {
		ngp::vulkan_and_ngx_init();
	} catch (const std::runtime_error& e) {
		tlog::warning() << "Could not initialize Vulkan and NGX. DLSS not supported. (" << e.what() << ")";
	}

  NerfRender* render = new NerfRender();
  string config_path = "freality.msgpack";
  render->reload_network_from_file(config_path);  // Init Model
  int scale = 4;
  Camera cam = {3550.115/scale, 3554.515/scale, 3010.45/scale, 1996.027/scale};
  Eigen::Matrix<float, 4, 4> start_cam_matrix, end_cam_matrix;
  start_cam_matrix << 0.15202418502518011, -0.037030882251750094, 0.9876828240656657, 4.584628725996896,
      0.988339491272669, 0.014375140391964181, -0.15158629664192835, -0.8852494170206848,
      -0.00858470495674949, 0.9992107230837872, 0.038784452328783726, 0.7921640240569777,
      0.0, 0.0, 0.0, 1.0;
  end_cam_matrix << -0.6748791705724709, 0.01465739161945682, 0.7377826685586811, 3.4630039248810722,
      0.7378504285927807, -0.0011163382655750731, 0.6749633314590863, 3.0284306856052963,
      0.010716816902559255, 0.9998919515062884, -0.010061569245718835, -0.2264369669902574,
      0.0, 0.0, 0.0, 1.0;
  Eigen::Vector2i resolution(6000/scale, 4000/scale);
  assert(resolution[0]*resolution[1]%NGPU==0);
  render -> set_resolution(resolution);
  clock_t start_t, end_t;

  int n_frames = 5;
  float shutter_fraction = 1;

  int in_height = resolution[1];
	int in_width = resolution[0];
  unsigned long out_height = in_height*2;
	unsigned long out_width = in_width*2;
  ngp::CudaRenderBuffer m_windowless_render_surface{std::make_shared<ngp::CudaSurface2D>()};
  m_windowless_render_surface.resize({in_width, in_height});
  // enable dlss
  tlog::info() << "custom enable dlss for render buffer";
	m_windowless_render_surface.enable_dlss({out_width, out_height});
  std::vector<float> result(out_height*out_width*4, 0.0);
  unsigned char* rgb_dlss = new unsigned char[out_height*out_width*3];

  auto prev_matrix = start_cam_matrix;
  for (int i = 0; i < n_frames; ++i) {
    float alpha = ((float)i)/(float)(n_frames-1) * shutter_fraction;
		//float end_alpha = ((float)i + 1.0f)/(float)n_frames * shutter_fraction;
    //auto sample_start_cam_matrix = log_space_lerp(start_cam_matrix, end_cam_matrix, start_alpha);
		auto sample_cam_matrix = log_space_lerp(start_cam_matrix, end_cam_matrix, alpha);
    //tlog::info() << sample_end_cam_matrix;
    start_t = clock();
    Image img = render->render_frame(cam, sample_cam_matrix);
    end_t = clock();
    double total_time = static_cast<double> (end_t - start_t) / 1 / CLOCKS_PER_SEC;
    printf("Process time : %f s / frame\n", total_time);

    string deep_file_name = "deep" + std::to_string(i) + ".png";
    string image_file_name = "image" + std::to_string(i) + ".png";

    stbi_write_png(deep_file_name.c_str(), img.W, img.H, 1, img.depth, img.W * 1);
    stbi_write_png(image_file_name.c_str(), img.W, img.H, 3, img.rgb, img.W * 3);
    
	  m_windowless_render_surface.reset_accumulation();
    
	  auto render_res = m_windowless_render_surface.in_resolution();
	  if (m_windowless_render_surface.dlss()) {
		  render_res = m_windowless_render_surface.dlss()->clamp_resolution(render_res);
	  }
	  m_windowless_render_surface.resize(render_res);

    start_t = clock();
    render_frame(sample_cam_matrix.block<3, 4>(0, 0), prev_matrix.block<3, 4>(0, 0), m_windowless_render_surface, img.depth, img.rgb, in_height, in_width);

    prev_matrix = sample_cam_matrix;

    //std::cout << "begin to transfer data..." << std::endl;

    cudaError_t x = cudaMemcpy2DFromArray(&result[0], out_width * sizeof(float) * 4, m_windowless_render_surface.surface_provider().array(), 0, 0, out_width * sizeof(float) * 4, out_height, cudaMemcpyDeviceToHost);
    CUDA_CHECK_THROW(x);

    for (int k=0; k<out_height*out_width; k++)
      for (int j=0; j<3; j++) {
        rgb_dlss[k*3+j] = (unsigned char)(result[k*4+j]*255);
      }

    end_t = clock();
    total_time = static_cast<double> (end_t - start_t) / 1 / CLOCKS_PER_SEC;
    printf("Process time : %f s / frame\n", total_time);
  
    string dlss_file_name = "dlss" + std::to_string(i) + ".png";
    stbi_write_png(dlss_file_name.c_str(), out_width, out_height, 3, rgb_dlss, out_width * 3);
  }


  
/*
  const std::vector<long unsigned> shape{out_height, out_width, 4};
	const bool fortran_order{false};
  const std::string path{"out.npy"};
	
	// try to save frame_buffer here?
	std::cout << "save frame buffer..." << std::endl;
	npy::SaveArrayAsNumpy(path, fortran_order, shape.size(), shape.data(), result);


  // save fig as numpy
  std::vector<unsigned char> image_data(img.rgb, img.rgb + img.W*img.H*3);

  const std::vector<long unsigned> shape_image{(unsigned long)(img.H), (unsigned long)(img.W), 3};
  const std::string path_image{"image.npy"};
	
	std::cout << "save image.npy..." << std::endl;
	npy::SaveArrayAsNumpy(path_image, fortran_order, shape_image.size(), shape_image.data(), image_data);

  std::vector<unsigned char> deep_data(img.depth, img.depth + img.W*img.H);

  const std::vector<long unsigned> shape_deep{(unsigned long)(img.H), (unsigned long)(img.W)};
  const std::string path_deep{"deep.npy"};
	
	std::cout << "save deep.npy..." << std::endl;
	npy::SaveArrayAsNumpy(path_deep, fortran_order, shape_deep.size(), shape_deep.data(), deep_data);

*/
}