#include <cuda.h>
#include <filesystem/directory.h>
#include <filesystem/path.h>
#include <nerf-cuda/common.h>
#include <nerf-cuda/nerf_network.h>
#include <nerf-cuda/nerf_render.h>
#include <nerf-cuda/render_utils.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <json/json.hpp>
#include <nerf-cuda/common_device.cuh>
#include <set>
#include <typeinfo>
#include <vector>

#define maxThreadsPerBlock 1024

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

NerfRender::NerfRender() {
  std::cout << "Hello, NerfRender!" << std::endl;
}

NerfRender::~NerfRender() {}

void NerfRender::load_nerf_tree(long* index_voxels_coarse_h,
                                float* sigma_voxels_coarse_h,
                                float* voxels_fine_h,
                                const int64_t* cg_s,
                                const int64_t* fg_s) {
  std::cout << "load_nerf_tree" << std::endl;
  m_cg_s << cg_s[0], cg_s[1], cg_s[2];
  m_fg_s << fg_s[0], fg_s[1], fg_s[2], fg_s[3], fg_s[4];
  std::cout << m_cg_s << std::endl;
  std::cout << m_fg_s << std::endl;

  int coarse_grid_num = m_cg_s[0] * m_cg_s[1] * m_cg_s[2];
  std::cout << "coarse_grid_num: " << coarse_grid_num << std::endl;

  int fine_grid_num = m_fg_s[0] * m_fg_s[1] * m_fg_s[2] * m_fg_s[3] * m_fg_s[4];
  std::cout << "coarse_grid_num: " << fine_grid_num << std::endl;

  m_index_voxels_coarse.resize(coarse_grid_num);
  m_index_voxels_coarse.copy_from_host(index_voxels_coarse_h);
  m_sigma_voxels_coarse.resize(coarse_grid_num);
  m_sigma_voxels_coarse.copy_from_host(sigma_voxels_coarse_h);
  m_voxels_fine.resize(fine_grid_num);
  m_voxels_fine.copy_from_host(voxels_fine_h);
  
  float host_data[3] = {0, 0, 0};
  m_sigma_voxels_coarse.copy_to_host(host_data,3);
  std::cout << "host_data[1]: " << host_data[1] << std::endl;
}

Eigen::Matrix<float, 4, 4> pose_spherical(float theta, float phi, float radius) {
  Eigen::Matrix<float, 4, 4> c2w;
  // TODO
  // line 90 @ efficient-nerf-render-demo/example-app/example-app.cpp
  return c2w;
}

// line 229-249 @ efficient-nerf-render-demo/example-app/example-app.cpp
__global__ void set_z_vals(float* z_vals, const int N ) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= N) {
    return;
  }
  float near = 2.0; // val_dataset.near
  float far = 6.0; // val_dataset.far
  z_vals[index] = near + (far-near) / (N-1) * index;
}

// line 251-254 @ efficient-nerf-render-demo/example-app/example-app.cpp
__global__ void set_xyz_coarse(MatrixView<float> xyz_coarse, 
                               MatrixView<float> rays_o,
                               MatrixView<float> rays_d,
                               float* z_vals, 
                               int N_rays, 
                               int N_samples_coarse) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= N_rays || j >= N_samples_coarse) {
    return;
  }
  xyz_coarse(i*N_samples_coarse+j, 0) = rays_o(i, 0) + z_vals[j] * rays_d(i, 0);
  xyz_coarse(i*N_samples_coarse+j, 1) = rays_o(i, 1) + z_vals[j] * rays_d(i, 1);
  xyz_coarse(i*N_samples_coarse+j, 2) = rays_o(i, 2) + z_vals[j] * rays_d(i, 2);
}

void NerfRender::render_rays(int N_rays,
                             tcnn::GPUMatrixDynamic<float>& rays_o,
                             tcnn::GPUMatrixDynamic<float>& rays_d,
                             int N_samples=128, 
                             int N_importance=5, 
                             float perturb=0.) {
  std::cout << "render_rays" << std::endl;
  int N_samples_coarse = N_samples;
  tcnn::GPUMemory<float> z_vals_coarse(N_samples_coarse);
  set_z_vals<<<div_round_up(N_samples_coarse, maxThreadsPerBlock), maxThreadsPerBlock>>> (z_vals_coarse.data(), N_samples_coarse);

  tcnn::GPUMatrixDynamic<float> xyz_coarse(N_rays*N_samples_coarse, 3, tcnn::RM);
  dim3 threadsPerBlock(maxThreadsPerBlock/32, 32);
  dim3 numBlocks(div_round_up(N_rays, int(threadsPerBlock.x)), div_round_up(N_samples_coarse, int(threadsPerBlock.y)));
  set_xyz_coarse<<<numBlocks, threadsPerBlock>>>(xyz_coarse.view(), rays_o.view(), rays_d.view(), z_vals_coarse.data(), N_rays, N_samples_coarse);

  
  float host_data[N_samples_coarse];
  z_vals_coarse.copy_to_host(host_data);
  std::cout << host_data[0] << " " << host_data[1] << " " << host_data[2] << std::endl;
}

void NerfRender::generate_rays(int w,
                               int h,
                               float focal,
                               Eigen::Matrix<float, 4, 4> c2w,
                               tcnn::GPUMatrixDynamic<float>& rays_o,
                               tcnn::GPUMatrixDynamic<float>& rays_d) {
  // TODO
  // line 287-292 @ efficient-nerf-render-demo/example-app/example-app.cpp
  // use cuda to speed up
  tlog::info() << c2w;
}

void NerfRender::render_frame(int w, int h, float theta, float phi, float radius) {
  auto c2w = pose_spherical(90, -30, 4);
  float focal = 0.5 * w / std::tan(0.5*0.6911112070083618);
  int N = w * h;  // number of pixels
  // initial points corresponding to pixels, in world coordination
  tcnn::GPUMatrixDynamic<float> rays_o(N, 3, tcnn::RM);
  // direction corresponding to pixels,in world coordination
  tcnn::GPUMatrixDynamic<float> rays_d(N, 3, tcnn::RM);

  generate_rays(w, h, focal, c2w, rays_o, rays_d);

  render_rays(N, rays_o, rays_d, 128);

}

NGP_NAMESPACE_END