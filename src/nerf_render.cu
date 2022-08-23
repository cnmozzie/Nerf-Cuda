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
#define PI acos(-1)

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

Eigen::Matrix<float, 4, 4> trans_t(float t){
  Eigen::Matrix<float, 4, 4> mat;
  mat << 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, t,
         0.0, 0.0, 0.0, 1.0;
  return mat;
}

Eigen::Matrix<float, 4, 4> rot_phi(float phi){
  Eigen::Matrix<float, 4, 4> mat;
  mat << 1.0, 0.0, 0.0, 0.0,
         0.0, cos(phi), -sin(phi), 0.0,
         0.0, sin(phi),  cos(phi), 0.0,
         0.0, 0.0, 0.0, 1.0;
  return mat;
}

Eigen::Matrix<float, 4, 4> rot_theta(float th){
  Eigen::Matrix<float, 4, 4> mat;
  mat << cos(th), 0.0, -sin(th), 0.0,
         0.0, 1.0, 0.0, 0.0,
         sin(th), 0.0,  cos(th), 0.0,
         0.0, 0.0, 0.0, 1.0;
  return mat;
}

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
  c2w = trans_t(radius);
  c2w = rot_phi(phi / 180. * (float)PI) * c2w;
  c2w = rot_theta(theta / 180. * (float)PI) * c2w;
  Eigen::Matrix<float, 4, 4> temp_mat;
  temp_mat << -1., 0., 0., 0.,
               0., 0., 1., 0.,
               0., 1., 0., 0.,
               0., 0., 0., 1.; 
  c2w = temp_mat * c2w; 
  return c2w;
}

// line 229-249 @ efficient-nerf-render-demo/example-app/example-app.cpp
__global__ void set_z_vals(float* z_vals, const int N) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= N) {
    return;
  }
  float near = 2.0; // val_dataset.near
  float far = 6.0; // val_dataset.far
  z_vals[index] = near + (far-near) / (N-1) * index;
}

// line 251-254 @ efficient-nerf-render-demo/example-app/example-app.cpp
__global__ void set_xyz(MatrixView<float> xyz, 
                        MatrixView<float> rays_o,
                        MatrixView<float> rays_d,
                        float* z_vals, 
                        int N_rays, 
                        int N_samples) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= N_rays || j >= N_samples) {
    return;
  }
  xyz(i*N_samples+j, 0) = rays_o(i, 0) + z_vals[j] * rays_d(i, 0);
  xyz(i*N_samples+j, 1) = rays_o(i, 1) + z_vals[j] * rays_d(i, 1);
  xyz(i*N_samples+j, 2) = rays_o(i, 2) + z_vals[j] * rays_d(i, 2);
}

// line 128-137 @ efficient-nerf-render-demo/example-app/example-app.cpp
__global__ void calc_index_coarse(MatrixView<int> ijk_coarse, MatrixView<float> xyz, int grid_coarse, const int N) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= N) {
    return;
  }
  float coord_scope = 3.0;
  float xyz_min = -coord_scope;
  float xyz_max = coord_scope;
  float xyz_scope = xyz_max - xyz_min;

  for (int i=0; i<3; i++) {
    ijk_coarse(index, i) = int((xyz(index, i) - xyz_min) / xyz_scope * grid_coarse);
    ijk_coarse(index, i) = ijk_coarse(index, i) < 0? 0 : ijk_coarse(index, i);
    ijk_coarse(index, i) = ijk_coarse(index, i) > grid_coarse-1? grid_coarse-1 : ijk_coarse(index, i);
  }

}

__global__ void query_coarse_sigma(float* sigmas, float* sigma_voxels_coarse, MatrixView<int> ijk_coarse, Eigen::Vector3i cg_s, const int N) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= N) {
    return;
  }
  int x = ijk_coarse(index, 0);
  int y = ijk_coarse(index, 1);
  int z = ijk_coarse(index, 2);
  sigmas[index] = sigma_voxels_coarse[x*cg_s[1]*cg_s[2] + y*cg_s[2] + z];
}

__global__ void set_dir(int w, int h, float focal, Eigen::Matrix<float, 4, 4> c2w, MatrixView<float> rays_o, MatrixView<float> rays_d){
  const int i = threadIdx.x + blockIdx.x * blockDim.x;   // h*w
  const int j = threadIdx.y + blockIdx.y * blockDim.y;   // w
  // tcnn::GPUMatrixDynamic<float> dirs(h * w, 3, tcnn::RM);
  if( i >= h*w || j >= w){
    return;
  }
  float dirs[3];
  dirs[0] = (float)((int)(i / w) - w/2) / focal;
  dirs[1] = (float)(j - h/2) / focal;
  dirs[2] = -1.;
  float sum = pow(dirs[0], 2) + pow(dirs[1], 2) + pow(dirs[2], 2);
  dirs[0] /= sum;
  dirs[1] /= sum;
  dirs[2] /= sum;
  // dirs((int)(i / w) * w + j, 0) = (float)((int)(i / w) - w/2) / focal;
  // dirs((int)(i / w) * w + j, 1) = (float)(j - h/2) / focal;
  // dirs((int)(i / w) * w + j, 2) = -1;
  // float sum =  dirs((int)(i / w) * w + j, 0) ^ 2 + dirs((int)(i / w) * w + j, 1) ^ 2 + dirs((int)(i / w) * w + j, 2) ^ 2;
  // dirs((int)(i / w) * w + j, 0) /= sum;
  // dirs((int)(i / w) * w + j, 1) /= sum;
  // dirs((int)(i / w) * w + j, 2) /= sum;

  // get_rays 
  // rays_d((int)(i / w) * w + j, 0) = c2w(0, 0) * dirs((int)(i / w) * w + j, 0) \
  //                                 + c2w(0, 1) * dirs((int)(i / w) * w + j, 1) \
  //                                 + c2w(0, 2) * dirs((int)(i / w) * w + j, 2);
  // rays_d((int)(i / w) * w + j, 1) = c2w(1, 0) * dirs((int)(i / w) * w + j, 0) \
  //                                 + c2w(1, 1) * dirs((int)(i / w) * w + j, 1) \
  //                                 + c2w(1, 2) * dirs((int)(i / w) * w + j, 2);
  // rays_d((int)(i / w) * w + j, 2) = c2w(2, 0) * dirs((int)(i / w) * w + j, 0) \
  //                                 + c2w(2, 1) * dirs((int)(i / w) * w + j, 1) \
  //                                 + c2w(2, 2) * dirs((int)(i / w) * w + j, 2);
  rays_d((int)(i / w) * w + j, 0) = c2w(0,0) * dirs[0] + c2w(0,1) * dirs[1] + c2w(0,2) * dirs[2];
  rays_d((int)(i / w) * w + j, 1) = c2w(1,0) * dirs[0] + c2w(1,1) * dirs[1] + c2w(1,2) * dirs[2];
  rays_d((int)(i / w) * w + j, 2) = c2w(2,0) * dirs[0] + c2w(2,1) * dirs[1] + c2w(2,2) * dirs[2];
  rays_o((int)(i / w) * w + j, 0) = c2w(0, 3);
  rays_o((int)(i / w) * w + j, 1) = c2w(1, 3);
  rays_o((int)(i / w) * w + j, 2) = c2w(2, 3);
}

__global__ void get_alphas(int len_z, float* z_vals, float* sigmas, float* alphas){
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index > len_z){
    return;
  }
  float delta_coarse;
  if(index < len_z) 
    delta_coarse = z_vals[index + 1] - z_vals[index];
  if(index == len_z)
    delta_coarse = 1e5;
  float alpha = 1.0 - exp(-delta_coarse * log(1 + std::exp(sigmas[index])));
  alpha = 1.0 - alpha + 1e-10;
  alphas[index] = alpha;
}

__global__ void get_cumprod(int len_z, float* alphas, float* alphas_cumprod){
  alphas[0] = 1.0;
  float cumprod = 1.0;
  for(int i = 0; i < len_z; i++){
    cumprod = cumprod * alphas[i];
    alphas_cumprod[i + 1] = cumprod;
  }
}

__global__ void get_weights(int len_z, float* alphas, float* alphas_cumprod, float* weights){
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index >= len_z){
    return;
  }
  weights[index] = alphas[index] * alphas_cumprod[index];
}

void sigma2weights(tcnn::GPUMemory<float>& weights,
                   tcnn::GPUMemory<float>& z_vals,
                   tcnn::GPUMemory<float>& sigmas) {
  // TODO
  // line 258-261 @ efficient-nerf-render-demo/example-app/example-app.cpp
  // use cuda to speed up
  int len_z = z_vals.size();
  std::cout << "len_z:" << len_z << std::endl;
  tcnn::GPUMemory<float> alphas(len_z);
  tcnn::GPUMemory<float> alphas_cumprod(len_z + 1);
  get_alphas<<<div_round_up(len_z, maxThreadsPerBlock), maxThreadsPerBlock>>>(len_z, z_vals.data(), sigmas.data(), alphas.data());
  std::cout << "get alphas" << std::endl;
  get_cumprod<<<1, 1>>>(len_z, alphas.data(), alphas_cumprod.data());
  get_weights<<<div_round_up(len_z, maxThreadsPerBlock), maxThreadsPerBlock>>>(len_z, alphas.data(), alphas_cumprod.data(), weights.data());
  std::cout << "sigma2weights" << std::endl;
}

__global__ void sum_rgbs(MatrixView<float> rgb_final, MatrixView<float> rgbs, float* weights, int N_samples, const int N) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) {
    return;
  }
  float weights_sum = 0;
  rgb_final(i, 0) = 0;
  rgb_final(i, 1) = 0;
  rgb_final(i, 2) = 0;
  for (int j=0; j<N_samples; j++) {
    weights_sum += weights[i*N_samples+j];
    rgb_final(i, 0) += weights[i*N_samples+j] * rgbs(i*N_samples+j, 0);
    rgb_final(i, 1) += weights[i*N_samples+j] * rgbs(i*N_samples+j, 1);
    rgb_final(i, 2) += weights[i*N_samples+j] * rgbs(i*N_samples+j, 2);
  }
  rgb_final(i, 0) = rgb_final(i, 0) + 1 - weights_sum;
  rgb_final(i, 1) = rgb_final(i, 1) + 1 - weights_sum;
  rgb_final(i, 2) = rgb_final(i, 2) + 1 - weights_sum;
}

void NerfRender::inference(int N_rays, int N_samples_,
                           tcnn::GPUMatrixDynamic<float>& rgb_fine,
                           tcnn::GPUMatrixDynamic<float>& xyz_,
                           tcnn::GPUMatrixDynamic<float>& dir_,
                           tcnn::GPUMemory<float>& z_vals,
                           tcnn::GPUMemory<float>& weights_coarse) {
  std::cout << "inference" << std::endl;
  tcnn::GPUMatrixDynamic<float> rgbs(N_rays*N_samples_, 3, tcnn::RM);
  tcnn::GPUMemory<float> sigmas(N_rays*N_samples_);
  // TODO
  // line 263-271 & 186-206 @ efficient-nerf-render-demo/example-app/example-app.cpp
  // use cuda to speed up

  tcnn::GPUMemory<float> weights(N_rays*N_samples_);
  sigma2weights(weights, z_vals, sigmas);

  sum_rgbs<<<div_round_up(N_rays, maxThreadsPerBlock), maxThreadsPerBlock>>> (rgb_fine.view(), rgbs.view(), weights.data(), N_samples_, N_rays);

}

void NerfRender::render_rays(int N_rays,
                             tcnn::GPUMatrixDynamic<float>& rgb_fine,
                             tcnn::GPUMatrixDynamic<float>& rays_o,
                             tcnn::GPUMatrixDynamic<float>& rays_d,
                             int N_samples=128, 
                             int N_importance=5, 
                             float perturb=0.) {
  std::cout << "render_rays" << std::endl;
  int N_samples_coarse = N_samples;
  tcnn::GPUMemory<float> z_vals_coarse(N_samples_coarse);
  set_z_vals<<<div_round_up(N_samples_coarse, maxThreadsPerBlock), maxThreadsPerBlock>>> (z_vals_coarse.data(), N_samples_coarse);
  int N_samples_fine = N_samples * N_importance;
  tcnn::GPUMemory<float> z_vals_fine(N_samples_fine);
  set_z_vals<<<div_round_up(N_samples_fine, maxThreadsPerBlock), maxThreadsPerBlock>>> (z_vals_fine.data(), N_samples_fine);

  tcnn::GPUMatrixDynamic<float> xyz_coarse(N_rays*N_samples_coarse, 3, tcnn::RM);
  dim3 threadsPerBlock(maxThreadsPerBlock/32, 32);
  dim3 numBlocks_coarse(div_round_up(N_rays, int(threadsPerBlock.x)), div_round_up(N_samples_coarse, int(threadsPerBlock.y)));
  set_xyz<<<numBlocks_coarse, threadsPerBlock>>>(xyz_coarse.view(), rays_o.view(), rays_d.view(), z_vals_coarse.data(), N_rays, N_samples_coarse);

  tcnn::GPUMatrixDynamic<float> xyz_fine(N_rays*N_samples_fine, 3, tcnn::RM);
  dim3 numBlocks_fine(div_round_up(N_rays, int(threadsPerBlock.x)), div_round_up(N_samples_fine, int(threadsPerBlock.y)));
  set_xyz<<<numBlocks_fine, threadsPerBlock>>>(xyz_fine.view(), rays_o.view(), rays_d.view(), z_vals_fine.data(), N_rays, N_samples_fine);

  // line 155-161 @ efficient-nerf-render-demo/example-app/example-app.cpp
  tcnn::GPUMatrixDynamic<int> ijk_coarse(N_rays*N_samples_coarse, 3, tcnn::RM);
  calc_index_coarse<<<div_round_up(N_rays*N_samples_coarse, maxThreadsPerBlock), maxThreadsPerBlock>>> (ijk_coarse.view(), xyz_coarse.view(), m_cg_s[0], N_rays*N_samples_coarse);
  tcnn::GPUMemory<float> sigmas(N_rays*N_samples_coarse);
  query_coarse_sigma<<<div_round_up(N_rays*N_samples_coarse, maxThreadsPerBlock), maxThreadsPerBlock>>> (sigmas.data(), m_sigma_voxels_coarse.data(), ijk_coarse.view(), m_cg_s, N_rays*N_samples_coarse);

  // line 261 @ efficient-nerf-render-demo/example-app/example-app.cpp
  tcnn::GPUMemory<float> weights_coarse(N_rays*N_samples_coarse);
  sigma2weights(weights_coarse, z_vals_coarse, sigmas);

  inference(N_rays, N_samples_fine, rgb_fine, xyz_fine, rays_d, z_vals_fine, weights_coarse);

  
  //float host_data[N_samples_fine];
  //z_vals_fine.copy_to_host(host_data);
  //std::cout << host_data[0] << " " << host_data[1] << " " << host_data[2] << std::endl;
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
  dim3 threadsPerBlock(maxThreadsPerBlock/32, 32);
  dim3 numBlocks(div_round_up(w * h, int(threadsPerBlock.x)), div_round_up(w, int(threadsPerBlock.y)));
  set_dir<<<numBlocks, threadsPerBlock>>>(w, h, focal, c2w, rays_o.view(), rays_d.view());
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

  tcnn::GPUMatrixDynamic<float> rgb_fine(N, 3, tcnn::RM);
  render_rays(N, rgb_fine, rays_o, rays_d, 128);

  // TODO
  // line 378-390 @ Nerf-Cuda/src/nerf_render.cu
  // save array as a picture

}

NGP_NAMESPACE_END