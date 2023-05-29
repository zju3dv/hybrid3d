#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <set>
#include <tuple>
#include <algorithm>
#include <random>
#include <functional>
#include <chrono>
#include <Eigen/Eigen>

#define print_var(x) std::cout << #x << " = " << x << std::endl;

using Clock = std::chrono::high_resolution_clock;

inline double tloop(const Clock::time_point &t_start, const Clock::time_point &t_end) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
}

inline Eigen::Vector2i round_to_image(float x, float y, int W, int H) {
    int xx = std::min(std::max(0, int(std::lround(x))), W - 1);
    int yy = std::min(std::max(0, int(std::lround(y))), H - 1);
    return Eigen::Vector2i(xx, yy);
}

// nominmax suppression over image plane
// @param in_corners(x, y, score)
// @retval survival corners and its indices
std::tuple<std::vector<Eigen::Vector3f>, std::vector<int>>
nms(const std::vector<Eigen::Vector3f> &in_corners, int H, int W, int grid_size, int verbose) {
    std::vector<Eigen::Vector3f> selected_keypoints;
    selected_keypoints.reserve(2048);
    std::vector<int> selected_indices;
    selected_indices.reserve(2048);
    Eigen::MatrixXi grid;
    grid.resize(H, W);
    // grid 0->unprocessed; 1->empty or suppressed; -1->selected
    grid.setZero();
    size_t n = 0;
    std::vector<size_t> idx_sorted;
    idx_sorted.resize(in_corners.size());
    std::generate(std::begin(idx_sorted),
                  std::end(idx_sorted),
                  [&n] { return n++; });
    std::sort(std::begin(idx_sorted),
              std::end(idx_sorted),
              [&in_corners](size_t i1, size_t i2) {
                  return in_corners[i1].z() > in_corners[i2].z();
              });
    for (size_t i = 0; i < idx_sorted.size(); ++i) {
        size_t corner_idx = idx_sorted[i];
        int r = std::lround(in_corners[corner_idx].y());
        int c = std::lround(in_corners[corner_idx].x());
        if (grid(r, c) == 0) {
            selected_keypoints.emplace_back(in_corners[corner_idx]);
            selected_indices.emplace_back(corner_idx);
            // fill nearby grids
            int min_r = std::max(0, r - grid_size);
            int min_c = std::max(0, c - grid_size);
            int max_r = std::min(H - 1, r + grid_size);
            int max_c = std::min(W - 1, c + grid_size);
            grid.block(min_r, min_c, max_r - min_r + 1, max_c - min_c + 1).setConstant(1);
            grid(r, c) = -1;
        }
    }
    // selected_keypoints.shrink_to_fit();
    // selected_indices.shrink_to_fit();
    return {selected_keypoints, selected_indices};
}

namespace py = pybind11;

PYBIND11_MODULE(c_utils, m) {
    m.def("nms", &nms,
          py::arg("in_corners"),
          py::arg("H"),
          py::arg("W"),
          py::arg("grid_size"),
          py::arg("verbose")=0);
}
