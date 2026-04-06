#include "borunov_v_complex_ccs/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#include "borunov_v_complex_ccs/common/include/common.hpp"
#include "util/include/util.hpp"

namespace borunov_v_complex_ccs {

BorunovVComplexCcsOMP::BorunovVComplexCcsOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(1);
}

bool BorunovVComplexCcsOMP::ValidationImpl() {
  const auto &a = GetInput().first;
  const auto &b = GetInput().second;
  if (a.num_cols != b.num_rows) {
    return false;
  }
  if (a.col_ptrs.size() != static_cast<std::size_t>(a.num_cols) + 1 ||
      b.col_ptrs.size() != static_cast<std::size_t>(b.num_cols) + 1) {
    return false;
  }
  return true;
}

bool BorunovVComplexCcsOMP::PreProcessingImpl() {
  const auto &a = GetInput().first;
  const auto &b = GetInput().second;
  auto &c = GetOutput()[0];

  c.num_rows = a.num_rows;
  c.num_cols = b.num_cols;
  c.col_ptrs.assign(c.num_cols + 1, 0);
  c.values.clear();
  c.row_indices.clear();

  return true;
}

bool BorunovVComplexCcsOMP::RunImpl() {
  const auto &a = GetInput().first;
  const auto &b = GetInput().second;
  auto &c = GetOutput()[0];

  const int num_threads = ppc::util::GetNumThreads();
  const int bc = b.num_cols;

  // Per-thread storage: each thread accumulates results for its column range
  std::vector<std::vector<std::complex<double>>> t_values(num_threads);
  std::vector<std::vector<int>> t_row_indices(num_threads);
  // t_col_nnz[tid][j - jstart] = nnz count for output column j
  std::vector<std::vector<int>> t_col_nnz(num_threads);

#pragma omp parallel default(none) shared(a, b, bc, t_values, t_row_indices, t_col_nnz) \
    num_threads(ppc::util::GetNumThreads())
  {
    const int tid = omp_get_thread_num();
    const int nt = omp_get_num_threads();
    const int jstart = (tid * bc) / nt;
    const int jend = ((tid + 1) * bc) / nt;

    t_col_nnz[tid].assign(jend - jstart, 0);

    // Thread-local working arrays (no false sharing, no synchronization needed)
    std::vector<std::complex<double>> acc(a.num_rows, {0.0, 0.0});
    std::vector<int> marker(a.num_rows, -1);
    std::vector<int> touched;
    touched.reserve(static_cast<std::size_t>(a.num_rows));

    for (int j = jstart; j < jend; ++j) {
      touched.clear();

      // Accumulate contributions from each non-zero in column j of B
      for (int bk = b.col_ptrs[j]; bk < b.col_ptrs[j + 1]; ++bk) {
        const int p = b.row_indices[bk];
        const std::complex<double> bval = b.values[bk];

        for (int ak = a.col_ptrs[p]; ak < a.col_ptrs[p + 1]; ++ak) {
          const int i = a.row_indices[ak];
          acc[i] += a.values[ak] * bval;
          if (marker[i] != j) {
            marker[i] = j;
            touched.push_back(i);
          }
        }
      }

      // Sort rows for canonical CCS output
      std::ranges::sort(touched);

      // Flush non-zero results to per-thread storage
      for (int i : touched) {
        if (std::abs(acc[i]) > 1e-9) {
          t_values[tid].push_back(acc[i]);
          t_row_indices[tid].push_back(i);
          ++t_col_nnz[tid][j - jstart];
        }
        acc[i] = {0.0, 0.0};
      }
    }
  }

  // Sequential merge: build final col_ptrs and concatenate per-thread data
  for (int tid = 0; tid < num_threads; ++tid) {
    const int jstart = (tid * bc) / num_threads;
    const int jend = ((tid + 1) * bc) / num_threads;
    for (int j = jstart; j < jend; ++j) {
      c.col_ptrs[j + 1] = c.col_ptrs[j] + t_col_nnz[tid][j - jstart];
    }
  }

  const int total_nnz = c.col_ptrs[bc];
  c.values.reserve(static_cast<std::size_t>(total_nnz));
  c.row_indices.reserve(static_cast<std::size_t>(total_nnz));

  for (int tid = 0; tid < num_threads; ++tid) {
    c.values.insert(c.values.end(), t_values[tid].begin(), t_values[tid].end());
    c.row_indices.insert(c.row_indices.end(), t_row_indices[tid].begin(), t_row_indices[tid].end());
  }

  return true;
}

bool BorunovVComplexCcsOMP::PostProcessingImpl() {
  return true;
}

}  // namespace borunov_v_complex_ccs
