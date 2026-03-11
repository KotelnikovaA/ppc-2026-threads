#include "maslova_u_mult_matr_crs/omp/include/ops_omp.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "maslova_u_mult_matr_crs/common/include/common.hpp"
#include "util/include/util.hpp"

namespace maslova_u_mult_matr_crs {

MaslovaUMultMatrOMP::MaslovaUMultMatrOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaslovaUMultMatrOMP::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);

  if (a.cols != b.rows || a.rows <= 0 || b.cols <= 0) return false;
  if (a.row_ptr.size() != static_cast<size_t>(a.rows) + 1) return false;
  if (b.row_ptr.size() != static_cast<size_t>(b.rows) + 1) return false;
  return true;
}

bool MaslovaUMultMatrOMP::PreProcessingImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  auto &c = GetOutput();
  c.rows = a.rows;
  c.cols = b.cols;
  return true;
}

bool MaslovaUMultMatrOMP::RunImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  auto &c = GetOutput();

  const int rows_a = a.rows;
  const int cols_b = b.cols;

  c.row_ptr.assign(static_cast<size_t>(rows_a) + 1, 0);

#pragma omp parallel num_threads(ppc::util::GetNumThreads())
  {
    std::vector<int> marker(cols_b, -1);
#pragma omp for schedule(dynamic, 20)
    for (int i = 0; i < rows_a; ++i) {
      int row_nnz = 0;
      for (int j = a.row_ptr[i]; j < a.row_ptr[i + 1]; ++j) {
        int col_a = a.col_ind[j];
        for (int k = b.row_ptr[col_a]; k < b.row_ptr[col_a + 1]; ++k) {
          int col_b = b.col_ind[k];
          if (marker[col_b] != i) {
            marker[col_b] = i;
            row_nnz++;
          }
        }
      }
      c.row_ptr[i + 1] = row_nnz;
    }
  }

  for (int i = 0; i < rows_a; ++i) {
    c.row_ptr[i + 1] += c.row_ptr[i];
  }

  c.values.resize(c.row_ptr[rows_a]);
  c.col_ind.resize(c.row_ptr[rows_a]);

#pragma omp parallel num_threads(ppc::util::GetNumThreads())
  {
    std::vector<double> sparse_accumulator(cols_b, 0.0);
    std::vector<int> marker(cols_b, -1);
    std::vector<int> used_cols;
    used_cols.reserve(cols_b);

#pragma omp for schedule(dynamic, 20)
    for (int i = 0; i < rows_a; ++i) {
      used_cols.clear();
      for (int j = a.row_ptr[i]; j < a.row_ptr[i + 1]; ++j) {
        int col_a = a.col_ind[j];
        double val_a = a.values[j];
        for (int k = b.row_ptr[col_a]; k < b.row_ptr[col_a + 1]; ++k) {
          int col_b = b.col_ind[k];
          if (marker[col_b] != i) {
            marker[col_b] = i;
            used_cols.push_back(col_b);
            sparse_accumulator[col_b] = val_a * b.values[k];
          } else {
            sparse_accumulator[col_b] += val_a * b.values[k];
          }
        }
      }

      std::sort(used_cols.begin(), used_cols.end());

      int write_pos = c.row_ptr[i];
      for (int col : used_cols) {
        c.values[write_pos] = sparse_accumulator[col];
        c.col_ind[write_pos] = col;
        write_pos++;
        sparse_accumulator[col] = 0.0;
      }
    }
  }

  return true;
}

bool MaslovaUMultMatrOMP::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_mult_matr_crs