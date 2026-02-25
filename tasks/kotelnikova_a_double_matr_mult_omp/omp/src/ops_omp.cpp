#include "kotelnikova_a_double_matr_mult_omp/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "kotelnikova_a_double_matr_mult_omp/common/include/common.hpp"

namespace kotelnikova_a_double_matr_mult_omp {

KotelnikovaATaskOMP::KotelnikovaATaskOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = SparseMatrixCCS();
}

bool KotelnikovaATaskOMP::IsMatrixValid(const SparseMatrixCCS &matrix) {
  if (matrix.rows < 0 || matrix.cols < 0) {
    return false;
  }
  if (matrix.col_ptrs.size() != static_cast<size_t>(matrix.cols) + 1) {
    return false;
  }
  if (matrix.values.size() != matrix.row_indices.size()) {
    return false;
  }

  if (matrix.col_ptrs.empty() || matrix.col_ptrs[0] != 0) {
    return false;
  }

  const int total_elements = static_cast<int>(matrix.values.size());
  if (matrix.col_ptrs[matrix.cols] != total_elements) {
    return false;
  }

  for (size_t i = 0; i < matrix.col_ptrs.size() - 1; ++i) {
    if (matrix.col_ptrs[i] > matrix.col_ptrs[i + 1] || matrix.col_ptrs[i] < 0) {
      return false;
    }
  }

  for (size_t i = 0; i < matrix.row_indices.size(); ++i) {
    if (matrix.row_indices[i] < 0 || matrix.row_indices[i] >= matrix.rows) {
      return false;
    }
  }

  return true;
}

bool KotelnikovaATaskOMP::ValidationImpl() {
  const auto &[a, b] = GetInput();

  if (!IsMatrixValid(a) || !IsMatrixValid(b)) {
    return false;
  }
  if (a.cols != b.rows) {
    return false;
  }

  return true;
}

bool KotelnikovaATaskOMP::PreProcessingImpl() {
  const auto &[a, b] = GetInput();
  GetOutput() = SparseMatrixCCS(a.rows, b.cols);
  return true;
}

SparseMatrixCCS KotelnikovaATaskOMP::MultiplyMatrices(const SparseMatrixCCS &a, const SparseMatrixCCS &b) {
  SparseMatrixCCS result(a.rows, b.cols);

  const double epsilon = 1e-10;

  std::vector<int> col_start(b.cols + 1, 0);
  std::vector<int> col_end(b.cols + 1, 0);

#pragma omp parallel for schedule(dynamic, 8)
  for (int j = 0; j < b.cols; ++j) {
    std::vector<double> temp(a.rows, 0.0);

    for (int b_idx = b.col_ptrs[j]; b_idx < b.col_ptrs[j + 1]; ++b_idx) {
      const int k = b.row_indices[b_idx];
      const double b_val = b.values[b_idx];

      for (int a_idx = a.col_ptrs[k]; a_idx < a.col_ptrs[k + 1]; ++a_idx) {
        const int i = a.row_indices[a_idx];
        temp[i] += a.values[a_idx] * b_val;
      }
    }

    int nnz_in_col = 0;
    for (int i = 0; i < a.rows; ++i) {
      if (std::abs(temp[i]) > epsilon) {
        nnz_in_col++;
      }
    }

    col_start[j] = nnz_in_col;
  }

  std::vector<int> col_ptr(b.cols + 1, 0);
  for (int j = 0; j < b.cols; ++j) {
    col_ptr[j + 1] = col_ptr[j] + col_start[j];
  }

  int total_nnz = col_ptr[b.cols];
  result.values.resize(total_nnz);
  result.row_indices.resize(total_nnz);
  result.col_ptrs = col_ptr;

#pragma omp parallel for schedule(dynamic, 8)
  for (int j = 0; j < b.cols; ++j) {
    std::vector<double> temp(a.rows, 0.0);

    for (int b_idx = b.col_ptrs[j]; b_idx < b.col_ptrs[j + 1]; ++b_idx) {
      const int k = b.row_indices[b_idx];
      const double b_val = b.values[b_idx];

      for (int a_idx = a.col_ptrs[k]; a_idx < a.col_ptrs[k + 1]; ++a_idx) {
        const int i = a.row_indices[a_idx];
        temp[i] += a.values[a_idx] * b_val;
      }
    }

    int pos = col_ptr[j];
    for (int i = 0; i < a.rows; ++i) {
      if (std::abs(temp[i]) > epsilon) {
        result.row_indices[pos] = i;
        result.values[pos] = temp[i];
        pos++;
      }
    }
  }

  return result;
}

bool KotelnikovaATaskOMP::RunImpl() {
  const auto &[a, b] = GetInput();
  GetOutput() = MultiplyMatrices(a, b);
  return true;
}

bool KotelnikovaATaskOMP::PostProcessingImpl() {
  return true;
}

}  // namespace kotelnikova_a_double_matr_mult_omp
