// redkina_a_integral_simpson_seq/omp/src/ops_omp.cpp
#include "redkina_a_integral_simpson_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "redkina_a_integral_simpson_seq/common/include/common.hpp"

namespace redkina_a_integral_simpson_seq {

namespace {

// Вспомогательные функции, используемые в вычислениях (аналогично seq версии)
void EvaluatePoint(const std::vector<double> &a, const std::vector<double> &h, const std::vector<int> &n,
                   const std::vector<int> &indices, const std::function<double(const std::vector<double> &)> &func,
                   std::vector<double> &point, double &sum) {
  size_t dim = a.size();
  double w_prod = 1.0;
  for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
    int idx = indices[dim_idx];
    point[dim_idx] = a[dim_idx] + (static_cast<double>(idx) * h[dim_idx]);

    int w = 0;
    if (idx == 0 || idx == n[dim_idx]) {
      w = 1;
    } else if (idx % 2 == 1) {
      w = 4;
    } else {
      w = 2;
    }
    w_prod *= static_cast<double>(w);
  }
  sum += w_prod * func(point);
}

bool AdvanceIndices(std::vector<int> &indices, const std::vector<int> &n) {
  int dim = static_cast<int>(indices.size());
  int d = dim - 1;
  while (d >= 0 && indices[d] == n[d]) {
    indices[d] = 0;
    --d;
  }
  if (d < 0) {
    return false;
  }
  ++indices[d];
  return true;
}

}  // namespace

// Статическая функция преобразования линейного индекса в многомерный
std::vector<int> RedkinaAIntegralSimpsonOMP::LinearToIndices(size_t lin, const std::vector<int> &n) {
  size_t dim = n.size();
  std::vector<int> indices(dim);
  size_t temp = lin;
  for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
    int base = n[d] + 1;
    indices[d] = static_cast<int>(temp % base);
    temp /= base;
  }
  return indices;
}

RedkinaAIntegralSimpsonOMP::RedkinaAIntegralSimpsonOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RedkinaAIntegralSimpsonOMP::ValidationImpl() {
  const auto &in = GetInput();
  size_t dim = in.a.size();

  if (dim == 0 || in.b.size() != dim || in.n.size() != dim) {
    return false;
  }

  for (size_t i = 0; i < dim; ++i) {
    if (in.a[i] >= in.b[i]) {
      return false;
    }
    if (in.n[i] <= 0 || in.n[i] % 2 != 0) {
      return false;
    }
  }

  return static_cast<bool>(in.func);
}

bool RedkinaAIntegralSimpsonOMP::PreProcessingImpl() {
  const auto &in = GetInput();
  func_ = in.func;
  a_ = in.a;
  b_ = in.b;
  n_ = in.n;
  result_ = 0.0;
  return true;
}

bool RedkinaAIntegralSimpsonOMP::RunImpl() {
  size_t dim = a_.size();

  // Шаги интегрирования по каждому направлению
  std::vector<double> h(dim);
  for (size_t i = 0; i < dim; ++i) {
    h[i] = (b_[i] - a_[i]) / static_cast<double>(n_[i]);
  }

  // Произведение шагов
  double h_prod = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    h_prod *= h[i];
  }

  // Общее количество узлов сетки (комбинаций индексов)
  size_t total_combinations = 1;
  for (int ni : n_) {
    total_combinations *= static_cast<size_t>(ni + 1);
  }

  double global_sum = 0.0;

#pragma omp parallel
  {
    int thread_num = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

    // Распределение итераций (блоков) между потоками
    size_t chunk_size = (total_combinations + static_cast<size_t>(num_threads) - 1) / static_cast<size_t>(num_threads);
    size_t start = static_cast<size_t>(thread_num) * chunk_size;
    size_t end = std::min(start + chunk_size, total_combinations);

    // Только потоки, которым досталась работа
    if (start < total_combinations) {
      // Локальные для потока данные
      std::vector<int> local_indices = LinearToIndices(start, n_);
      std::vector<double> local_point(dim);
      double local_sum = 0.0;

      // Перебор всех комбинаций в выделенном диапазоне
      for (size_t lin = start; lin < end; ++lin) {
        EvaluatePoint(a_, h, n_, local_indices, func_, local_point, local_sum);
        // Переход к следующей комбинации (кроме последней итерации)
        if (lin + 1 < end) {
          AdvanceIndices(local_indices, n_);
        }
      }

      // Атомарное добавление локальной суммы в глобальную
#pragma omp atomic
      global_sum += local_sum;
    }
  }

  // Знаменатель формулы Симпсона: 3^dim
  double denominator = std::pow(3.0, static_cast<double>(dim));
  result_ = (h_prod / denominator) * global_sum;

  return true;
}

bool RedkinaAIntegralSimpsonOMP::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace redkina_a_integral_simpson_seq
