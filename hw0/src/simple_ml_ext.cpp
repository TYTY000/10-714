#include <cmath>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void matmul(const float *x, const float *y, float *z, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      z[i * k + j] = 0.f;
      for (int h = 0; h < n; h++) {
        z[i * k + j] += x[i * n + h] * y[h * k + j];
      }
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
  /**
   * A C++ version of the softmax regression epoch code.  This should run a
   * single epoch over the data defined by X and y (and sizes m,n,k), and
   * modify theta in place.  Your function will probably want to allocate
   * (and then delete) some helper arrays to store the logits and gradients.
   *
   * Args:
   *     X (const float *): pointer to X data, of size m*n, stored in row
   *          major (C) format
   *     y (const unsigned char *): pointer to y data, of size m
   *     theta (float *): pointer to theta data, of size n*k, stored in row
   *          major (C) format
   *     m (size_t): number of examples
   *     n (size_t): input dimension
   *     k (size_t): number of classes
   *     lr (float): learning rate / SGD step size
   *     batch (int): SGD minibatch size
   *
   * Returns:
   *     (None)
   */

  /// BEGIN YOUR CODE
  int iters = (m + batch - 1) / batch;
  for (int iter = 0; iter < iters; iter++) {
    const float *xb = &X[iter * batch * n];
    float *z = new float[batch * k];
    matmul(xb, theta, z, batch, n, k);
    for (size_t i = 0; i < batch * k; i++) {
      z[i] = std::exp(z[i]);
    }
    for (size_t i = 0; i < batch; i++) {
      float sum = 0.f;
      for (size_t j = 0; j < k; j++) {
        sum += z[i * k + j];
      }
      for (size_t j = 0; j < k; j++) {
        z[i * k + j] /= sum;
      }
    }
    for (size_t i = 0; i < batch; i++) {
      z[i * k + y[iter * batch + i]] -= 1;
    }
    float *xt = new float[n * batch];
    float *grad = new float[n * k];
    for (size_t i = 0; i < batch; i++) {
      for (size_t j = 0; j < n; j++) {
        xt[j * batch + i] = xb[i * n + j];
      }
    }
    matmul(xt, z, grad, n, batch, k);
    for (size_t i = 0; i < n * k; i++) {
      theta[i] -= lr * grad[i] / batch;
    }
    delete[] xt;
    delete[] grad;
    delete[] z;
  }
  /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
  m.def(
      "softmax_regression_epoch_cpp",
      [](py::array_t<float, py::array::c_style> X,
         py::array_t<unsigned char, py::array::c_style> y,
         py::array_t<float, py::array::c_style> theta, float lr, int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float *>(X.request().ptr),
            static_cast<const unsigned char *>(y.request().ptr),
            static_cast<float *>(theta.request().ptr), X.request().shape[0],
            X.request().shape[1], theta.request().shape[1], lr, batch);
      },
      py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
      py::arg("batch"));
}
