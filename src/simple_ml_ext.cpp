#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <cstdlib>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): posize_ter to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): posize_ter to y data, of size m
     *     theta (float *): posize_ter to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (size_t): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t epochs = m/batch+((m-batch*m/batch>0)?1:0);
    for (size_t i=0;i<epochs;i++){
        size_t x_start=i*n;
        size_t num_example=((i+1)*batch>m?m:((i+1)*batch))-i*batch;

        float* data = (float*)malloc(num_example * k * sizeof(float));

        float* data_sum = (float*)malloc(num_example * 1 * sizeof(float));


        size_t* Iy = (size_t*)malloc(num_example * k * sizeof(size_t));//y:m, Iy:m*k 

        for (size_t i = 0; i < num_example * k; ++i) {
            Iy[i] = 0;
        }

        for(size_t m_index=0;m_index<num_example;m_index++){
            Iy[m_index*y[m_index]+y[m_index]]=1;
        }

        for(size_t m_index=0;m_index<num_example;m_index++){
            data_sum[m_index]=0;
            for(size_t k_index=0;k_index<k;k_index++){
                data[m_index*k+k_index]=0;
                for(size_t n_index=0;n_index<n;n_index++){
                    data[m_index*k+k_index]+=X[x_start+m_index*n+n_index]*theta[n_index*k+k_index];
                }
                data[m_index*k+k_index]=std::exp(data[m_index*k+k_index]);//分子
                data_sum[m_index]+=data[m_index*k+k_index];//分母
            }
        }

        float* Z = (float*)malloc(num_example * k * sizeof(float));

        for (size_t i = 0; i < num_example * 1; ++i) {
            Z[i] = 0;
        }

        for(size_t m_index=0;m_index<num_example;m_index++){
      k_indexsize_t      for(size_t k_index=0;k_index<k;k_index++){
                Z[m_index*k+k_index]=data[m_index*k+k_index]/data_sum[m_index];
            }
        }

        float* dt = (float*)malloc(n * k * sizeof(float));

        for(size_t n_index=0;n_index<n;n_index++){
            for(size_t k_index=0;k_index<k;k_index++){
                dt[n_index*k+k_index]=0;
                for(size_t m_index=0;m_index<num_example;m_index++){
                    dt[n_index*k+k_index]+=X[x_start+m_index*n+n_index]*(Z[m_index*k+k_index]-Iy[m_index*k+k_index])/m;
                }
                theta[n_index*k+k_index]=theta[n_index*k+k_index]-lr*dt[n_index*k+k_index];
            }
        }
    };
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           size_t batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
