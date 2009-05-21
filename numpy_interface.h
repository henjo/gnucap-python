#include <gnucap/m_matrix.h>
#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *to_double_array(double *data, int len);
PyObject *get_complex_array(COMPLEX *data, int len);
void set_complex_array(COMPLEX *data, PyObject *srcarray);

PyObject *wave_to_arrays(WAVE *wave);
PyObject *bsmatrix_to_array_d(BSMATRIX<double> &A);
PyObject *bsmatrix_to_array_c(BSMATRIX<COMPLEX> &A);
void init_numpy();

// Function for performing fb substitution of a BSMATRIX with
// a numpy array as the rhs vector or matrix
template <class T>
void bsmatrix_fbsub_array(BSMATRIX<T> *A, PyObject *rhs, PyObject *dest) {
/*   if( !PyArray_Check(rhs) || */
/*       ( PyArray_NDIM(rhs ) > 2) || */
/*       ( PyArray_NDIM(rhs ) < 1) */
/*       ) */
/*     throw Exception("Dest array has incorrect type"); */
/*   if( !PyArray_Check(dest) || */
/*       ( PyArray_NDIM(dest) != PyArray_NDIM(rhs))) */
/*     throw Exception("Dest array has incorrect type"); */

  int rows = PyArray_DIM(rhs, 0);

  // RHS and solution vector
  T v[rows+1];

  if(PyArray_NDIM(rhs) == 1) {
    // RHS is a vector

    // Copy rhs to v
    for(int i=0; i < rows; i++)
      v[i+1] = *(T *)PyArray_GETPTR1(rhs, i);

    A->fbsub(v);

    // Copy result to dest array
    for(int i=0; i < rows; i++)
      *(T *)PyArray_GETPTR1(dest, i) = v[i+1];
  } else {
    // RHS is a matrix
    int cols = PyArray_DIM(rhs, 1);

    for(int col=0; col < cols; col++) {
      // Copy rhs to v
      for(int i=0; i < rows; i++)
        v[i+1] = *(T *)PyArray_GETPTR2(rhs, i, col);

      A->fbsub(v);
      
      // Copy result to dest array
      for(int i=0; i < rows; i++)
        *(T *)PyArray_GETPTR2(dest, i, col) = v[i+1];
    }
  }
}



