#include <gnucap/m_matrix.h>
#include <gnucap/m_wave.h>
#include "numpy_interface.h"
#include <Python.h>
#include <numpy/arrayobject.h>

void init_numpy() {
  import_array();
}

PyObject *to_double_array(double *data, int len) {
  npy_intp dims[1] = {len};
  return PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (void *)(data+1));
}

PyObject *get_complex_array(COMPLEX *data, int len) {
  npy_intp dims[1] = {len};
  PyObject *arr = PyArray_SimpleNew(1, dims, PyArray_COMPLEX128);

  npy_complex128 *outdata = (npy_complex128 *) PyArray_DATA(arr);
  
  // copy data
  for(int i=0; i < len; i++) {
    outdata[i].real = data[i+1].real();
    outdata[i].imag = data[i+1].imag();
  }
  return arr;
}

void set_complex_array(COMPLEX *data, PyObject *srcarray) {
  if( !PyArray_Check(srcarray) ||
       ( PyArray_TYPE(srcarray ) != PyArray_COMPLEX128 ) ||
      ( PyArray_NDIM(srcarray ) != 1 ))
    throw Exception("Source array has incorrect type");
 
  npy_complex128 *indata = (npy_complex128 *) PyArray_DATA(srcarray);
  for(int i=0; i < PyArray_DIM(srcarray, 0); i++)
    data[i] = COMPLEX(indata[i].real, indata[i].imag);
}

PyObject *wave_to_arrays(WAVE *wave) {
  PyObject *x, *y;
  int i, nrows = 0;

  if(wave == NULL) 
    throw Exception("wave_to_arrays got NULL instead of WAVE pointer");

  // Count number of elements
  // FIXME, WAVE should provide a method for this
  for(WAVE::const_iterator wi = wave->begin(); wi < wave->end(); wi++) 
    nrows++;

  // Create arrays
  npy_intp dims[1] = {nrows};
  x = PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
  y = PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
  
  // Copy data
  double *xdata = (double *) PyArray_DATA(x);
  double *ydata = (double *) PyArray_DATA(y);
  i=0;
  for(WAVE::const_iterator wi = wave->begin(); wi < wave->end(); wi++, i++) {
    xdata[i] = wi->first;
    ydata[i] = wi->second;
  }

  // Return tuple of x,y
  PyObject *pTuple = PyTuple_New(2); // new reference
  PyTuple_SetItem(pTuple, 0, x);
  PyTuple_SetItem(pTuple, 1, y);

  return pTuple;
}

PyObject *bsmatrix_to_array_d(BSMATRIX<double> &A) {
  PyObject *Aarray;
  int i,j;

  // Create arrays
  npy_intp dims[2] = {A.size(), A.size()};
  Aarray = PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
  
  // Copy data
  double *ptr = (double *) PyArray_DATA(Aarray);
  for(i = 1; i <= A.size(); i++) 
    for(j = 1; j <= A.size(); j++)
      *ptr++ = A.s(i,j);

  return Aarray;
}

PyObject *bsmatrix_to_array_c(BSMATRIX<COMPLEX> &A) {
  PyObject *Aarray;
  int i,j;

  // Create arrays
  npy_intp dims[2] = {A.size(), A.size()};
  Aarray = PyArray_SimpleNew(2, dims, PyArray_COMPLEX128);
  
  // Copy data
  npy_complex128 *ptr = (npy_complex128 *) PyArray_DATA(Aarray);
  for(i = 1; i <= A.size(); i++) 
    for(j = 1; j <= A.size(); j++) {
      ptr->real = A.s(i,j).real();
      ptr->imag = A.s(i,j).imag();
      ptr++;
    }
  return Aarray;
}

