#include "gnucap/m_wave.h"
#include "numpy_interface.h"
#include <Python.h>
#include <numpy/arrayobject.h>

void init_numpy() {
  import_array();
}

PyObject *wave_to_arrays(WAVE *wave) {
  PyObject *x, *y;
  int i, nrows = 0;

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

