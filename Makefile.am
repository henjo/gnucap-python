## $Id$
##
## Top level automake file for gnucap

pluginsdir = $(libdir)/gnucap

plugins_LTLIBRARIES = python.la

python_la_SOURCES = c_python.cc gnucap.cc gnucap_wrap.cxx
if NUMPY
  python_la_SOURCES += numpy_interface.cc
  SWIG_FLAGS = -DHAS_NUMPY
endif

python_la_CPPFLAGS = @PYTHON_CPPFLAGS@ \
	-DPYTHON_SO=\"@ac_python_libdir@/@ac_python_soname@\"

python_la_CFLAGS = -fPIC 

python_la_LDFLAGS = -shared -module -avoid-version @LDFLAGS@ @PYTHON_LDFLAGS@

python_la_LIBS = @PYTHON_EXTRA_LIBS@

ACLOCAL_AMFLAGS = -I m4

%_wrap.cxx: %.i
	swig -python -c++ -Wall $(SWIG_FLAGS) $<
