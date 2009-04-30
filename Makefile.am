## $Id$
##
## Top level automake file for gnucap

pluginsdir = `pwd`/gnucap-plugins

plugins_LTLIBRARIES = python.la

python_la_SOURCES = c_python.cc
python_la_CPPFLAGS = @PYTHON_CPPFLAGS@
python_la_CFLAGS = -fPIC
python_la_LDFLAGS = -shared -module -avoid-version @LDFLAGS@ @PYTHON_LDFLAGS@
python_la_LIBS = @PYTHON_EXTRA_LIBS@

ACLOCAL_AMFLAGS = -I m4

