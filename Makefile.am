## $Id$
##
## Top level automake file for gnucap

all: c_python.so

ACLOCAL_AMFLAGS = -I m4

CFLAGS = @CFLAGS@ -export-dynamic -shared -fPIC @PYTHON_CPPFLAGS@
LDFLAGS = @LDFLAGS@ @PYTHON_LDFLAGS@
LIBS = @LIBS@ @PYTHON_EXTRA_LIBS@

c_python.so: c_python.cc $(GNUCAP_HEADERS)
	$(CPP)  $< -o $@ $(CFLAGS) $(LIBS) $(LDFLAGS)