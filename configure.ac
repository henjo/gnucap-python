#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# $Id$

AC_PREREQ(2.59)
AC_INIT([c_python.cc])
AM_INIT_AUTOMAKE(gnucap-python, 2009-04-30)
AM_CONFIG_HEADER([config.h])

# Checks for libraries.
# this is a c++ program so use c++ for the tests
AC_LANG([C++])
AC_CHECK_LIB([dl], [dlopen])
AC_CHECK_LIB([dl], [dlopen])

# Check for python
AC_PYTHON_DEVEL([>= '2.3'])

## Libtool init
LT_INIT
AC_CONFIG_MACRO_DIR([m4])

## Check for gnucap
a=1
AC_CHECK_HEADER(gnucap/c_comand.h, [], [a=0])
AC_CHECK_LIB(gnucap, status, [], [a=0])

if test $a == 0
then
   AC_MSG_ERROR([Error, gnucap as shared library not found.])
fi

AC_SUBST(GNUCAP_CFLAGS)
AC_SUBST(GNUCAP_LDFLAGS)

# exports symbols to plugins
LDFLAGS="$LDFLAGS -rdynamic"

AC_OUTPUT([
	Makefile
	])

AC_MSG_RESULT([
** Configuration summary for $PACKAGE $VERSION:

   prefix:                   $prefix
   CPPFLAGS:                 $CPPFLAGS
   CFLAGS:                   $CFLAGS
   CXXFLAGS:                 $CXXFLAGS
   LDFLAGS:                  $LDFLAGS
   LIBS:                     $LIBS

])

