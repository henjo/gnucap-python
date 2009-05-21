#!/usr/bin/env python

"""
setup.py file for gnucap python extension
"""

from distutils.core import setup, Extension

gnucap_module = Extension('_gnucap',
                          sources=['gnucap.cc', 'numpy_interface.cc',
                                   'gnucap.i'],
                          language = 'c++',
                          libraries = ['gnucap'],
                          swig_opts = ['-c++', '-DHAS_NUMPY -Wall']
                          )

setup (name = 'gnucap',
       version = '0.0',
       author      = "Henrik Johansson",
       description = """Python extension for the Gnucap circuit simulator""",
       ext_modules = [gnucap_module],
       py_modules = ["gnucap"],
       )
