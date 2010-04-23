This package contains a python plugin for the circuit simulator Gnucap which\    allows the user to write custom commands in Python. 
It also provides a gnucap extension for Python where gnucap can be used 
as a library. No gnucap binary is needed. 

There is also waveform numpy array
conversion functions which means that post-processing of gnucap simulation
results can be done in Python using numpy/scipy/matplotlib.

 Requirements
-------------

You have to installed the modified version of gnucap that is compiled as a
shared library. You can download it from the git repo at 
http://github.com/henjo/gnucap.

Other requirements are:
  * Python >= 2.4
  * Swig
  * Numpy (with development headers/libraries)

 Installation
-------------

Build python plugin for gnucap

.. code-block:: sh
   $ ./autogen.sh
   $ make install

The resulting plugin will be written to "gnucap-plugins/python.so".

Build gnucap extension

.. code-block:: sh
   $ python setup.py build
   $ sudo python setup.py install

Examples
--------

From gnucap
~~~~~~~~~~~

.. code-block:: sh
   $ gnucap
   gnucap> attach gnucap-plugins/python.so
   gnucap> python example/loadplot.py
   gnucap> get example/eq2-145.ckt
   gnucap> store ac vm(2)
   gnucap> ac oct 10 1k 100k
   gnucap> myplot vm(2)

First the gnucap plugin is loaded. The second line loads a new command called 
"myplot" that plots a stored waveform using matplotlib. Line 3-5 loads a 
circuit and runs an ac analysis. Finally the ac magnitude of node 2 is plotted
using the new plotting command.

From Python
~~~~~~~~~~~

Do the same directly from Python

.. code-block:: sh
   $ cd examples
   $ python simple.py
