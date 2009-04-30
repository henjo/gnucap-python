#include "gnucap/u_lang.h"
#include "gnucap/c_comand.h"
#include "gnucap/globals.h"

#include <Python.h>

#define PYTHON_SO "/usr/lib/libpython2.6.so"

/*--------------------------------------------------------------------------*/
namespace {
  static int python_loaded = 0;

  static PyObject *command(PyObject *self, PyObject *args) {
    int ok;
    char *command;

    ok = PyArg_ParseTuple(args, "s", &command);

    if(ok) {
      CMD::command(std::string(command), &CARD_LIST::card_list);
    }

    Py_RETURN_NONE;
  }

  static PyMethodDef gnucap_methods[] = {
    {"command", command, METH_VARARGS,
     "Return the number of arguments received by the process."},
    {NULL, NULL, 0, NULL}
  };

/*--------------------------------------------------------------------------*/
void eval_python(CS& cmd, OMSTREAM out, CARD_LIST* scope)
{
  char *argv[] = {};
  if(!python_loaded) {
    dlopen(PYTHON_SO, RTLD_NOW|RTLD_GLOBAL);
    Py_Initialize();
    PySys_SetArgv(0, argv);
    Py_InitModule("gnucap", gnucap_methods);
    python_loaded = 1;
  }
  PyRun_SimpleString("import IPython");
  PyRun_SimpleString("IPython.Shell.IPShell().mainloop(sys_exit=1);");
}

/*--------------------------------------------------------------------------*/
class CMD_PYTHON : public CMD {
public:
  void do_it(CS& cmd, CARD_LIST* Scope)
  {
    eval_python(cmd, IO::mstdout, Scope);
  }
} p1;

DISPATCHER<CMD>::INSTALL d1(&command_dispatcher, "python", &p1);
/*--------------------------------------------------------------------------*/
}
