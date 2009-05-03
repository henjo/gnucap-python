#include "gnucap/u_lang.h"
#include "gnucap/c_comand.h"
#include "gnucap/globals.h"
#include "gnucap/s__.h"

#include <Python.h>

// Swig _gnucap init function prototype
extern "C" void init_gnucap();

/*--------------------------------------------------------------------------*/
namespace {
  static int python_loaded = 0;

/*--------------------------------------------------------------------------*/
void eval_python(CS& cmd, OMSTREAM out, CARD_LIST* scope)
{
  std::string file_name;
  char *argv[] = {};
  FILE *fp;

  cmd >> file_name;
  
  fp = fopen(file_name.c_str(), "r");
  
  if(fp == NULL) 
    throw Exception_File_Open(std::string("Could not open ") + file_name);
  
  if(!python_loaded) {
    dlopen(PYTHON_SO, RTLD_NOW|RTLD_GLOBAL);
    Py_Initialize();
    PySys_SetArgv(0, argv);
    
    // Call init function of SWIG _gnucap module
    init_gnucap();

    python_loaded = 1;
  }

  PyRun_SimpleFile(fp, file_name.c_str());
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
