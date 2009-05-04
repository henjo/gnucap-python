%module(directors="1") gnucap

%include stl.i
%include std_string.i

%{
#include "gnucap.h"
#include "gnucap/ap.h"
#include "gnucap/c_comand.h"
#include "gnucap/l_dispatcher.h"
#include "gnucap/s__.h"
#include "gnucap/m_wave.h"
#include "gnucap/u_opt.h"
%}

#ifdef HAS_NUMPY
%{
#include "numpy_interface.h"
%}
#endif

%exception {
    try {
        $action
    } catch (Exception& e) {
      PyErr_SetString(PyExc_Exception, e.message().c_str());
      return NULL;
    }
}
%allowexception;

///////////////////////////////////////////////////////////////////////////////
// Major gnucap classes
///////////////////////////////////////////////////////////////////////////////

class CS {
public:
      enum STRING {_STRING};
      CS(CS::STRING, const std::string& s);
      const std::string fullstring()const;
};

%feature("director") CMD;
class CMD { 
public:            
      CMD();
      virtual void do_it(CS& cmd, CARD_LIST*)=0;
};

%feature("director") SIM;
class SIM : public CMD {
public:
        static WAVE* find_wave(const std::string&);
      
private:
        const std::string long_label()const {unreachable(); return "";}
private:
        virtual void  setup(CS&)      = 0;
        virtual void  sweep()         = 0;
        virtual void  finish()        {}
        virtual bool  is_step_rejected()const {return false;}

};

enum RUN_MODE {
  rPRE_MAIN,    /* it hasn't got to main yet                    */
  rPRESET,      /* do set up commands now, but not simulation   */
                /* store parameters, so bare invocation of a    */
                /* simulation command will do it this way.      */
  rINTERACTIVE, /* run the commands, interactively              */
  rSCRIPT,      /* execute now, as a command, then restore mode */
  rBATCH        /* execute now, as a command, then exit         */
};

class SET_RUN_MODE {
public:
      SET_RUN_MODE(RUN_MODE rm);
};

///////////////////////////////////////////////////////////////////////////////
// gnucap functions
///////////////////////////////////////////////////////////////////////////////
void command(char *command);
DISPATCHER<CMD>::INSTALL *attach_command(char *command_name, CMD *cmd);
void detach_command(DISPATCHER<CMD>::INSTALL *installer);

///////////////////////////////////////////////////////////////////////////////
// non-gnucap utility functions
///////////////////////////////////////////////////////////////////////////////
#ifdef HAS_NUMPY
void init_numpy();
PyObject *wave_to_arrays(WAVE *wave);
#endif

///////////////////////////////////////////////////////////////////////////////
// init
///////////////////////////////////////////////////////////////////////////////
#ifdef HAS_NUMPY
%init %{
      init_numpy();
%}
#endif
