#include "gnucap/c_comand.h"
#include "gnucap/l_dispatcher.h"
#include "gnucap/s__.h"

class SIMWrapper : public SIM {
public:
  explicit SIMWrapper():SIM()  {}
  virtual void  setup(CS&)=0;
  virtual void  sweep()=0;
};

std::string command(char *command);
DISPATCHER<CMD>::INSTALL *attach_command(char *command_name, CMD *cmd);
void detach_command(DISPATCHER<CMD>::INSTALL *installer);

