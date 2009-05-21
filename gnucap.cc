#include "gnucap.h"
#include "gnucap/u_lang.h"
#include "gnucap/c_comand.h"
#include "gnucap/globals.h"
#include "gnucap/m_wave.h"
#include "gnucap/s__.h"

#include "numpy_interface.h"

#include <string>

void command(char *command) {
  CMD::command(std::string(command), &CARD_LIST::card_list);
}

DISPATCHER<CMD>::INSTALL *attach_command(char *command_name, CMD *cmd) {
  return new DISPATCHER<CMD>::INSTALL(&command_dispatcher, command_name, cmd);
}

void detach_command(DISPATCHER<CMD>::INSTALL *installer) {
  delete installer;
}
