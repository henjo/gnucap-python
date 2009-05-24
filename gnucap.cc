#include "gnucap.h"
#include "gnucap/u_lang.h"
#include "gnucap/c_comand.h"
#include "gnucap/globals.h"
#include "gnucap/m_wave.h"
#include "gnucap/s__.h"
#include "gnucap/io_.h"

#include "numpy_interface.h"

#include <stdio.h>
#include <string>
#include <fstream>

std::string command(char *command) {
  
  char filename[L_tmpnam];
  
  tmpnam(filename);
  
  // supress output to stdout
  IO::mstdout.detach(stdout);

  // send output to file
  CMD::command(std::string("> ") + std::string(filename), &CARD_LIST::card_list);

  CMD::command(std::string(command), &CARD_LIST::card_list);

  CMD::command(">", &CARD_LIST::card_list);

  // Open file an read it
  std::ifstream ifs(filename);

  std::ostringstream oss;

  oss << ifs.rdbuf();

  std::string output(oss.str());

  unlink(filename);
  
  return output;
}

DISPATCHER<CMD>::INSTALL *attach_command(char *command_name, CMD *cmd) {
  return new DISPATCHER<CMD>::INSTALL(&command_dispatcher, command_name, cmd);
}

void detach_command(DISPATCHER<CMD>::INSTALL *installer) {
  delete installer;
}
