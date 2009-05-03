#include "gnucap/c_comand.h"
#include "gnucap/l_dispatcher.h"

void command(char *command);
DISPATCHER<CMD>::INSTALL *attach_command(char *command_name, CMD *cmd);
void detach_command(DISPATCHER<CMD>::INSTALL *installer);

