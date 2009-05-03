import os
import numpy
import pylab

import gnucap

gnucap.command("set lang=acs")

## Set gnucap run mode
runmode = gnucap.SET_RUN_MODE(gnucap.rBATCH)

## Define a new plotting command
class Plot(gnucap.CMD):
    def do_it(self, cmd, b):
        args = cmd.fullstring().split(" ")
        w=gnucap.SIM.find_wave(args[1])
        x,y = gnucap.wave_to_arrays(w)
        pylab.plot(x,y)
        pylab.show()

## Attach it
plot = Plot()
gnucap.attach_command("myplot", plot)

gnucap.command("get eq2-145.ckt")
gnucap.command("store ac vm(2)")
gnucap.command("ac oct 10 1k 100k")

## Now use the new command to plot vm(2)
gnucap.command("myplot vm(2)")

