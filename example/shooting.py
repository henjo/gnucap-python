"""
PSS analysis experiments

Currently the TRANSIENT analysis is subclassed and the accept method is
overloaded where the shooting newton iteration jacobian is formed using
the C matrix and the transient jacobian 

"""

import os
import numpy as np
import pylab

import gnucap

gnucap.command("set lang=acs")

## Set gnucap run mode
runmode = gnucap.SET_RUN_MODE(gnucap.rBATCH)

gnucap.command("get shooting.ckt")

class MyTransient(gnucap.TRANSIENT):
    def do_it(self, cmd, scope):
        n = gnucap.cvar.status.total_nodes
        self.Jshoot = np.eye(n)
        self.lastC = None

        ## Prepare AC analysis
        card_list = gnucap.cvar.CARD_LIST_card_list
        gnucap.cvar.SIM_jomega = 1j;
        acx =  gnucap.cvar.CKT_BASE_acx ## Static attributes must be accessed through cvar
        acx.reallocate()
        card_list.ac_begin()
        
        self.first = True

        gnucap.TRANSIENT.do_it(self, cmd, scope)

    def accept(self):
        gnucap.TRANSIENT.accept(self)
        
        t = gnucap.cvar.SIM_time0
        if True:
            print "Accept at ", gnucap.cvar.SIM_time0
            n = gnucap.cvar.status.total_nodes
            aa = gnucap.bsmatrix_to_array_d(gnucap.cvar.CKT_BASE_aa)
            i = gnucap.to_double_array(gnucap.cvar.SIM_i, n)
            v0 = gnucap.to_double_array(gnucap.cvar.SIM_v0, n)
            print v0
            if self.first:
                self.v0_0 = v0.copy()
                self.first = False
            else:
                self.v0_n = v0.copy()

            ## Solve system ourself and compare with v0
            if self.lastC != None:
                myv0 = np.zeros(aa.shape)
                h = gnucap.cvar.SIM_time0 - gnucap.cvar.SIM_time1

                gnucap.bsmatrix_fbsub_array_double(gnucap.cvar.CKT_BASE_lu, 
                                                   np.dot(self.Jshoot, self.lastC) / h, 
                                                   self.Jshoot)

            ## Get C matrix from ac-analysis
            ## FIXME, there must be a better way
            acx =  gnucap.cvar.CKT_BASE_acx ## Static attributes must be accessed through cvar
            acx.zero()
            card_list = gnucap.cvar.CARD_LIST_card_list
    #        card_list.do_ac()
#            card_list.ac_load()
            C_bs = gnucap.bsmatrix_to_array_c(acx)
            self.lastC = np.imag(C_bs)
            self.lastC = np.array([[1e-6, 0],
                                    [0, 0]])


        
mytran = MyTransient()

gnucap.attach_command("mytran", mytran)

gnucap.command("store tran v(2)")

t0 = 0
alpha = 1

gnucap.command("options method euler")
gnucap.command("mytran 1e-4 1e-3 0")

while True:
    n = gnucap.cvar.status.total_nodes

    residual = mytran.v0_n - mytran.v0_0

    print "residual: ", np.sqrt(np.dot(residual, residual))

    Jshoot = (np.eye(n) - alpha * mytran.Jshoot)
    print mytran.Jshoot

    newx = mytran.v0_0 + np.linalg.solve(Jshoot, residual)
    print "newton, last", newx, mytran.v0_n

    gnucap.to_double_array(gnucap.cvar.SIM_vdc, n)[:] = newx[:]

    print gnucap.to_double_array(gnucap.cvar.SIM_vdc, n)

#    gnucap.cvar.SIM_last_time = t0

    gnucap.command("mytran")

w= gnucap.SIM.find_wave("v(2)")
x,y = gnucap.wave_to_arrays(w)
for x,y in  zip(x,y):
    print x,y
