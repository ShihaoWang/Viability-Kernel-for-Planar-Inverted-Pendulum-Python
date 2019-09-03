import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display,Markdown

import numpy as np
import math
from scipy.integrate import solve_ivp
from hjb import *

from math import *
import time
import struct

coeff = 2.25

LLow = 0.35;                  LUpp = 1.05;
LdotLow = -1.0;               LdotUpp = 1.0;
ThetaLow = -math.pi/6.0;      ThetaUpp = math.pi/2.0;
ThetadotLow = -1.5;           ThetadotUpp = -1.0 * ThetadotLow;

L_Grids = 71
Ldot_Grids = 21
Theta_Grids = 211
Thetadot_Grids = 31

L_List = np.linspace(LLow, LUpp, num=L_Grids)
Ldot_List = np.linspace(LdotLow, LdotUpp, num=Ldot_Grids)
Theta_List = np.linspace(ThetaLow, ThetaUpp, num=Theta_Grids)
Thetadot_List = np.linspace(ThetadotLow, ThetadotUpp, num=Thetadot_Grids)

L_unit = (LUpp - LLow)/(1.0 * L_Grids - 1.0)
Ldot_unit = (LdotUpp - LdotLow)/(1.0 * Ldot_Grids - 1.0)
Theta_unit = (ThetaUpp - ThetaLow)/(1.0 * Theta_Grids - 1.0)
Thetadot_unit = (ThetadotUpp - ThetadotLow)/(1.0 * Thetadot_Grids - 1.0)

bounds = [(LLow, LUpp), (LdotLow, LdotUpp), (ThetaLow, ThetaUpp), (ThetadotLow, ThetadotUpp)]

dt = 0.1   # Simulation time step length
INTEGRATE = False

class PIPDynamics(DynamicSystem):
    """
        The state space is (L, Ldot, theta, thetadot) and the control is the external force
    """
    def __init__(self, g=9.81):
        self.g = g
    def dynamics(self, state, control):
        """
            This function calculates PIP's system dynamics according to the current state and control
        """
        L = state[0]
        Ldot= state[1]
        theta = state[2]
        thetadot = state[3]
        u = control                    # Here u is already the effective control stands for L's acceleration.

        EoM = [0]*4
        EoM[0] = Ldot
        EoM[1] = u
        EoM[2] = thetadot
        EoM[3] = self.g/L * sin(theta) - 2.0 * thetadot * Ldot/L

        return np.array(EoM)
    def nextState(self,state,control,dt):
        if dt < 0:
            #allow reverse dynamics
            if INTEGRATE:
                res = solve_ivp((lambda t,y:-self.dynamics(y,control)),[0,-dt],state,rtol=INTEGRATION_REL_TOLERANCE,atol=INTEGRATION_ABS_TOLERANCE)
                x = res.y[:,-1]
            else:
                x = state + dt*self.dynamics(state,control)
        else:
            if INTEGRATE:
                res = solve_ivp((lambda t,y:self.dynamics(y,control)),[0,dt],state,rtol=INTEGRATION_REL_TOLERANCE,atol=INTEGRATION_ABS_TOLERANCE)
                x = res.y[:,-1]
            else:
                x = state + dt*self.dynamics(state,control)
        #normalize the angle for this model it is the third angle.
        x[2] = x[2]%(2.0*math.pi)
        if x[0] < 0:
            x[0] += (2.0*math.pi)
        return x

    def validState(self,state):
        return True

    def validControl(self,state,control):
        return True

class PendulumControlSampler(ControlSampler):
    # This part gives the control to the pip model.
    # The output from this function should be the robot's external contact force.
    def __init__(self, lmin, lmax, g, dt):
        self.lmin = lmin    # Lower length bound
        self.lmax = lmax    # Higher length bound
        self.g = g;
        self.dt = dt
    def sample(self,state):
        # Based on the physical meaning of the contact force, it should always be supportive so the minimal value is 0
        # As for the maximum value, according to the observation of robot jump behavior, a linear maximum force is assumed.
        L = state[0]
        Ldot = state[1]
        theta = state[2]
        thetadot = state[3]

        # The control will be directly assigned to the pendulum length state variable.
        u_min = -self.g * cos(theta)
        u_max = coeff * self.g * cos(theta) - coeff * self.g * cos(theta)/(self.lmax - self.lmin) * (L-self.lmin)
        if u_min>0:
            u_min = 0.0
        if u_max<0:
            u_max = 0.0
        return [u_min, 0, u_max]

class TimeObjectiveFunction(ObjectiveFunction):
    def edgeCost(self,state,control,dt,nextState):
        return abs(dt)

class EffortObjectiveFunction(ObjectiveFunction):
    def edgeCost(self,state,control,dt,nextState):
        return np.linalg.norm(control)**2*dt

def GoalFunc(x):
    # This function is used to categorize whether the given state is a goal state or not
    L_k = x[0]
    Ldot_k = x[1]
    theta_k = x[2]
    thetadot_k = x[3]
    # As long as the pendulum is on the positive side and it is angular velocity is positive.
    Res = True
#     if(theta_k>0) and (thetadot_k>=0) and (Ldot_k>=0):
    if(theta_k>0) and (thetadot_k>=0):
        Res = True
    else:
        Res = False
    return Res

def do_value_iteration(i, hjb):
    print "Running",i,"HJB iterations"
    hjb.valueIteration(iters=i)
    # hjbdisplay.refresh(hjb.value,hjb.policy)
    # if hjb.getPolicy(start) is not None:
    #     #show the HJB policy
    #     xs,us = rolloutPolicy(dynamics,start,(lambda x:hjb.getPolicy(x)),dt*0.5,200)
    #     hjbdisplay.plotTrajectory(xs,color='r',zorder=3)
    #
    #     la_policy = LookaheadPolicy(dynamics,dt,controlSampler,objective,(lambda x:False),hjb.interpolateValue)
    #     xs,us = rolloutPolicy(dynamics,start,la_policy,dt,200)
    #     hjbdisplay.plotTrajectory(xs,color='y',zorder=4)
    # hjbdisplay.plotFlow(lambda x:hjb.getPolicy(x))

def hjbvaluewriter(hjb, Angle):
    PointNumber = L_Grids * Ldot_Grids * Theta_Grids * Thetadot_Grids
    ObjList = [0] * PointNumber
    CurrentIndex = 0
    for i in range(0, L_Grids):
        for j in range(0, Ldot_Grids):
            for k in range(0, Theta_Grids):
                for l in range(0, Thetadot_Grids):
                    if math.isinf(hjb.value[i,j,k,l]) is True:
                        ObjList[CurrentIndex] = -1.0
                    else:
                        ObjList[CurrentIndex] = hjb.value[i,j,k,l]
                    CurrentIndex+=1
    s = struct.pack('f'*len(ObjList), *ObjList)
    fName = "PVKFailureMetric" + str(round(Angle)) + ".bin"
    f = open(fName,'wb')
    f.write(s)
    f.close()

def main():
    # This is the main computation code for HJB Viability Kernel
    ValueIterationNumber = 10
    # The default idea is to sample gravity from 0 degree to 80 degree
    AngleList = np.linspace(0, 80, num=9)
    for i in range(0, 9):
        start_time = time.time()
        Angle_i = AngleList[i]
        print "Computation for Angle: " + str(Angle_i)
        g = 9.81 * cos(Angle_i/180.0 * 3.1415926535897);
        dynamics = PIPDynamics(g)
        controlSampler = PendulumControlSampler(LLow, LUpp, g, dt)
        objective = TimeObjectiveFunction()
        hjb = HJBSolver(dynamics, controlSampler, dt, objective, bounds, [L_Grids, Ldot_Grids, Theta_Grids, Thetadot_Grids],  GoalFunc)
        
        do_value_iteration(ValueIterationNumber, hjb)
        hjbvaluewriter(hjb, Angle_i)

        elapsed_time = time.time() - start_time
        print "Elapsed Time: " + str(elapsed_time) + " s"

if __name__== "__main__":
    main()
