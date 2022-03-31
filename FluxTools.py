# This is a class to fake neutrino fluxes for neutrino oscillation experiments
# Ryan Nichol <r.nichol@ucl.ac.uk>
# 3rd June 2020

from scipy.stats import lognorm
from scipy.stats import norm
import numpy as np
import random

class Flux:
    def __init__(self):
        self.name="Raw Flux"

    def pdf(self,x):
        return lognorm.pdf(x,s = 0.28076380165674314, loc = -0.5864272515923314, scale = 2.9210296869807912)

    def name(self):
        return self.name

class LogNormalFlux(Flux):
    #Here is the lognormal flux
    #
    def __init__(self, num = 0, bin_edges =[0,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5.0], bias=0 , resolution = 0, smear = False,shape=0.28076380165674314, loc=-0.5864272515923314, scale=2.9210296869807912, exposure=500):

        self.name="Lognormal Flux"
        self.shape=shape
        self.loc=loc
        self.scale=scale
        self.exposure=exposure
        self.smear = smear
        self.bias = bias
        self.resolution = resolution

        if smear == True:
            
            smeared = np.empty((0))

            for i in range(len(bin_edges)-1):
                mid = (bin_edges[i]+bin_edges[i+1])/2
                del_sd = resolution(mid)
                del_mu = bias(mid)

                #Find mean and sd with uncertainty

                cor_mean = mid-del_mu
                
                sd = 0.8702307 # standard deviation of true data
                cor_sd = sd+del_sd


                a = np.random.normal(cor_mean, cor_sd,num[i])
                smeared = np.append(smeared, a, axis =0)

            self.shape, self.loc, self.scale = lognorm.fit(smeared)

        
    
    def pdf(self,x):    
        return lognorm.pdf(x,self.shape,self.loc,self.scale)

    def flux(self,x):
        return self.exposure*self.pdf(x)

    

class FluxTools: 
    def __init__(self, xvals = np.linspace(0.25,5.25,501), binEdges = [0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5.0]):
        self.binEdges= binEdges
        self.xvals= xvals
        self.lastnumuflux=None
        self.lastnumuosccalc=None
        self.lastnueflux=None
        self.lastnueosccalc=None


    def pdfwrap(self,flux):
        return flux.pdf(self.xvals)

    def unity(self,x):
        return 1.0

    def getAsimov(self,flux,probcalc):
        fluxVals=flux.flux(self.xvals)
        binVals=[0]*(len(self.binEdges)-1)
        thisBin=0
        lastx=0
        for x,y in zip(self.xvals,fluxVals):
            if(thisBin<len(self.binEdges)-2):
                if(x>self.binEdges[thisBin+1]):
                    thisBin+=1
            binVals[thisBin]+=y*(x-lastx)*probcalc(x)
            lastx=x
        return binVals

    def getNoOscAsimov(self,flux):
        return self.getAsimov(flux,self.unity)

    def getNuMuAsimov(self, flux, osccalc,force=False):
        if(flux!=self.lastnumuflux or hash(osccalc)!=self.lastnumuosccalc or force):
            binVals=self.getAsimov(flux,osccalc.MuToMu)
            self.lastnumuflux=flux
            self.lastnumuosccalc=hash(osccalc)
            self.lastnumuVals=binVals
            return binVals
        return self.lastnumuVals
    
    def genNuMuExperiment(self,flux,osccalc):
        binVals=self.getNuMuAsimov(flux,osccalc)
        return [np.random.poisson(x) for x in binVals]


    def getNuElecAsimov(self, flux, osccalc,force=False):
        if(flux!=self.lastnueflux or hash(osccalc)!=self.lastnueosccalc or force):
            binVals=self.getAsimov(flux,osccalc.MuToElec)
            self.lastnueflux=flux
            self.lastnueosccalc=hash(osccalc)
            self.lastnueVals=binVals
            return binVals
        return self.lastnueVals
    
    def genNuElecExperiment(self,flux,osccalc):
        binVals=self.getNuElecAsimov(flux,osccalc)
        return [np.random.poisson(x) for x in binVals]

    def makeNuMuAsimovArray(self,flux,osccalc,dm32Array,sinSq23Array,dcpArray):
        a = []
        for dcp in dcpArray:
            mat = []
            for sSq23 in sinSq23Array:
                row = []
                for dm32 in dm32Array:
                    osccalc.updateOscParams(sinSqTheta23=sSq23,deltamSq32=dm32,dcp=dcp)
                    val=self.getNuMuAsimov(flux,osccalc,force=True)
                    row.append(val)
                mat.append(row)
            a.append(mat)
        return a

