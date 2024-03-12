import numpy as np
import matplotlib.pyplot as plt
# Disable interactive mode.
plt.ioff()

from pathlib import Path
pathtohere = Path()

from .crossSection import CrossSection


LHC_s = (7e+3)**2 # [GeV^2] 

fermiConstant = 1.1663788e-5 # [GeV^-2] # https://arxiv.org/abs/1010.0991
sin2WeinbergAngle = 0.23142 # https://arxiv.org/abs/1509.07645

BR_Z_ee  = 0.03363 # https://pdg.lbl.gov/2009/tables/rpp2009-sum-gauge-higgs-bosons.pdf

class Zboson(CrossSection):
    def __init__(self, ZmassSqrd:float,
                 data1:np.ndarray, data2:np.ndarray,x1:np.ndarray,
                 x2s:np.ndarray,qs:np.ndarray,
                 Z_index:int):
        """
        Initialise the class.
        
        Inputs:
            - ZmassSqrd:float: Squared mass of the Z boson.
            - data1:np.ndarray: PDFS associated with x1 (Q, x, PDF).
            - data2:np.ndarray: PDFS associated with x2 (Q, x, PDF).
            - x1:np.ndarray: Momentum fractions for proton 1.
            - x2s:np.ndarray: Momentum fractions for proton 2 at all Q.
            - qs:np.ndarray: Qs evaluated at.
            - Z_index:int: Index of data for the Z boson Q.
        """
        CrossSection.__init__(self,data1,data2,x1,x2s,qs)

                 
        self.ZmassSqrd = ZmassSqrd
        self.Z_index = Z_index
                
        self.Z_dy = None
        self.Z = None
        
    
    def calculate_Z_dy(self):
        """
        Calculate the Z boson contribution to the cross-section.
        """

        
        upTypePDFs = self.data[:,self.Z_index,:,self.upType_i]
        downTypePDFs = self.data[:,self.Z_index,:,self.downType_i]
        upTypeBarPDFs = self.data[:,self.Z_index,:,self.upTypeBar_i]
        downTypeBarPDFs = self.data[:,self.Z_index,:,self.downTypeBar_i]

        
        self.Z_dy = np.zeros(self.numX)

        # Up-type.
        V_q = 0.5 -2.*self.upCharge*sin2WeinbergAngle
        A_q = 0.5
        poisson = np.zeros(self.numX)
        for i in range(2):
            poisson += np.sum(upTypePDFs[i] * upTypeBarPDFs[1-i], axis=1)
        self.Z_dy += (V_q*V_q + A_q*A_q)*poisson*self.upCharge*self.upCharge
        

        # Down-type.
        V_q = -0.5 - 2.*self.downCharge*sin2WeinbergAngle
        A_q = -0.5
        poisson = np.zeros(self.numX)
        for i in range(2):
            poisson += np.sum(downTypePDFs[i] * downTypeBarPDFs[1-i], axis=1)
        self.Z_dy += (V_q*V_q + A_q*A_q)*poisson*self.downCharge*self.downCharge

        self.Z_dy *= np.pi/self.numColours*np.sqrt(2.) * fermiConstant*self.ZmassSqrd
        self.Z_dy /= LHC_s
        self.Z_dy /= self.x1*self.x2s[self.Z_index]

        self.Z_dy *= 1e+9*0.3894 # [GeV^-2 -> pb]
        self.Z_dy *= BR_Z_ee

        
        
    def calculate_Z(self, cutType:str=None, lowerCut:float=None, upperCut:float=None):
        """
        Integrate Z contribution over rapidity.
        
        Inputs:
            - cutType:str: Variable to perform cuts on.
            - lowerCut:float: Lower bound of cut.
            - upperCut:float: Upper bound of cut.
        """

        self.rapidities_Z = 0.5 * np.log(self.x1 / self.x2s[self.Z_index])

        whereInclude = self._getCuts(cutType, lowerCut,
                 upperCut, self.rapidities_Z,self.ZmassSqrd)

        if self.Z_dy is None:
            self.calculate_Z_dy()

        # Determine active changes.
        splits = [0]
        prev = whereInclude[0]

        for i in range(1,self.numX):
            if whereInclude[i] != prev:
                prev = whereInclude[i]
                splits.append(i)
                
        splits.append(self.numX)
                
        
        # Integrate over rapidity.
        self.Z = 0.
        for i in range(len(splits)-1):
            if not whereInclude[splits[i]]:
                continue
            binRange = slice(splits[i],splits[i+1])
            self.Z += np.sum((self.Z_dy[binRange][:-1] + self.Z_dy[binRange][1:])
                            *(self.rapidities_Z[binRange][1:] - self.rapidities_Z[binRange][:-1]) / 2.)
            
            
            
    def display_Z(self, xaxis:str='rapidity'):
        """
        Display how the Z cross section contribution varies with a quantity.
         
        Inputs:
            - xaxis:str: x-axis variable to use.
        """
        
        if xaxis=='rapidity':
            x = self.rapidities_Z
            jacobian = 1.
            y_label = r'differential cross section / $\frac{d\sigma}{dy_Z}$ [pb]'
        elif xaxis=='cos' or xaxis=='pt':
            x = self._ytocos(self.rapidities_Z)
            jacobian = self._dy_dcos(x)
            y_label = r'differential cross section / $\frac{d\sigma}{d\cos\theta^*}$ [pb]'

            if xaxis=='pt':
                x = self._costopt(cos, self.ZmassSqrd)
                jacobian *= self._dcos_dpt(x, self.ZmassSqrd)
                y_label = r'differential cross section / $\frac{d\sigma}{dp_T}$ [pb $GeV^{-1}$]'
        else:
            raise Exception('Unknown xaxis.')
        
        if self.Z_dy is None:
            raise Exception('calculate_Z_dy() has not been called.')
            
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()
        
        ax.scatter(x, self.Z_dy*jacobian,c='orange')
        ax.grid()
        
        # Create appropiate labels.
        if xaxis=='rapidity':
            ax.set_xlabel(r'Z boson rapidity / $y_Z$',fontsize=self.labelSize)
        elif xaxis=='cos':
            ax.set_xlabel(r'$\cos \theta^*$',fontsize=self.labelSize)
        elif xaxis=='pt':
            ax.set_xlabel(r'$p_{Te}$ [GeV]',fontsize=self.labelSize)        
        ax.set_ylabel(y_label,fontsize=self.labelSize)
        ax.set_title('Z',fontsize=self.titleSize)
        
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)
        
        plt.savefig(pathtohere / 'plots/Z.png', bbox_inches='tight')
        plt.close(fig)
        
        
        
    