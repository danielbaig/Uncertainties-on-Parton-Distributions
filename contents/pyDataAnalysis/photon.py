import numpy as np
import matplotlib.pyplot as plt
# Disable interactive mode.
plt.ioff()

from pathlib import Path
pathtohere = Path()

from .crossSection import CrossSection


LHC_s = (7e+3)**2 # [GeV^2] 
fineStructure = 1. / 137.035999206 # https://www.nature.com/articles/s41586-020-2964-7


class Photon(CrossSection):
    
    def __init__(self, WmassSqrd:float, ZmassSqrd:float,
                 data1:np.ndarray, data2:np.ndarray,x1:np.ndarray,
                 x2s:np.ndarray,qs:np.ndarray,W_index:int,
                 Z_index:int):
        """
        Initialise the class.
        
        Inputs:
            - WmassSqrd:float: Squared mass of the W boson.
            - ZmassSqrd:float: Squared mass of the Z boson.
            - data1:np.ndarray: PDFS associated with x1 (Q, x, PDF).
            - data2:np.ndarray: PDFS associated with x2 (Q, x, PDF).
            - x1:np.ndarray: Momentum fractions for proton 1.
            - x2s:np.ndarray: Momentum fractions for proton 2 at all Q.
            - qs:np.ndarray: Qs evaluated at.
            - W_index:int: Index of data for the W boson Q.
            - Z_index:int: Index of data for the Z boson Q.
        """
        CrossSection.__init__(self,data1,data2,x1,x2s,qs)

        self.W_index = W_index
        self.Z_index = Z_index
        
        self.sigma0 = 4.*np.pi*fineStructure*fineStructure / (3.*qs*qs)
        
        self.photon_dydm = None
        self.photon_dm = None
        
    
    
    def calculate_photon_dydm(self):
        """
        Calculate photon contribution to the cross section.
        """
        
            
        #include = np.argwhere


        upTypePDFs = np.delete(self.data[:,:,:,self.upType_i], (self.W_index,self.Z_index),1)
        downTypePDFs = np.delete(self.data[:,:,:,self.downType_i], (self.W_index,self.Z_index),1)
        upTypeBarPDFs = np.delete(self.data[:,:,:,self.upTypeBar_i], (self.W_index,self.Z_index),1)
        downTypeBarPDFs = np.delete(self.data[:,:,:,self.downTypeBar_i], (self.W_index,self.Z_index),1)
        
        
        
        # Photon
        self.photon_dydm = np.zeros((self.numQ,self.numX))
        for i in range(2):
            self.photon_dydm += self.upCharge*self.upCharge * np.sum(upTypePDFs[i] * upTypeBarPDFs[1-i], axis=2)
            self.photon_dydm += self.downCharge*self.downCharge * np.sum(downTypePDFs[i] * downTypeBarPDFs[1-i], axis=2)

            
        self.photon_dydm *= self.sigma0[2:,None] / (LHC_s * self.numColours)
        self.photon_dydm /= self.x1*self.x2s[2:]

        
    
    def calculate_photon_dm(self):
        """
        Integrate photon contribution over rapidity.
        """
        
        if self.photon_dydm is None:
            self.calculate_photon_dydm()
        
        ## The 2: slice here needs to be changed using np.argwhere.
        rapidities = 0.5*np.log(self.x1 / self.x2s[2:])
        
        
        # Integrate over rapidity.
        self.photon_dm = np.sum((self.photon_dydm[:,:-1] + self.photon_dydm[:,1:])
                                *(rapidities[:,1:] - rapidities[:,:-1]) / 2., axis=1)
        
    def calculate_photon(self):
        """
        Integrate photon contribution over mass.
        """
        
        if self.photon_dm is None:
            self.calculate_photon_dm()
            
        
        # Integrate over mass.
        self.photon = np.sum((self.photon_dm[:-1] + self.photon_dm[1:]) 
                             * (self.qs[1:]*self.qs[1:] - self.qs[:-1]*self.qs[:-1]) / 2.)
        self.photon *= 1e+9*0.3894 # [GeV^-2 -> pb]
    
    def display_photon_dm(self):
        """
        Display how the photon cross section contribution varies with the Q^2.
        """
        
        if self.photon_dm is None:
            raise Exception('calculate_photon_dm() has not been called.')
        
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()
        
        ax.scatter(self.qs*self.qs, self.photon_dm,c='orange')
        ax.grid()
        
        # Create appropiate labels.
        ax.set_xlabel(r'$Q^2$ [$GeV^2$]',fontsize=self.labelSize)
        ax.set_ylabel(r'differential cross section / $\frac{d\sigma}{dm^2}$ [pb $GeV^{-2}$]',fontsize=self.labelSize)
        ax.set_title(r'$\gamma$',fontsize=self.titleSize)
        
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.offsetText.set_fontsize(self.tickSize)

        
        plt.savefig(pathtohere / 'plots/photon_dm.png', bbox_inches='tight')
        plt.close(fig)
    