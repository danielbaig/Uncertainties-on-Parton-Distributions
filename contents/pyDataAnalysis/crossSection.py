import numpy as np
import matplotlib.pyplot as plt
# Disable interactive mode.
plt.ioff()

from pathlib import Path
pathtohere = Path()

class CrossSection:
    numUpType:int = 2
    upCharge = 2. / 3.
    numDownType:int = 3
    downCharge = -1./3.
    numColours:int = 3
    
    upType_i = slice(0,2)
    downType_i = slice(2,5)
    upTypeBar_i = slice(5,7)
    downTypeBar_i = slice(7,10)
    gluon_i = 10
    
    tickSize = 14
    labelSize = 16
    titleSize = 16
    
    
    
    def __init__(self,
                 data1:np.ndarray, data2:np.ndarray,x1:np.ndarray,
                 x2s:np.ndarray,qs:np.ndarray):
        """
        Initialise the class.
        
        Inputs:
            - data1:np.ndarray: PDFS associated with x1 (Q, x, PDF).
            - data2:np.ndarray: PDFS associated with x2 (Q, x, PDF).
            - x1:np.ndarray: Momentum fractions for proton 1.
            - x2s:np.ndarray: Momentum fractions for proton 2 at all Q.
            - qs:np.ndarray: Qs evaluated at.
        """
                 

        if data1.shape[1] != data2.shape[1]:
            raise Exception('Mismatch between num x1 and num x2,'
                            + ' ensure PDF generation has been run for both x1 and x2.')
        self.data = np.asarray((data1,data2))
        self.x1 = x1
        self.x2s = x2s
        self.numX = x1.shape[0]
        self.qs = qs[2:]
        self.numQ = qs.shape[0] - 2
        
        
        
        # To fix cut dependence func:
        self.Z = None
        self.W_dye = {
            '+' : None,
            '-' : None
        }
        self.W_dye_int = {
            '+' : None,
            '-' : None
        }
        
### Internal #########################################################
    @staticmethod
    def _ytocos(y:np.ndarray):
        """
        Transforms a variable in rapidity to one in cos theta.
        
        Inputs:
            - y:np.ndarray: Data points in rapidity.
            
        Outputs:
            - cos:np.ndarray: Data points in cos theta.        
        """

        return (np.exp(2.*y) - 1.) / (np.exp(2.*y) + 1.)
    
    @staticmethod
    def _costoy(cos:np.ndarray):
        """
        Transforms a variable in cos theta to one in rapidity.
        
        Inputs:
            - cos:np.ndarray: Data points in cos theta.
        
        Outputs:
            - y:np.ndarray: Data points in rapidity.
        """
        
        return 0.5 * np.log((1 + cos) / (1 - cos))
    
    @staticmethod
    def _dy_dcos(cos:np.ndarray):
        """
        Jacobian of transforming rapidity to cos theta (dy/dcostheta).
        
        Inputs:
            - cos:np.ndarray: Data in cos theta.
            
        Outputs:
            - dy_dcos:np.ndarray: Jacobian.
        
        """

        return 1. / ((1 - cos)*(1 + cos))
    
    @staticmethod
    def _dcos_dy(y:np.ndarray):
        """
        Jacobian of transforming cos theta to rapidity (dcostheta/dy).
        
        Inputs:
            - y:np.ndarray: Data in rapidity.
            
        Outputs:
            - dcos_dy:np.ndarray: Jacobian.
        """
        
        exp_y = np.exp(2*y)
        
        return 8*exp_y / (exp_y + 1) / (exp_y + 1)


    @staticmethod
    def _costopt(cos:np.ndarray, massSqrd:float):
        """
        Transforms a variable in cos theta to one in transverse momentum.
        
        Inputs:
            - cos:np.ndarray: Data points in cos theta.
            - massSqrd:float: Mass squared of boson produced.
            
        Outputs:
            - pt:np.ndarray: Data points in transverse momentum.
        """

        return np.sqrt(massSqrd) / 2. * np.sqrt(1. - cos*cos)
    
    @staticmethod
    def _pttocos(pT:np.ndarray, massSqrd:float):
        """
        Transforms a variable in transverse momentum to one in rapidity.
        
        Inputs:
            - pT:np.ndarray: Data points in transverse momentum.
            - massSqrd:float: Mass squared of boson produced.

        Outputs:
            - y:np.ndarray: Data points in rapidity.
        """
        
        return np.sqrt(1 - 4*pT*pT / massSqrd)
    
    
    
    @staticmethod
    def _dcos_dpt(pt:np.ndarray, massSqrd:float):
        """
        Jacobian of transforming cos theta to missing transverse momentum (dcostheta/dpt).
        
        Inputs:
            - pt:np.ndarray: Data in missing transverse momentum.
            - massSqrd:float: Mass squared of boson produced.
            
        Outputs:
            - dcos_dpt:np.ndarray: Jacobian.
        
        """

        return 4.*pt/massSqrd / np.sqrt(1. - 4.*pt*pt/massSqrd)
    
    def _getCuts(self,cutType:str, lowerCut:float,
                 upperCut:float, rapidities:np.ndarray,massSqrd:float):
        """
        
        Inputs:
            - cutType:str: Variable to perform cuts on.
            - lowerCut:float: Lower bound of cut.
            - upperCut:float: Upper bound of cut.
            - rapidities:float: The rapidities.
            - massSqrd:float: Boson mass squared.
        
        Outputs:
            - whereInclude:np.ndarray: What variables to include in the cut.
        """
        
        theOnes = np.ones(rapidities.shape[0],dtype=bool)
        
        # Implement cuts.
        if cutType is None:
            whereInclude = theOnes
        elif cutType=='rapidity':
            includeLower = rapidities>lowerCut if lowerCut is not None else theOnes
            includeUpper = rapidities<upperCut if upperCut is not None else theOnes
            whereInclude = includeLower & includeUpper
            jacobian = 1.
        elif cutType=='cos' or cutType=='pt':
            variableToCut = self._ytocos(rapidities)
            if cutType=='pt':
                variableToCut = self._costopt(variableToCut, massSqrd)
                
            includeLower = variableToCut>lowerCut if lowerCut is not None else theOnes
            includeUpper = variableToCut<upperCut if upperCut is not None else theOnes
            whereInclude = includeLower & includeUpper
            
        
        else:
            raise Exception('Invalid cut variable, must be one of: rapidity, cos, pt.')
            
        return whereInclude

    
    def calculate_cutRatioDependence(self, boson:str):
        """
        Calculate the cross-section ratio as the cut on the pt distribution is
        varied.
        
        Inputs:
            - boson:str: Boson being analysed.
        
        Outputs:
            - ptCuts:np.ndarray: Cuts applied to the pt distribution.
            - cs_ratios:np.ndarray: Cross-section ratio to the uncut cross-section.
        """
        
        assert boson in ('W+','W-','Z')
        i = np.argwhere(boson==np.asarray(('W+','W-','Z')))[0,0]
        
        numSamples = 20
        ptCuts = np.linspace(0,40, numSamples)
        cs_ratios = np.empty(numSamples)
        
        for j in range(numSamples):
            if i==2:
                self.calculate_Z(cutType='pt', lowerCut=ptCuts[j])
            else:
                self.calculate_W_dye(Wcharge=boson[-1],numTheta=1000,cutType='pt', lowerCut=ptCuts[j])
                self.calculate_W_dye_int(Wcharge=boson[-1])
                
            cs_ratios[j] = (self.W_dye_int['+'],self.W_dye_int['-'],self.Z)[i]
            
        cs_ratios /= cs_ratios[0]
        
        return ptCuts, cs_ratios

    
    def display_cutRatioDependence(self,ptCuts:np.ndarray, cs_ratios:np.ndarray, title:str):
        """
        Display how the cross-section ratio varies with pt cut.
        
        Inputs:
            - ptCuts:np.ndarray: Cuts applied to the pt distribution.
            - cs_ratios:np.ndarray: Cross-section ratio to the uncut cross-section.
            - title:str: Boson being analysed.
        """
        
        assert title in ('W+','W-','Z')
        
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()
        
        ax.scatter(ptCuts, cs_ratios,label='numeric')
        ax.grid()
        
        # Fit
        fittedFunc = lambda p: (2*np.sqrt(1 - 4*p*p/self.WmassSqrd) + 2/3*(1 - 4*p*p/self.WmassSqrd)**1.5) / (8/3)
        ax.plot(ptCuts, fittedFunc(ptCuts), c='g',label='analytic')
        
        # Create appropiate labels.
        ax.set_xlabel(r'$p_T$ cut [GeV]', fontsize=self.labelSize)
        ax.set_ylabel('differential cross section ratio', fontsize=self.labelSize)
        ax.set_title(title, fontsize=self.titleSize)
        ax.legend(loc='best',fontsize=self.labelSize)
        
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)
                
        if title in {'W+','W-'}:
            rootName = f'W_LO/{title}/'
        else:
            rootName = f'{title}_'
        plt.savefig(pathtohere / f'plots/{rootName}cutRatioDependence.png', bbox_inches='tight')
        plt.close(fig)
        
        
        

        
    @staticmethod
    def _shootDiag_inv(xstart:float,ystart:float,extent:tuple, grad:float=1.):
        """
        Determine the end of a diagonal straight line travelling (-1,1).
        
        Inputs:
            - xstart:float: x starting coordinate.
            - ystart:float: y starting coordinate.
            - extent:tuple: Extent of the graph.
            - grad:float: Negative gradient.
        """
        
        # Hits left boundary.
        if xstart - extent[0] < extent[3] - ystart:
            xend = extent[0]
            yend = ystart - grad*(xend - xstart)
        # Hits top boundary.
        elif xstart - extent[0] > extent[3] - ystart:
            yend = extent[3]
            xend = xstart - (yend - ystart) / grad
        # Hits corner exactly.
        else:
            xend = extent[0]
            yend = extent[3]
            
        return xend,yend
        