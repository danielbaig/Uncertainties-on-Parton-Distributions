import numpy as np
import matplotlib.pyplot as plt
# Disable interactive mode.
plt.ioff()

from tqdm import tqdm
from pathlib import Path
pathtohere = Path()

from ..crossSection import CrossSection


LHC_s = (7e+3)**2 # [GeV^2] 
CKM = np.asarray([[0.97373, 0.2243, 0.00382], [0.221, 0.975, 0.0408], [0.0086, 0.0415, 1.014]]) 
fermiConstant = 1.1663788e-5 # [GeV^-2] # https://arxiv.org/abs/1010.0991

BR_W_enu = 0.1083 # https://arxiv.org/abs/2201.07861



class Wboson_LO(CrossSection):
    
    def __init__(self, WmassSqrd:float,
                 data1:np.ndarray, data2:np.ndarray,x1:np.ndarray,
                 x2s:np.ndarray,qs:np.ndarray,W_index:int):
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
        """
        
        CrossSection.__init__(self,data1,data2,x1,x2s,qs)

                 
        self.WmassSqrd = WmassSqrd
        self.W_index = W_index
        
        self.W_dy = {
            '+' : None,
            '-' : None
        }
        self.W = {
            '+' : None,
            '-' : None
        }
        
        self.W_dye = {
            '+' : None,
            '-' : None
        }
        self.W_dye_int = {
            '+' : None,
            '-' : None
        }
    
    def calculate_W_dy(self, Wcharge:str):
        """
        Calculate the W boson contribution to the cross-section.
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."

        upTypePDFs = self.data[:,self.W_index,:,self.upType_i]
        downTypePDFs = self.data[:,self.W_index,:,self.downType_i]
        upTypeBarPDFs = self.data[:,self.W_index,:,self.upTypeBar_i]
        downTypeBarPDFs = self.data[:,self.W_index,:,self.downTypeBar_i]
        
        
        W_dy = np.zeros((2,self.numX))
        for i in range(self.numUpType):
            for j in range(self.numDownType):
                poisson = np.zeros((2,self.numX))
                for k in range(2):
                    if Wcharge=='+':
                        poisson[k] += upTypePDFs[k,:,i] * downTypeBarPDFs[1-k,:,j]

                    else:
                        poisson[k] += upTypeBarPDFs[k,:,i] * downTypePDFs[1-k,:,j]

                W_dy += CKM[i,j]*CKM[i,j] * poisson

        W_dy *= np.pi/self.numColours*np.sqrt(2.) * fermiConstant*self.WmassSqrd
        W_dy /= LHC_s
        W_dy /= self.x1*self.x2s[self.W_index]
        W_dy *= 1e+9*0.3894 # [GeV^-2 -> pb]
        W_dy *= BR_W_enu
        
        self.W_dy[Wcharge] = W_dy



    def calculate_W(self, Wcharge:str, cutType:str=None, lowerCut:float=None, upperCut:float=None):
        """
        Integrate W contribution over rapidity.
        
        Inputs:
            - Wcharge:str: W boson charge to integrate.
            - cutType:str: Variable to perform cuts on.
            - lowerCut:float: Lower bound of cut.
            - upperCut:float: Upper bound of cut.
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."

        
        self.rapidities_W = 0.5 * np.log(self.x1 / self.x2s[self.W_index])

        whereInclude = self._getCuts(cutType, lowerCut,
                 upperCut, self.rapidities_W, self.WmassSqrd)

            
        if self.W_dy[Wcharge] is None:
            self.calculate_W_dy(Wcharge)
        

        # Determine active changes.
        splits = [0]
        prev = whereInclude[0]

        for i in range(1,self.numX):
            if whereInclude[i] != prev:
                prev = whereInclude[i]
                splits.append(i)
                
        splits.append(self.numX)
                
        
        # Integrate over rapidity.
        self.W[Wcharge] = 0.
        for k in range(2):
            for i in range(len(splits)-1):
                if not whereInclude[splits[i]]:
                    continue
                binRange = slice(splits[i],splits[i+1])
                self.W[Wcharge] += np.sum((self.W_dy[Wcharge][k][binRange][:-1] + self.W_dy[Wcharge][k][binRange][1:])
                                *(self.rapidities_W[binRange][1:] - self.rapidities_W[binRange][:-1]) / 2.)
            
    
        
    
### Display ##################################################################
    def display_W(self, Wcharge, xaxis:str='rapidity'):
        """
        Display how the W cross section contribution varies with a quantity.
        
        Inputs:
            - xaxis:str: x-axis variable to use.
            - Wcharge:str: W boson charge to display (+/-).
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."

        
        if xaxis=='rapidity':
            x = self.rapidities_W
            y_label = r'differential cross section / $\frac{d\sigma}{dy}$ [pb]'
            jacobian = 1.
        elif xaxis=='cos' or xaxis=='pt':
            x = self._ytocos(self.rapidities_W)
            jacobian = self._dy_dcos(x)
            y_label = r'differential cross section / $\frac{d\sigma}{d\cos\theta^*}$ [pb]'
            
            if xaxis=='pt':
                x = self._costopt(x,self.WmassSqrd)
                jacobian *= self._dcos_dpt(x,self.WmassSqrd)
                y_label = r'differential cross section / $\frac{d\sigma}{dp_T}$ [pb $GeV^{-1}$]'
            

        else:
            raise Exception('Unknown xaxis.')
        
        if self.W_dy[Wcharge] is None:
            raise Exception('calculate_W_dy() has not been called.')
            
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()
        
        ax.scatter(x, np.sum(self.W_dy[Wcharge],axis=0)*jacobian,c='b')
        ax.grid()
        
        # Create appropiate labels.
        if xaxis=='rapidity':
            ax.set_xlabel(r'W boson rapidity / $y_W$', fontsize=self.labelSize)
        elif xaxis=='cos':
            ax.set_xlabel(r'$\cos \theta^*$',fontsize=self.labelSize)
        elif xaxis=='pt':
            ax.set_xlabel(r'$p_{Te}$ [GeV]',fontsize=self.labelSize)
        ax.set_ylabel(y_label,fontsize=self.labelSize)
        ax.set_title(f'W{Wcharge}',fontsize=self.titleSize)
        
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)
        
        plt.savefig(pathtohere / f'plots/W_LO/W{Wcharge}/W.png', bbox_inches='tight')
        plt.close(fig)
        
        
    ########## Electron rapidity #################################################################
    def calculate_W_dye_int(self,Wcharge:str):
        """
        Integrate the W boson contribution as a function of electron rapidity.
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."
        
        if self.W_dye[Wcharge] is None:
            print(f'Warning: W_dye[{Wcharge}] not calculated, calculating with default numTheta=100.')
            self.calculate_W_dye(Wcharge,100)
        
        
        # Integrate over rapidity_e.
        self.W_dye_int[Wcharge] = np.sum((self.W_dye[Wcharge][:-1] + self.W_dye[Wcharge][1:])
                        *(self.rapidities_W_ye[1:] - self.rapidities_W_ye[:-1]) / 2.)
    
    def calculate_W_dye(self,Wcharge:str,numTheta:int, cutType:str=None,
                                             lowerCut:float=None,upperCut:float=None, displayContours:bool=False):
        """
        Create a visulisation for the integral to implement the theta
        dependence of the emitted lepton.
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
            - numTheta:int: Number of angle points to take of the electron wrt W boson.
            - cutType:str: Variable to apply cuts on.
            - lowerCut:float: Lower cut boundary.
            - upperCut:float: Upper cut boundary.
            - displayContours:bool: Whether to display the 2D cross section heatmap.
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."

        rapidities_e_2d,sigma_e,extent,y_star = self._get_ye_properties(Wcharge, numTheta)
            
        whereInclude = self._getCuts(cutType, lowerCut,
                 upperCut, y_star,self.WmassSqrd)
        
        sigma_e[:,np.logical_not(whereInclude)] = 0.
        
        if displayContours:
            self._display_rapidityIntegralContours(sigma_e,extent,Wcharge)
        
        self.rapidities_W_ye,self.W_dye[Wcharge] = self._integrateRapidityContours(sigma_e,
                                                                   rapidities_e_2d,extent,y_star,numTheta)
         
        
        
    def display_W_dye(self,Wcharge:str):
        """
        Display how the cross section varies with electron rapidity.
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."
        
        if Wcharge=='total':
            for Wq in ('+','-'):
                if self.W_dye[Wq] is None:
                    self.display_thetaConvolutionVisulisation(Wq)
                    
            Wcont = self.W_dye['+'] + self.W_dye['-']
            title = r'$W^+$+$W^-$'
        
        else:
            if self.W_dye[Wcharge] is None:
                self.display_thetaConvolutionVisulisation(Wcharge)
            Wcont = self.W_dye[Wcharge]
            title = f'W{Wcharge}'
            
                
        whereInclude = np.where((self.rapidities_W_ye>self.rapidities_W.min()) 
                                & (self.rapidities_W_ye<self.rapidities_W.max()))
        
        # Generate figure.
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()
        
        ax.scatter(self.rapidities_W_ye[whereInclude][1:-1], Wcont[whereInclude][1:-1],c='m')
        ax.grid()
        
        # Create appropiate labels.
        ax.set_xlabel(r'electron rapidity / $y_e$',fontsize=self.labelSize)
        ax.set_ylabel(r'differential cross section  / $\frac{d\sigma}{dy_e}$ [pb]', fontsize=self.labelSize)
        ax.set_title(title, fontsize=self.titleSize)
        
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)
        
        plt.savefig(pathtohere / f'plots/W_LO/W{Wcharge}/W_dye.png', bbox_inches='tight')
        plt.close(fig)
        
    def _integrateRapidityContours(self, sigma_e:np.ndarray,rapidities_e:np.ndarray, extent:tuple,
                                   y_star:np.ndarray,numTheta:int):
        """
        Integrate the convoluted cross section over given rapidity contours.
        
        Inputs:
            - sigma_e:np.ndarray: Convoluted differential cross sections.
            - rapidities_e:np.ndarray: Matrix of added rapidities orthogonally.
            - extent:tuple: Extent of the rapidity.
            - y_star:np.ndarray: Rapidities of the lepton angle wrt W boson.
            - numTheta:int: Number of theta (lepton to W boson) values used.
            
        Outputs:
            - y_e_wanted:np.ndarray: Rapidity contours integrated over.
            - cs_e:np.ndarray: Cross section distribution wrt rapidities_e.
        """
        
        numYWanted = 30
        y_e_binWidth = 0.21 # https://arxiv.org/abs/1612.03016
        y_e_wanted = np.arange(-3,3,y_e_binWidth)
        
        cs_e = np.empty_like(y_e_wanted,dtype=float)
        
        # For each rapidities_e find which y_e_wanted it is closest to.
        rapidities_e_centre = (rapidities_e[1:,1:] + rapidities_e[:-1,:-1]) / 2
        contourDiff = abs(rapidities_e_centre[:,:,None] - y_e_wanted[None,None,:])
        
        closest = np.empty(contourDiff.shape[:2])
        for i in range(contourDiff.shape[0]):
            for j in range(contourDiff.shape[1]):
                closest[i,j] = np.argwhere(contourDiff[i,j]==np.min(contourDiff[i,j]))[0][0]

        # Determine all 2D bin areas.
        binAreas = np.abs((self.rapidities_W[1:] - self.rapidities_W[:-1])[:,None] @ (y_star[1:] - y_star[:-1])[None,:])
        sigma_centres = (sigma_e[:-1,:-1] + sigma_e[1:,1:]) / 2.
        

        for i in range(len(y_e_wanted)):
            include = np.where(closest==i)
            cs_e[i] = np.sum(sigma_centres[include] * binAreas[include])
        cs_e /= y_e_binWidth
            
        return y_e_wanted,cs_e
        
        
    def _get_ye_properties(self, Wcharge:str, numTheta:int, starParameter:str='rapidity'):
        """
        Get the matrix rapidities and differential cross sections of the rapidity
        convolution (in addition to the rapidity extent and y_star.)
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
            - numTheta:int: Number of angle samples of the lepton wrt W boson.
            - starParameter:str: Parameter of the electron's angle wrt to the W boson ('rapidity'/'pT').
            
        Outputs:
            - rapidities_e:np.ndarray: Matrix of added rapidities orthogonally.
            - sigma_e:np.ndarray: Convoluted differential cross sections.
            - extent:tuple: Extent of the rapidity.
            - y_star:np.ndarray: Rapidities of the lepton angle wrt W boson.
        """
                
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."
        assert starParameter in {'rapidity', 'pT'}, "starParameter must be one of {'rapidity','pT'}."
        
        # Lepton angle wrt the W boson.
        thetas_star = np.linspace(0,2*np.pi,numTheta+2)[1:-1][::-1]
        y_star = 0.5*np.log((1 + np.cos(thetas_star)) / (1 - np.cos(thetas_star)))
        pT_star = self._costopt(np.cos(thetas_star), self.WmassSqrd)
        rapidities_e = self.rapidities_W[:,None] + y_star[None,:]

        
        sigma_e = np.zeros((self.rapidities_W.shape[0],numTheta))
        for k in range(2):
            # Probability distribution as a function of theta.
            if (Wcharge=='+' and k==1) or (Wcharge=='-' and k==0):
                sigma_theta = (1. + np.cos(thetas_star))**2
            else:
                sigma_theta = (1. - np.cos(thetas_star))**2

            if starParameter=='rapidity':
                sigma_y_star = self._dcos_dy(y_star) * sigma_theta
            elif starParameter=='pT':
                sigma_y_star = self._dcos_dpt(pT_star, self.WmassSqrd) * sigma_theta
            sigma_e += self.W_dy[Wcharge][k][:,None] @ sigma_y_star[None,:]

        
        extent = (y_star.min(),y_star.max(),self.rapidities_W[0],self.rapidities_W[-1])
        
        
        return rapidities_e, sigma_e, extent, y_star
        
    def _display_rapidityIntegralContours(self, sigma_e:np.ndarray, extent:tuple, Wcharge:str):
        """
        Display a figure showing the cross section heatmap with example
        integral contours in red.
        
        Inputs:
            - sigma_e:np.ndarray: Matrix of convoluted differential cross sections.
            - extent:tuple: Extent of the rapidity ranges.
            - Wcharge:str: W boson charge to calculate (+/-).
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."

        # Integral example contours.
        xstarts = np.linspace(extent[0],extent[1],4+1)[1:]
        ystarts = np.linspace(extent[2],extent[3],3+2)[1:-1]
        
        aspect = abs((extent[1] - extent[0]) / (extent[3] - extent[2]))


        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()

        # Create heatmap.
        im = ax.imshow(sigma_e, origin='lower',cmap='Greens',extent=extent,aspect=aspect, interpolation=None)
        ax.contour(sigma_e, origin='lower',colors='cyan',extent=extent)
        cbar = fig.colorbar(im, ax=ax)
        
        # Create integral contours starting on the bottom.
        ystart = extent[2]
        for xstart in xstarts:
            xend,yend = self._shootDiag_inv(xstart,ystart,extent)
            ax.plot((xstart,xend),(ystart,yend),c='r')
            
        # Create integral contours starting on the right side.
        xstart = extent[1]
        for ystart in ystarts:
            xend,yend = self._shootDiag_inv(xstart,ystart,extent)
            ax.plot((xstart,xend),(ystart,yend),c='r')
            

        # Create appropiate labels.
        ax.set_title(r'$\sigma(y^*) * \sigma(y_W)$', fontsize=self.titleSize)
        ax.set_xlabel(r'$y^*$',fontsize=self.labelSize)
        ax.set_ylabel(r'$y_W$',fontsize=self.labelSize)
        cbar.ax.yaxis.set_tick_params(labelsize=self.tickSize)
        
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)

        plt.savefig(pathtohere / f'plots/W_LO/W{Wcharge}/W_dywdystar.png', bbox_inches='tight')
        plt.close(fig)
        
        
    def calculate_thetaIntegralVariation(self, Wcharge:str, minNumTheta:int,
                                         maxNumTheta:int, numThetaSamples:int):
        """
        Calculate the integral over y_e for a W boson for various numTheta
        values on a logarithmic scale.
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
            - minNumTheta:int: Lower numTheta value to calculate.
            - maxNumTheta:int: Upper numTheta value to calculate.
            - numThetaSamples:int: Number of numTheta values to test.
        
        Outputs:
            - numThetas:np.ndarray: numThetas tested.
            - cs_W_ye_int:np.ndarray: Total cross-section contributions.
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."

        
        numThetas = np.logspace(np.log10(minNumTheta), np.log10(maxNumTheta),
                                numThetaSamples,dtype=int)
        
        cs_W_ye_int = np.empty(numThetaSamples)
        for i,numTheta in enumerate(tqdm(numThetas)):
            self.calculate_W_dye(Wcharge,numTheta)
            self.calculate_W_dye_int(Wcharge)
            
            cs_W_ye_int[i] = self.W_dye_int[Wcharge]
            
        return numThetas, cs_W_ye_int
            
        
    def display_thetaIntegralVariation(self,Wcharge,numThetas:np.ndarray,cs_W_ye_int):
        """
        Display how the integral for the W_ye contribution varies as numTheta varies.
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
            - numThetas:np.ndarray: numThetas tested.
            - cs_W_ye_int:np.ndarray: Total cross-section contributions.
        """
        
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()
        
        ax.scatter(numThetas, cs_W_ye_int, c='r')
        ax.grid()
        
        # Create appropiate labels.
        ax.set_xlabel('number of theta values',fontsize=self.labelSize)
        ax.set_ylabel('W boson total cross-section',fontsize=self.labelSize)
        ax.set_title(f'W{Wcharge}',fontsize=self.titleSize)
        
        ax.set_xscale('log')
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)
        
        plt.savefig(pathtohere / f'plots/W_LO/W{Wcharge}/W_ye_integral_convergence.png', bbox_inches='tight')
        plt.close(fig)