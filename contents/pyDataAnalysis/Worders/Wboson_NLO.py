from tqdm import tqdm
from time import perf_counter

from pyDataAnalysis.crossSection import *

LHC_s = (7e+3)**2 # [GeV^2] 
CKM = np.asarray([[0.97373, 0.2243, 0.00382], [0.221, 0.975, 0.0408], [0.0086, 0.0415, 1.014]]) 
fermiConstant = 1.1663788e-5 # [GeV^-2] # https://arxiv.org/abs/1010.0991
electronCharge = 1.6e-19
alpha_s = 0.1183 # https://arxiv.org/abs/2309.12986
alpha_W = 1 / 29.5 # Griffiths D. Introduction to elementary particles. John Wiley & Sons; 2020 Dec 18. p.315

BR_W_enu = 0.1083 # https://arxiv.org/abs/2201.07861

class Wboson_NLO(CrossSection):
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
        
        self.x2 = x1

        self.s_hat = LHC_s*self.x1[:,None]*self.x2[None,:]

        self.num_pT = 20
        self.WmassSqrd = WmassSqrd


        self.pTs = np.linspace(5,180,self.num_pT)

        self.costheta_W = np.sqrt(1 - 4*self.pTs[None,None,:]**2 *self.s_hat[:,:,None] 
                           / (self.s_hat[:,:,None]-self.WmassSqrd)**2)
        self.rapidities_W = 0.5*np.log((self.s_hat[:,:,None] + self.WmassSqrd 
                                       + (self.s_hat[:,:,None] - self.WmassSqrd)*self.costheta_W)
                                      / (self.s_hat[:,:,None] + self.WmassSqrd 
                                       - (self.s_hat[:,:,None] - self.WmassSqrd)*self.costheta_W))


        epsilon = 0.25 * self.pTs[None,None,:]
        self.threshold = (self.pTs[None,None,:] + np.sqrt(self.WmassSqrd + self.pTs[None,None,:]**2) 
                          + epsilon)
                         
        self.W_index = W_index
        
        self.W_dx1dx2dpT = {
            '+' : None,
            '-' : None
        }
        self.W_dpT = {
            '+' : None,
            '-' : None
        }
        
        self.W_dyW = {
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
        
    def calculate_W_dx1dx2dpT(self, Wcharge:str):
        """
        Calculate the W boson contribution to the cross-section as a function of
        x1, x2 and transverse momentum.
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."
        

        upTypePDFs = self.data[:,self.W_index,:,self.upType_i]
        downTypePDFs = self.data[:,self.W_index,:,self.downType_i]
        upTypeBarPDFs = self.data[:,self.W_index,:,self.upTypeBar_i]
        downTypeBarPDFs = self.data[:,self.W_index,:,self.downTypeBar_i]
        
        gluonPDF = self.data[:,self.W_index,:,self.gluon_i]
        
        # Set x2 PDFs to x1 PDFs.
        upTypePDFs[1] = upTypePDFs[0]
        downTypePDFs[1] = downTypePDFs[0]
        upTypeBarPDFs[1] = upTypeBarPDFs[0]
        downTypeBarPDFs[1] = downTypeBarPDFs[0]
        gluonPDF[1] = gluonPDF[0]
        
        
        
        # Derived expressions
        whereInvalid = np.where(np.sqrt(self.s_hat[:,:,None]) < self.threshold)

        
        dsigma_dpT_qq = (16*alpha_s*alpha_W*alpha_W
                        *((self.WmassSqrd*self.WmassSqrd + self.s_hat[:,:,None]
                           *(self.s_hat[:,:,None] - 2*self.pTs[None,None,:]**2))
                        / (27*self.pTs[None,None,:]*self.WmassSqrd
                           *(self.WmassSqrd - self.s_hat[:,:,None])*self.s_hat[:,:,None]**2*-self.costheta_W)))
        
        
        dsigma_dpT_qg = ((2*alpha_s*alpha_W*alpha_W
                          *self.pTs[None,None,:] *(-3*self.WmassSqrd*self.WmassSqrd 
                          + 2*self.pTs[None,None,:]**2 * self.s_hat[:,:,None]
                          + 4*self.WmassSqrd* self.s_hat[:,:,None] - 3*self.s_hat[:,:,None]**2
                          + (self.WmassSqrd*self.WmassSqrd + self.s_hat[:,:,None]**2)* -self.costheta_W))
                            /(9*self.WmassSqrd *(self.WmassSqrd - self.s_hat[:,:,None])**2 * self.s_hat[:,:,None]**2
                                                     * -self.costheta_W* (1 - self.costheta_W)))
        
        # Set unphysical cs to zero.
        dsigma_dpT_qq[whereInvalid] = 0.
        dsigma_dpT_qg[whereInvalid] = 0.
        
        if np.any(dsigma_dpT_qq<0):
            raise Exception('There are elements of dsigma_dpT_qq<0.')
        if np.any(dsigma_dpT_qg<0):
            raise Exception('There are elements of dsigma_dpT_qg<0.')
        
        
        # PDF bracket term.
        W_dy_qq = np.zeros((2,self.numX,self.numX))
        W_dy_qg = np.zeros((2,self.numX,self.numX))
        for i in range(self.numUpType):
            for j in range(self.numDownType):
                poisson_qq = np.zeros((2,self.numX,self.numX))
                poisson_qg = np.zeros((2,self.numX,self.numX))
                for k in range(2):
                    if Wcharge=='+':
                        poisson_qq[k] += upTypePDFs[0,:,None,i] * downTypeBarPDFs[1,None,:,j]
                        poisson_qg[k] += (upTypePDFs[0,:,None,i] * gluonPDF[1,None,:] 
                                       + gluonPDF[0,:,None] * downTypeBarPDFs[1,None,:,j])
                        poisson_qq[k] += upTypePDFs[1,None,:,i] * downTypeBarPDFs[0,:,None,j]
                        poisson_qg[k] += (upTypePDFs[1,None,:,i] * gluonPDF[0,:,None] 
                                       + gluonPDF[1,None,:] * downTypeBarPDFs[0,:,None,j])
                        

                    else:
                        poisson_qq[k] += upTypeBarPDFs[0,:,None,i] * downTypePDFs[1,None,:,j]
                        poisson_qg[k] += (upTypeBarPDFs[0,:,None,i] * gluonPDF[1,None,:]
                                       + gluonPDF[0,:,None] * downTypePDFs[1,None,:,j])
                        poisson_qq[k] += upTypeBarPDFs[1,None,:,i] * downTypePDFs[0,:,None,j]
                        poisson_qg[k] += (upTypeBarPDFs[1,None,:,i] * gluonPDF[0,:,None]
                                       + gluonPDF[1,None,:] * downTypePDFs[0,:,None,j])


                W_dy_qq += CKM[i,j]*CKM[i,j] * poisson_qq
                W_dy_qg += CKM[i,j]*CKM[i,j] * poisson_qg
                
        
        # Convolute with kinematic differential cross-sections
        W_dx1dx2dpT = (W_dy_qq[:,:,:,None] * dsigma_dpT_qq[None,:,:,:] 
                        + W_dy_qg[:,:,:,None] * dsigma_dpT_qg[None,:,:,:])
        
        W_dx1dx2dpT /= (self.x1[:,None]*self.x2[None,:])[None,:,:,None]
        W_dx1dx2dpT *= 1e+9*0.3894 # [GeV^-2 -> pb]
        W_dx1dx2dpT *= BR_W_enu
        
        self.W_dx1dx2dpT[Wcharge] = W_dx1dx2dpT

       
        
    def calculate_W_dpT(self, Wcharge:str):
        """
        Integrate W contribution over both x_1 and x_2 over the region where the W boson can be generated.
        
        Inputs:
            - Wcharge:str: W boson charge to integrate.
        """
        
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."
            
        if self.W_dx1dx2dpT[Wcharge] is None:
            self.calculate_W_dx1dx2pT(Wcharge)
        

        # Region where there is sufficient energy to generate a W boson.
        whereInclude = np.argwhere((np.sqrt(self.s_hat[:-1,:-1,None]) > self.threshold) 
                                   & (np.sqrt(self.s_hat[1:,1:,None]) > self.threshold))
        
        binAreas = ((self.x1[1:] - self.x1[:-1])[:,None] 
                    * (self.x2[1:] - self.x2[:-1])[None,:])
        sigma_centres = (self.W_dx1dx2dpT[Wcharge][:,:-1,:-1,:] + self.W_dx1dx2dpT[Wcharge][:,1:,1:,:]) / 2.
    
        # Add integral contributions.
        self.W_dpT[Wcharge] = np.zeros(self.num_pT)
        for i,j,m in whereInclude:
            for k in range(2):
                # Impose pT<=p_W
                self.W_dpT[Wcharge][m] += sigma_centres[k,i,j,m] * binAreas[i,j]
                
                
        if np.any(self.W_dpT[Wcharge][:-1] - self.W_dpT[Wcharge][1:] < 0):
            print('Warning: pT distribution is not all decreasing. epsilon should be increased.')
            print(self.W_dpT[Wcharge][:-1] - self.W_dpT[Wcharge][1:])
            
            
    def calculate_W_dyW(self,Wcharge:str):
        """
        Determine the distribution of the cross-sections wrt the W boson rapidity.
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."
        
        if self.W_dx1dx2dpT[Wcharge] is None:
            self.calculate_W_dx1dx2dpT(Wcharge)
        
        whereInvalid = (np.sqrt(self.s_hat[:,:,None]) < self.threshold)
        
        for i in range(2):
            Wsign = 1. - 2.*i
        
            # Determine sign on y_W.
            momentumDifference = ((self.x1[:,None,None] - self.x2[None,:,None]) * np.sqrt(LHC_s)/2. 
                                    + (self.s_hat[:,:,None]-self.WmassSqrd)/(2*np.sqrt(self.s_hat[:,:,None])) 
                                  * Wsign * self.costheta_W)
            momentumSign = np.zeros(momentumDifference.shape)
            momentumSign[(momentumDifference<0) & np.logical_not(whereInvalid)] = -1.
            momentumSign[(momentumDifference>0) & np.logical_not(whereInvalid)] = 1.
            
            cs_temp = self._integrate_to_dyW(Wcharge, momentumSign)
            
            if i==0:
                self.W_dyW[Wcharge] = np.zeros((2,self.yW_bins.shape[0]),dtype=float)
            self.W_dyW[Wcharge] += cs_temp
        
                
        
            
    def _integrate_to_dyW(self,Wcharge:str, momentumSign:np.ndarray):
        """
        Integrate over free parameters and bin into yW.
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
            - momentumSign:np.ndarray: Sign of yW due to emitted W boson direction.
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."
        
        # Define bins.
        num_yW = 78
        self.yW_bins = np.linspace(-1.5,1.5,num_yW)
        y_W_binWidth = (self.yW_bins[-1] - self.yW_bins[0]) / num_yW
        
        # Get distance between all points and contours.
        rapidities_W_centre = ((momentumSign*self.rapidities_W)[1:,1:,1:] + (momentumSign*self.rapidities_W)[:-1,:-1,:-1]) / 2
        contourDiff = np.abs(rapidities_W_centre[:,:,:,None] - self.yW_bins[None,None,None,:])
        
        isAllowed = ((np.sqrt(self.s_hat[:-1,:-1,None]) > self.threshold) 
                                   & (np.sqrt(self.s_hat[1:,1:,None]) > self.threshold))
        isAllowed = isAllowed[:,:,1:] & isAllowed[:,:,:-1]
        cs_yW = np.empty((2,num_yW),dtype=float)
        for k in range(2):

            closest = -np.ones(contourDiff.shape[:3],dtype=int)
            # Find the indices of minimum values along the last axis of contourDiff
            closest = np.argmin(contourDiff, axis=3)
            
            # Assign the minimum indices to the corresponding positions in closest
            closest[np.logical_not(isAllowed)] = -1

            # Determine all 3D bin areas.
            binAreas = np.abs((momentumSign*self.rapidities_W)[1:,1:,1:] - (momentumSign*self.rapidities_W)[:-1,:-1,:-1])
            

            sigma_centres = (self.W_dx1dx2dpT[Wcharge][k,:-1,:-1,:-1] + self.W_dx1dx2dpT[Wcharge][k,1:,1:,1:]) / 2.


            for i in range(num_yW):
                include = np.where(closest==i)
                cs_yW[k,i] = np.sum(sigma_centres[include] * binAreas[include])
        cs_yW /= y_W_binWidth
            
        return cs_yW

            
            
        
    ############# Display #######################################################
    def display_W_dxidpT(self, Wcharge:str, iofx:int, xother_j:int):
        """
        Display a colourmap of the cross-section wrt a momentum fraction and transverse momentum of
        the W boson.
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
            - i:int: Momentum fraction to use {1/2}.
            - xother_j:int: Index of the other momentum fraction to use.
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."
        assert iofx in {1,2}, 'i must be one of {1,2}.'
        assert (xother_j<self.numX) and (xother_j>=0), 'xother_j must be in range 0<=x<numX'
        
        if self.W_dx1dx2dpT[Wcharge] is None:
            self.calculate_W_dx1dx2dpT(Wcharge)
        
        
        fig = plt.figure(figsize=(10,10),tight_layout=True,dpi=200)
        fig.suptitle(f'W{Wcharge}, index(x{iofx%2+1})={xother_j}')
        if iofx==1:
            extent = (self.pTs[0],self.pTs[-1],np.log10(self.x1[0]),np.log10(self.x1[-1]))
            immap = self.W_dx1dx2dpT[Wcharge][:,:,xother_j]
        else:
            extent = (self.pTs[0],self.pTs[-1],self.x2[0],self.x2[-1])
            immap = self.W_dx1dx2dpT[Wcharge][:,xother_j]
            
            
        aspect=(extent[1] - extent[0]) / (extent[3] - extent[2])
        
        _cut = -1 # -8
        immap[:,_cut:] = 0.

        for i in range(2):
            ax = fig.add_subplot(2,2,i+1)
            
            mask = np.asarray(immap[i]==0,dtype=float)
            im = ax.imshow(immap[i],extent=extent,aspect=aspect,origin='lower',interpolation=None,zorder=1)
            ax.imshow(mask,cmap='binary',zorder=2,alpha=mask,aspect=aspect,
                      origin='lower',interpolation=None,extent=extent)
            ax.plot([extent[0],extent[1]],[np.log10(self.x1[_cut]),np.log10(self.x1[_cut])],c='r')
            
            cbar = fig.colorbar(im,ax=ax)
            
            # Create appropiate labels.
            title = ('particle','antiparticle')[i]
            ax.set_title(f'$x_1$={title}\n $x_{iofx%2+1}$={self.x2[xother_j]}',fontsize=self.titleSize)
            ax.set_xlabel('pT',fontsize=self.labelSize)
            ax.set_ylabel(f'log($x_{iofx})$',fontsize=self.labelSize)
            cbar.ax.set_ylabel(r'log differential cross-section / $\log(\frac{d^3\sigma}{dx_1x_2dp_T})$',
                              fontsize=self.labelSize)
            
            immap_centres = (immap[:,1:] + immap[:,:-1]) / 2.

            ax = fig.add_subplot(2,2,i+3)
            ax.scatter(self.pTs, np.sum(immap_centres[i]*(self.x1[1:] - self.x1[:-1])[:,None], axis=0),
                       c='m', marker='.')
            ax.grid()
            
            # Create appropiate labels.
            ax.set_ylabel(r'differential cross-section / $\frac{d\sigma}{dp_T}|_{x_2}$',fontsize=self.labelSize)
            ax.set_xlabel('pT',fontsize=self.labelSize)
            
            ax.xaxis.set_tick_params(labelsize=self.tickSize)
            ax.yaxis.set_tick_params(labelsize=self.tickSize)
        
        plt.savefig(pathtohere / f'plots/W_NLO/W{Wcharge}/W_dxidpT.png', bbox_inches='tight')
        plt.close(fig)
        
        
    def display_W_dyW(self,Wcharge:str):
        """
        Display how the W cross-section varies with W rapidity.
        
        Inputs:
            - Wcharge:str: W boson charge to integrate.
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."
            
        if self.W_dyW[Wcharge] is None:
            self.calculate_W_dyW(Wcharge)
            
        whereValid = (np.sqrt(self.s_hat[:,:,None]) > self.threshold)
        exclusionThreshold = np.min(self.rapidities_W[whereValid])
        
        whereInclude = (self.yW_bins > exclusionThreshold) |  (self.yW_bins < -exclusionThreshold)
            
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()

        ax.scatter(self.yW_bins[whereInclude][1:-1], np.sum(self.W_dyW[Wcharge],axis=0)[whereInclude][1:-1],
                   marker='.',c='b')
        
        ax.grid()
        
        # Create appropiate labels.
        ax.set_xlabel(f'W boson rapidity / $y_W$',fontsize=self.labelSize)
        ax.set_ylabel(r'differential cross-section / $\frac{d\sigma}{dy_W}$ $[pb]$',
                      fontsize=self.labelSize)
        ax.set_title(f'W{Wcharge}',fontsize=self.titleSize)
        
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)
        
        plt.savefig(pathtohere / f'plots/W_NLO/W{Wcharge}/W_dyW.png', bbox_inches='tight')
        plt.close(fig)
        
        
    def display_W_dpT(self, Wcharge:str):
        """
        Display how the W cross-section varies with transverse momentum.
        
        Inputs:
            - Wcharge:str: W boson charge to integrate.
        """
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}."
            
        if self.W_dpT[Wcharge] is None:
            self.calculate_W_dpT(Wcharge)
            
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()
        
        ax.scatter(self.pTs, self.W_dpT[Wcharge],c='brown',marker='.')
        ax.grid()
        
        # Create appropiate labels.
        ax.set_xlabel(f'W{Wcharge} transverse momentum / pT',fontsize=self.labelSize)
        ax.set_ylabel(r'differential cross-section / $\frac{d\sigma}{dp_T}$ $[pb\, GeV^{-1}]$',
                      fontsize=self.labelSize)
        ax.set_title(f'W{Wcharge}',fontsize=self.titleSize)
        
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)
        
#         ax.set_yscale('log')
                
        plt.savefig(pathtohere / f'plots/W_NLO/W{Wcharge}/W_dpT.png', bbox_inches='tight')
        plt.close(fig)
        
        
    
        
        
    ############# Electron rapidity #########################  
    def calculate_W_dye_int(self,Wcharge:str):
        """
        Integrate the W boson contribution as a function of electron rapidity.
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
        """
        
        if self.W_dye[Wcharge] is None:
            print(f'Warning: W_dye[{Wcharge}] not calculated, calculating with default numTheta=100.')
            self.calculate_W_dye(Wcharge,100)
            
        
        # Integrate over rapidity_e.
        self.W_dye_int[Wcharge] = np.sum((self.W_dye[Wcharge][1:-1][:-1] + self.W_dye[Wcharge][1:-1][1:])
                        *(self.rapidities_W_ye[1:-1][1:] - self.rapidities_W_ye[1:-1][:-1]) / 2.)

    
    
    def calculate_W_dye(self,Wcharge:str,numTheta:int, cutType:str=None,
                             lowerCut:float=None,upperCut:float=None, displayContours:bool=False,
                       cutParticle:str='e'):
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
        
        assert cutParticle in {'e','nu'}, "cutParticle must be one of {'e','nu'}."
        
        for rapidityComb_W in ['+','-']:
            rapidities_e_2d,sigma_e,extent,y_star = self._get_ye_properties(Wcharge, numTheta,
                                                                rapidityComb_W)

            if cutParticle=='e':
                whereCutApplied = (np.abs(self.electron_pT)<=lowerCut)
            else:
                whereCutApplied = (np.abs(self.neutrino_pT)<=lowerCut) | (np.abs(self.electron_pT)<=lowerCut)

            sigma_e[whereCutApplied] = 0.


            # Perform integral.
            self.rapidities_W_ye,temp_cs = self._integrateRapidityContours(sigma_e,
                                           rapidities_e_2d,extent,y_star,numTheta)
            if rapidityComb_W=='+':
                self.W_dye[Wcharge] = np.zeros(temp_cs.shape[0])

            self.W_dye[Wcharge] += temp_cs
                    

    def _get_ye_properties(self, Wcharge:str, numTheta:int, rapidityComb_W:str,
                           starParameter:str='rapidity'):
        """
        Get the matrix rapidities and differential cross sections of the rapidity
        convolution (in addition to the rapidity extent and y_star.)
        
        Inputs:
            - Wcharge:str: W boson charge to calculate (+/-).
            - numTheta:int: Number of angle samples of the lepton wrt W boson.
            - rapidityComb_W:str: Whether the W boson is moving 'forward' or 'backwards' (wrt x1).
            - starParameter:str: Parameter of the electron's angle wrt to the W boson ('rapidity'/'pT').
            
        Outputs:
            - rapidities_e:np.ndarray: Matrix of added rapidities orthogonally.
            - sigma_e:np.ndarray: Convoluted differential cross sections.
            - extent:tuple: Extent of the rapidity.
            - y_star:np.ndarray: Rapidities of the lepton angle wrt W boson.
        """
                
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}"
        assert rapidityComb_W in {'+','-'}, "rapidityComb must be one of {'+','-'}"

        whereInvalid = (np.sqrt(self.s_hat[:,:,None]) < self.threshold)

        
        # Lepton angle wrt the W boson.
        thetas_star = np.linspace(0,2*np.pi,numTheta+1)[:-1]
        Wsign = 1. if rapidityComb_W=='+' else -1.
        # Determine sign on y_W.
        momentumDifference = ((self.x1[:,None,None] - self.x2[None,:,None]) * np.sqrt(LHC_s)/2. 
                                + (self.s_hat[:,:,None]-self.WmassSqrd)/(2*np.sqrt(self.s_hat[:,:,None])) 
                              * Wsign * self.costheta_W)
        momentumSign = np.zeros(momentumDifference.shape)
        momentumSign[(momentumDifference<0) & np.logical_not(whereInvalid)] = -1.
        momentumSign[(momentumDifference>0) & np.logical_not(whereInvalid)] = 1.
            
        
        # Lepton transverse momenta.
        self.electron_pT = (self.pTs[None,None,:,None] *(1. + np.cos(thetas_star)[None,None,None,:]/2.) 
                           + (self.s_hat[:,:,None,None] - self.WmassSqrd) / (4*np.sqrt(self.s_hat[:,:,None,None]))
                           *Wsign*self.costheta_W[:,:,:,None]*np.sin(thetas_star)[None,None,None,:])
        self.neutrino_pT = (self.pTs[None,None,:,None] *(1. + np.cos(thetas_star)[None,None,None,:]/2.) 
                           - (self.s_hat[:,:,None,None] - self.WmassSqrd) / (4*np.sqrt(self.s_hat[:,:,None,None]))
                           *Wsign*self.costheta_W[:,:,:,None]*np.sin(thetas_star)[None,None,None,:])
        
        # Electron rapidity in W frame.
        pz_e_dir = (Wsign*self.costheta_W[:,:,:,None] * np.cos(thetas_star)[None,None,None,:]
                    -2*self.pTs[None,None,:,None]*np.sqrt(self.s_hat[:,:,None,None])
                    /(self.s_hat[:,:,None,None] - self.WmassSqrd)*np.sin(thetas_star)[None,None,None,:])
        y_star = 0.5*np.log((1 + pz_e_dir) / (1 - pz_e_dir))
                

            
        rapidities_e = momentumSign[:,:,:,None]*self.rapidities_W[:,:,:,None] + y_star[:,:,:,:]


        
        sigma_e = np.zeros((self.numX,self.numX,self.num_pT,numTheta))
        for k in range(2):
            # Probability distribution as a function of theta.
            if (Wcharge=='+' and k==1) or (Wcharge=='-' and k==0):
                sigma_theta = (1. + np.cos(thetas_star))**2
            else:
                sigma_theta = (1. - np.cos(thetas_star))**2

            # Apply Jacobian dcos*/dy*.
            sigma_y_star = 2*np.abs(np.sin(thetas_star)[None,None,None,:]*((np.sin(thetas_star)[None,None,None,:]
                                 *Wsign*self.costheta_W[:,:,:,None])
                         + np.cos(thetas_star)[None,None,None,:]*
                            2*self.pTs[None,None,:,None]*np.sqrt(self.s_hat[:,:,None,None])
                /(self.s_hat[:,:,None,None] - self.WmassSqrd)) * sigma_theta[None,None,None,:])
            

            sigma_e += self.W_dx1dx2dpT[Wcharge][k][:,:,:,None] * sigma_y_star[:,:,:,:]

        whereInclude = np.where((np.sqrt(self.s_hat[:-1,:-1,None]) > self.threshold) 
                                   & (np.sqrt(self.s_hat[1:,1:,None]) > self.threshold))
        
        whereInclude_star = np.where((np.sqrt(self.s_hat[:-1,:-1,None,None]) > self.threshold) 
                                   & (np.sqrt(self.s_hat[1:,1:,None,None]) > self.threshold))
        
        sigma_e[whereInvalid,:] = 0.
        
        
        extent = ((y_star[whereInclude_star]).min(),(y_star[whereInclude_star]).max(),
                  (momentumSign*self.rapidities_W)[whereInclude].min(),
                  (momentumSign*self.rapidities_W)[whereInclude].max())
        
        return rapidities_e, sigma_e, extent, y_star
    
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
                
        isAllowed = ((np.sqrt(self.s_hat[:-1,:-1,None]) > self.threshold) 
                                   & (np.sqrt(self.s_hat[1:,1:,None]) > self.threshold))
        isAllowed = isAllowed[:,:,1:] & isAllowed[:,:,:-1]
        
        #y_e_binWidth = 0.21 # https://arxiv.org/abs/1612.03016
#         y_e_wanted = np.arange(-2,2,y_e_binWidth)
        num_ye = 20#y_e_wanted.shape[0]
        y_e_wanted = np.linspace(-5,5,num_ye)
        y_e_binWidth = (y_e_wanted[-1] - y_e_wanted[0]) / num_ye
        
        
        # For each rapidities_e find which y_e_wanted it is closest to.
        rapidities_e_centre = (rapidities_e[1:,1:,1:,1:] + rapidities_e[:-1,:-1,:-1,:-1]) / 2
        
        startTime = perf_counter()
        contourDiff = abs(rapidities_e_centre[:,:,:,:,None] - y_e_wanted[None,None,None,None,:])
        
        
        closest = -np.ones(contourDiff.shape[:4],dtype=int)

        startTime = perf_counter()
        # Find the indices of minimum values along the last axis of contourDiff
        closest = np.argmin(contourDiff, axis=4)

        # Assign the minimum indices to the corresponding positions in closest
        closest[np.logical_not(isAllowed),:] = -1
        
                    
        # Determine all 4D bin areas.
        binAreas = np.abs((self.rapidities_W[1:,1:,1:] - self.rapidities_W[:-1,:-1,:-1])[:,:,:,None] 
                    * (y_star[1:,1:,1:,1:] - y_star[:-1,:-1,:-1,:-1])[:,:,:,:])
        
        
        sigma_centres = (sigma_e[:-1,:-1,:-1,:-1] + sigma_e[1:,1:,1:,1:]) / 2.
        
            
        cs_e = np.empty((num_ye),dtype=float)
        for i in range(len(y_e_wanted)):
            include = np.where(closest==i)
            cs_e[i] = np.sum(sigma_centres[include] * binAreas[include])
        cs_e /= y_e_binWidth
            
        return y_e_wanted,cs_e
    
    
        
    def display_W_dye(self,Wcharge:str):
        """
        Display how the differential cross-section varies with electron rapidity wrt the beamline.
        
        Inputs:
            - Wcharge:str: Charge of the W boson to display. (+/-)
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}"
        
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()
        
        ax.scatter(self.rapidities_W_ye[1:-1],self.W_dye[Wcharge][1:-1],marker='.',c='purple')
        ax.grid()
        
        # Create appropiate labels
        ax.set_xlabel(f'electron rapidity / $y_e$',fontsize=self.labelSize)
        ax.set_ylabel(r'differential cross-section $\frac{d\sigma}{dy_e}$ [pb]',fontsize=self.labelSize)
        ax.set_title(f'W{Wcharge}',fontsize=self.titleSize)
        
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)
        
        plt.savefig(pathtohere / f'plots/W_NLO/W{Wcharge}/WNLO_dye.png', bbox_inches='tight')
        plt.close(fig)
        
        
    def calculate_cutRatioDependence_NLO(self, Wcharge:str):
        """
        Calculate the cross-section ratio as the cut on the pt distribution is
        varied.
        
        Inputs:
            - Wcharge:str: Charge of the W boson to display. (+/-)
        
        Outputs:
            - ptCuts:np.ndarray: Cuts applied to the pt distribution.
            - cs_ratios:np.ndarray: Cross-section ratio to the uncut cross-section.
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}"

        
        numSamples = 10
        ptCuts = np.linspace(0,100, numSamples)
        cs_ratios_e = np.empty(numSamples)
        cs_ratios_nu = np.empty(numSamples)
        
        for j in tqdm(range(numSamples)):
            self.calculate_W_dye(Wcharge=Wcharge,numTheta=100,cutType='pt', lowerCut=ptCuts[j], cutParticle='e')
            self.calculate_W_dye_int(Wcharge=Wcharge)
                
            cs_ratios_e[j] = self.W_dye_int[Wcharge]
            
            self.calculate_W_dye(Wcharge=Wcharge,numTheta=100,cutType='pt', lowerCut=ptCuts[j], cutParticle='nu')
            self.calculate_W_dye_int(Wcharge=Wcharge)
                
            cs_ratios_nu[j] = self.W_dye_int[Wcharge]
            
        cs_ratios_e /= cs_ratios_e[0]
        cs_ratios_nu /= cs_ratios_nu[0]

        
        return ptCuts, cs_ratios_e, cs_ratios_nu
    
    def display_cutRatioDependence_NLO(self,ptCuts:np.ndarray,
                                       cs_ratios_e:np.ndarray, cs_ratios_nu:np.ndarray,Wcharge):
        """
        Display how the cross-section ratio varies with pt cut.
        
        Inputs:
            - ptCuts:np.ndarray: Cuts applied to the pt distribution.
            - cs_ratios:np.ndarray: Cross-section ratio to the uncut cross-section.
            - Wcharge:str: Charge of the W boson to display. (+/-)

        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}"
        
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()
        
        ax.scatter(ptCuts, cs_ratios_e,label='electron',c='g',zorder=1)
        ax.scatter(ptCuts, cs_ratios_nu,label='electron and neutrino',c='r',marker='x',zorder=2)

        ax.grid()

        
        # Create appropiate labels.
        ax.set_xlabel(r'$p_T$ cut [GeV]',fontsize=self.labelSize)
        ax.set_ylabel('cross section ratio',fontsize=self.labelSize)
        ax.set_title(f'W{Wcharge}',fontsize=self.titleSize)
        ax.legend(loc='best',fontsize=self.labelSize)
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)
                
        plt.savefig(pathtohere / f'plots/W_NLO/W{Wcharge}/WNLO_cutRatioDependence.png', bbox_inches='tight')
        plt.close(fig)
        
    def display_csratioratio(self,ptCuts:np.ndarray,
                       cs_ratios_e:np.ndarray, cs_ratios_nu:np.ndarray,Wcharge):
        """
        Display how the ratio cross-section ratios for the electron and
        electron+neutrino cuts varies with pt cut.
        
        Inputs:
            - ptCuts:np.ndarray: Cuts applied to the pt distribution.
            - cs_ratios:np.ndarray: Cross-section ratio to the uncut cross-section.
            - Wcharge:str: Charge of the W boson to display. (+/-)
        """
        
        assert Wcharge in {'+','-'}, "Wcharge must be one of {'+','-'}"
        
        fig = plt.figure(figsize=(8,8),dpi=200)
        ax = fig.add_subplot()
        
        ax.scatter(ptCuts, cs_ratios_e/cs_ratios_nu,c='b')

        ax.grid()

        
        # Create appropiate labels.
        ax.set_xlabel('$p_T$ cut [GeV]',fontsize=self.labelSize)
        ax.set_ylabel('ratio of electron to electron+neutrino cut cross section ratios',
                      fontsize=self.labelSize)
        ax.set_title(f'W{Wcharge}',fontsize=self.titleSize)
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)
                
        plt.savefig(pathtohere / f'plots/W_NLO/W{Wcharge}/W_csRR.png', bbox_inches='tight')
        plt.close(fig)