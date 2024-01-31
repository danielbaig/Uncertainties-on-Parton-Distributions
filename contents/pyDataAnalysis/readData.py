import numpy as np

from pathlib import Path
pathtohere = Path.cwd()


def read_parameterFile():
    """
    Read variables from parameter file and extract useful quantities.
    
    Outputs:
        - LHC_s:float: CoM squared energy.
        - m_W:float: Mass of W boson used.
        - m_Z:float: Mass of Z boson used.
        - Qs:np.ndarray: Q^2 values used.
    """
    
    with open(pathtohere / 'data/parameterFile.txt') as f:
        while line:=f.readline():
            thisLine = line.rstrip().split(' ')
            if thisLine[0]=='s':
                LHC_s = float(thisLine[-1])
            elif thisLine[0]=='m_W':
                m_W = float(thisLine[-1])
            elif thisLine[0]=='m_Z':
                m_Z = float(thisLine[-1])
            elif thisLine[0]=='Qmin':
                Qmin = float(thisLine[-1])
            elif thisLine[0]=='Qmax':
                Qmax = float(thisLine[-1])
            elif thisLine[0]=='numQ':
                numQ = int(thisLine[-1])
                
    Qs = np.linspace(Qmin, Qmax, numQ)
    
    return LHC_s,m_W,m_Z,Qs
                


def read_data(file:str, m_W:float, m_Z:float, Qs:np.ndarray):
    """
    Read a given data file.
    
    Inputs:
        - file:str: Filename.
        - m_W:float: Mass of W boson used.
        - m_Z:float: Mass of Z boson used.
        - Qs:np.ndarray: Q^2 values used.
        
    Outputs:
        - xs:np.ndarray: Momentum fractions.
        - data:np.ndarray: PDFs. (Q^2,x,flavour)
        - qs:np.ndarray: Q^2 used.
    """
    
    # Get data from file for x1.
    data_all = np.loadtxt(pathtohere / file, dtype=str)
    whereNan = np.where('-nan(ind)'==data_all)
    data_all[whereNan] = '0.'
    data_all = data_all.astype(float)
    
    # Split into different Q^2
    qs = np.concatenate((np.asarray([m_W,m_Z]), Qs))
    numQ = int(qs.shape[0])
    numX = int(data_all.shape[0]/numQ)
    data = np.empty((numQ,numX,data_all.shape[1]),dtype=float)
    for i in range(numQ):
        data[i,:,:] = data_all[i*numX:(i+1)*numX]
        
    xs = data[:,:,0]

    
    data = data[:,:,1:]
    
    return xs, data, qs
