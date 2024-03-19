# Uncertainties on Parton Distributions
This project was undertaken in the academic year 2023-2024 as part of the MSci Theoretical Physics at University College London. It consists of code related to generating Parton Distribution Functions (PDFs) using code translated from [FORTRAN](https://www.hep.ucl.ac.uk/mmht/code.shtml) and exploring the effect of applying cuts to the transverse momentum of electrons and electron-neutrinos from W boson produced at the Large Hadron Collider at leading order (LO) and next-to-leading order (NLO).

## Operation
The C++ code should be run first to generate the PDFs. This requires the eigen module to be installed.

The Python `main.ipynb` file can then be run to extract the PDFs and perform the below analyses.

## Features
* At LO
  * Calculate the total cross-section contributions of the bosons.
  * Create distributions of the photon differential cross-section with respect to the invariant mass of the produced dilepton pair.
  * Create distributions of the Z boson differential cross-section with respect to the Z boson's rapidity.
  * Create distributions of the W boson differential cross-section with respect to the W boson's rapidity.
  * Create distributions of the electron's differential cross-section with respect to the electron's rapidity when decaying from a W boson.
  * Create visulisations for the contour integrals in the electron differential cross-section calculation.
  * Determine how the $W^{\pm}\rightarrow e^{\pm} \overset{(-)}{\nu}$ cross-section varies with cut on the transverse momentum.
* At NLO
  * Create distributions of the W boson differential cross-section with respect to the W boson's rapidity.
  * Create distributions of the W boson differential cross-section with respect to the W boson's transverse momentum.
  * Create distributions of the electron's differential cross-section with respect to the electron's rapidity when decaying from a W boson.
  * Determine how the $W^{\pm}\rightarrow e^{\pm} \overset{(-)}{\nu}$ cross-section varies with cut on the transverse momenta of the electron and electron+neutrino.

