// uncertaintiesOfPartonDistributions.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <fstream>
#include <initializer_list>
#include <iomanip>



#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <unsupported/Eigen/CXX11/Tensor>

#include "polate.h"
#include "deriv_locx.h"

/*********************************************************************
**
* Program for generating Electromagnetic Structure Functions using*
* consistent treatment of charmand bottom structure functions*
* Not included are the effects due to NLO corrections to photon - *
*gluon fusion.Charm mass = 1.4 GeV   Bottom mass = 4.75 GeV *
**
*The program should be run only with iord set to 1 *
*The calculation of F_L includes the full order(alpha_s ^ 2) *
*contribution *
*The program is self contained, only requiring the subroutine *
*mrst2002.f and the grid file mrst2002nlo.dat.dat to be accessible *
**
***********************************************************************/





void InitialisePDF(int ip, int np, int ih, int nhess, int nx, int my, int myc0, int myb0,
    Eigen::VectorXd xx, Eigen::VectorXd yy, Eigen::Tensor<double, 3> ff, Eigen::Tensor<double, 6>* cc)
{
    //integer nhess, ih, nx, my, myc0, myb0, j, k, l, m, n, ip, np
    /*
    double precision xx(nx), yy(my), ff(np, nx, my),
        & ff1(nx, my), ff2(nx, my), ff12(nx, my), ff21(nx, my),
        & yy0(4), yy1(4), yy2(4), yy12(4), z(16),
        & cl(16), cc(np, 0:nhess, nx, my, 4, 4), iwt(16, 16),
        & polderiv1, polderiv2, polderiv3, d1, d2, d1d2, xxd
        */

    Eigen::MatrixXd ff1(nx, my);
    Eigen::MatrixXd ff2(nx, my);
    Eigen::MatrixXd ff12(nx, my);
    Eigen::MatrixXd ff21(nx, my);
    Eigen::VectorXd yy0(4);
    Eigen::VectorXd yy1(4);
    Eigen::VectorXd yy2(4);
    Eigen::VectorXd yy12(4);
    Eigen::VectorXd z(16);
    Eigen::VectorXd cl(16);
    Eigen::MatrixXd iwt(16, 16);

    double d1{};
    double d2{};
    double d1d2{};
    double xxd{};
    


    iwt << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0,
        2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1,
        0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1,
        -3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0,
        9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2,
        -6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2,
        2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0,
        -6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1,
        4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1;



    for (int m{ 1 }; m <= my; ++m)
    {
        ff1(0, m-1) = polderiv1(xx(0), xx(1), xx(2),
            ff(ip-1, 0, m-1), ff(ip-1, 1, m-1), ff(ip-1, 2, m-1));
        ff1(nx-1, m-1) = polderiv3(xx(nx - 3), xx(nx - 2), xx(nx-1),
            ff(ip-1, nx - 3, m-1), ff(ip-1, nx - 2, m-1), ff(ip-1, nx-1, m-1));
        for (int n{ 2 }; n <= nx - 1; ++n)
        {
            ff1(n-1, m-1) = polderiv2(xx(n - 2), xx(n-1), xx(n),
                ff(ip-1, n - 2, m-1), ff(ip-1, n-1, m-1), ff(ip-1, n, m-1));
        }
    }

    //   Calculate the derivatives at qsq = mc2, mc2 + eps, mb2, mb2 + eps
    //   in a similar way as at the endpoints qsqmin and qsqmax.
    for (int n{ 1 }; n <= nx; ++n)
    {
        for (int m{ 1 }; m <= my; ++m)
        {
            if (m == 1 || m == myc0 + 1 || m == myb0 + 1)
            {
                ff2(n-1, m-1) = polderiv1(yy(m-1), yy(m), yy(m + 1),
                    ff(ip-1, n-1, m-1), ff(ip-1, n-1, m), ff(ip-1, n-1, m + 1));
            }
            else if (m==my || m==myc0 || m==myb0)
            {
                ff2(n-1, m-1) = polderiv3(yy(m - 3), yy(m - 2), yy(m-1),
                    ff(ip-1, n-1, m - 3), ff(ip-1, n-1, m - 2), ff(ip-1, n-1, m-1));
            }
            else
            {
                ff2(n-1, m-1) = polderiv2(yy(m - 2), yy(m-1), yy(m),
                    ff(ip-1, n-1, m - 2), ff(ip-1, n-1, m-1), ff(ip-1, n-1, m));
            }

        }
    }

    //   Calculate the cross derivatives(d / dx)(d / dy).
    for (int m{ 1 }; m <= my; ++m)
    {
        ff12(0, m-1) = polderiv1(xx(0), xx(1), xx(2),
            ff2(0, m-1), ff2(1, m-1), ff2(2, m-1));


        ff12(nx-1, m-1) = polderiv3(xx(nx - 3), xx(nx - 2), xx(nx-1),
            ff2(nx - 3, m-1), ff2(nx - 2, m-1), ff2(nx-1, m-1));

        for (int n{ 2 }; n <= nx - 1; ++n)
        {
            ff12(n-1, m-1) = polderiv2(xx(n - 2), xx(n-1), xx(n),
                ff2(n - 2, m-1), ff2(n-1, m-1), ff2(n, m-1));
        }
    }

    //   Calculate the cross derivatives(d / dy)(d / dx).
    for (int n{ 1 }; n <= nx; ++n)
    {
        for (int m{ 1 }; m <= my; ++m)
        {
            if (m == 1 || m == myc0 + 1 || m == myb0 + 1)
            {
                ff21(n-1, m-1) = polderiv1(yy(m-1), yy(m), yy(m + 1),
                    ff1(n-1, m-1), ff1(n-1, m), ff1(n-1, m + 1));
            }
            else if (m == my || m == myc0 || m == myb0)
            {
                ff21(n-1, m-1) = polderiv3(yy(m - 3), yy(m - 2), yy(m-1),
                    ff1(n-1, m - 3), ff1(n-1, m - 2), ff1(n-1, m-1));
            }
            else
            {
                ff21(n-1, m-1) = polderiv2(yy(m - 2), yy(m-1), yy(m),
                    ff1(n-1, m - 2), ff1(n-1, m-1), ff1(n-1, m));
            }
        }
    }

    //   Take the average of(d / dx)(d / dy) and (d / dy)(d / dx).
    for (int n{ 1 }; n <= nx; ++n)
    {
        for (int m{ 1 }; m <= my; ++m)
        {
            ff12(n-1, m-1) = 0.5 * (ff12(n-1, m-1) + ff21(n-1, m-1));
        }
    }

    for (int n{ 1 }; n <= nx - 1; ++n)
    {
        for (int m{ 1 }; m <= my - 1; ++m)
        {
            d1 = xx(n) - xx(n-1);
            d2 = yy(m) - yy(m-1);
            d1d2 = d1 * d2;

            yy0(0) = ff(ip-1, n-1, m-1);
            yy0(1) = ff(ip-1, n, m-1);
            yy0(2) = ff(ip-1, n, m);
            yy0(3) = ff(ip-1, n-1, m);

            yy1(0) = ff1(n-1, m-1);
            yy1(1) = ff1(n, m-1);
            yy1(2) = ff1(n, m);
            yy1(3) = ff1(n-1, m);

            yy2(0) = ff2(n-1, m-1);
            yy2(1) = ff2(n, m-1);
            yy2(2) = ff2(n, m);
            yy2(3) = ff2(n-1, m);


            yy12(0) = ff12(n-1, m-1);
            yy12(1) = ff12(n, m-1);
            yy12(2) = ff12(n, m);
            yy12(3) = ff12(n-1, m);

            for (int k{ 1 }; k <= 4; ++k)
            {
                z(k-1) = yy0(k-1);
                z(k + 3) = yy1(k-1) * d1;
                z(k + 7) = yy2(k-1) * d2;
                z(k + 11) = yy12(k-1) * d1d2;
            }

            for (int l{ 1 }; l <= 16; ++l)
            {

                xxd = 0.;
                for (int k{ 1 }; k <= 16; ++k)
                {
                    xxd = xxd + iwt(l-1, k-1) * z(k-1);
                    //std::cout << "iwt: " << k
                      //  << "," << l << " "
                        //<< iwt(l-1, k-1) << '-' 
                        //<< z(k-1)
                        //<< '\n';
                }
                cl(l-1) = xxd;

                //std::cout << cl(l - 1) << '\n';
            }
            int l{ 0 };
            for (int k{ 1 }; k <= 4; ++k)
            {
                for (int j{ 1 }; j <= 4; ++j)
                {
                    ++l;
                    
                    (*cc)(ip-1, ih, n-1, m-1, k-1, j-1) = cl(l-1);
                    //if (cl(l-1) > 0.1) std::cout << cc(ip - 1, ih, n - 1, m - 1, k - 1, j - 1) << '\n';
                }
            }
            if (ih == -1)
            {
                std::cout << "cc_i " << n << ',' 
                    << m << ": " << (*cc)(9 - 1, ih, n - 1, m - 1, 0, 3) << '\n';
            }
        }
    }


    return;
}






int nx{ 64 };
int nq{ 48 };
int np{ 12 };
int nhess{ 2 * 32 };

std::vector<std::string> oldprefix(nhess+1);
int nqc0{};
int nqb0{};

Eigen::Tensor<double, 3> ff(np, nx, nq);
Eigen::Tensor<double, 6> cc(np, nhess + 1, nx, nq, 4, 4);

Eigen::VectorXd xx(nx);
Eigen::VectorXd qq(nq);

Eigen::VectorXd xxl(nx);
Eigen::VectorXd qql(nq);


double GetOnePDF32(std::string prefix, int ih, double x, double q, int f)
{
    bool warn{ true };
    bool fatal{ true };
    // Set warn = .true.to turn on warnings when extrapolating.
    // Set fatal = .false.to return zero instead of terminating when
    // invalid input values of x and q are used.

    double xmin{ 1e-6 };
    double xmax{ 1. };
    double qsqmin{ 1. };
    double qsqmax{ 1e+9 };
    double eps{ 1e-6 };





    std::string set{};
    std::string filename{};

    char dummyChar{};
    std::string dummyWord{};

    

    xx << 1e-6, 2e-6, 4e-6, 6e-6, 8e-6,
        1e-5, 2e-5, 4e-5, 6e-5, 8e-5,
        1e-4, 2e-4, 4e-4, 6e-4, 8e-4,
        1e-3, 2e-3, 4e-3, 6e-3, 8e-3,
        1e-2, 1.4e-2, 2e-2, 3e-2, 4e-2, 6e-2, 8e-2,
        0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275,
        0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475,
        0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675,
        0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875,
        0.9, 0.925, 0.95, 0.975, 1.;

    qq << 1.,
        1.25, 1.5, 0., 0., 2.5, 3.2, 4., 5., 6.4, 8.,
        1e1, 1.2e1, 0., 0., 2.6e1, 4e1, 6.4e1, 1e2,
        1.6e2, 2.4e2, 4e2, 6.4e2, 1e3, 1.8e3, 3.2e3, 5.6e3, 1e4,
        1.8e4, 3.2e4, 5.6e4, 1e5, 1.8e5, 3.2e5, 5.6e5, 1e6,
        1.8e6, 3.2e6, 5.6e6, 1e7, 1.8e7, 3.2e7, 5.6e7, 1e8,
        1.8e8, 3.2e8, 5.6e8, 1e9;
    // Store heavy quark masses and alphaS parameters in COMMON block.
    //common / mcmbas / mCharm, mBottom, alphaSQ0, alphaSMZ, alphaSorder,
    //& alphaSnfmax

    if (f < -6. || f>13)
    {
        std::cout << "Error: invalid parton flavour = " << f << '\n';
        return -1;
    }

    if (ih<0. || ih>nhess)
    {
        std::cout << "Error: invalid eigenvector number = " << ih << '\n';
        return -1;
    }


    // Check if the requested parton set is already in memory.
    if (oldprefix[ih] != prefix)
    {

        //   Start of initialisation for eigenvector set "i" ...
        //  Do this only the first time the set "i" is called,
        //   OR if the prefix has changed from the last time.




        set = std::to_string(ih);  //convert integer to string
        if (ih < 10)
        {
            set = "0" + set;
        }
        //  Remove trailing blanks from prefix before assigning filename.
        std::string filename = prefix + '.' + set + ".dat";
        // Line below can be commented out if you don't want this message.
        std::cout << "Reading PDF grid " << set << " from " << filename << '\n';

        std::ifstream file;

        int io{};
        double distance{}, tolerance{}, mCharm{}, mBottom{}, alphaSQ0{}, alphaSMZ{};
        int alphaSorder{}, alphaSnfmax{}, nExtraFlavours{};

        // Open the file
        file.open(filename.c_str(), std::ios::in);

        // Check if the file is opened successfully
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << '\n';
            return -1;
        }

        // Skip the first two lines
        for (int i{ 0 }; i < 2; ++i)
        {
            getline(file, dummyWord);
        }

        // Read header.
        for (int i{ 0 }; i < 4; ++i)
        {
            file >> dummyWord;
        }
        distance = std::stod(dummyWord);
        file >> dummyWord;
        tolerance = std::stod(dummyWord);
        for (int i{ 0 }; i < 4; ++i)
        {
            file >> dummyWord;
        }
        mCharm = std::stod(dummyWord);
        for (int i{ 0 }; i < 4; ++i)
        {
            file >> dummyWord;
        }
        mBottom = std::stod(dummyWord);
        for (int i{ 0 }; i < 4; ++i)
        {
            file >> dummyWord;
        }
        alphaSQ0 = std::stod(dummyWord);
        for (int i{ 0 }; i < 4; ++i)
        {
            file >> dummyWord;
        }
        alphaSMZ = std::stod(dummyWord);
        for (int i{ 0 }; i < 4; ++i)
        {
            file >> dummyWord;
        }
        dummyWord.pop_back();
        alphaSorder = std::stoi(dummyWord);
        file >> dummyWord;
        alphaSnfmax = std::stoi(dummyWord);
        for (int i{ 0 }; i < 4; ++i)
        {
            file >> dummyWord;
        }
        nExtraFlavours = std::stoi(dummyWord);
        for (int i{ 0 }; i < 3; ++i)
        {
            getline(file, dummyWord);
        }




        double mc2{ mCharm * mCharm };
        double mb2{ mBottom * mBottom };



        // Check that the heavy quark masses are sensible.
        // Redistribute grid points if not in usual range.
        if (mc2 <= qq(0) || mc2 + eps >= qq(7))
        {
            std::cout << "Error in GetOnePDF: invalid mCharm = " << mCharm << '\n';
            return -1;
        }
        else if (mc2 < qq(1))
        {
            nqc0 = 2;
            qq(3) = qq(1);
            qq(4) = qq(2);
        }
        else if (mc2 < qq(2))
        {
            nqc0 = 3;
            qq(4) = qq(2);
        }
        else if (mc2 < qq(5))
        {
            nqc0 = 4;
        }
        else if (mc2 < qq(6))
        {
            nqc0 = 5;
            qq(3) = qq(5);
        }
        else
        {
            nqc0 = 6;
            qq(3) = qq(5);
            qq(4) = qq(6);
        }
        if (mb2 <= qq(11) || mb2 + eps >= qq(16))
        {
            std::cout << "Error in GetOnePDF: invalid mBottom = " << mBottom << '\n';
            return -1;
        }
        else if (mb2 < qq(12))
        {
            nqb0 = 13;
            qq(14) = qq(12);
        }
        else if (mb2 < qq(15))
        {
            nqb0 = 14;
        }
        else
        {
            nqb0 = 15;
            qq(13) = qq(15);
        }
        qq(nqc0 - 1) = mc2;
        qq(nqc0) = mc2 + eps;
        qq(nqb0 - 1) = mb2;
        qq(nqb0) = mb2 + eps;


        // The nExtraFlavours variable is provided to aid compatibility
        //   with future grids where, for example, a photon distribution
        //   might be provided(cf.the MRST2004QED PDFs).
        if (nExtraFlavours < 0 || nExtraFlavours>1)
        {
            std::cout << "Error in GetOnePDF: invalid nExtraFlavours = " << nExtraFlavours << '\n';
            return -1;
        }



        for (int n{ 1 }; n <= nx - 1; ++n)
        {
            for (int m{ 1 }; m <= nq; ++m)
            {
                if (nExtraFlavours > 0)
                {
                    if (alphaSorder == 2)
                    {
                        // NNLO
                        for (int ip{ 1 }; ip <= 12; ++ip)
                        {
                            file >> dummyWord;
                            ff(ip - 1, n - 1, m - 1) = std::stod(dummyWord);
                        }
                    }
                    else
                    {
                        // LO or NLO
                        ff(9, n - 1, m - 1) = 0.0; // chm-cbar
                        ff(10, n - 1, m - 1) = 0.0; // bot-bbar
                        for (int ip{ 1 }; ip <= 9; ++ip)
                        {
                            file >> dummyWord;
                            ff(ip - 1, n - 1, m - 1) = std::stod(dummyWord);
                        }
                        file >> dummyWord;
                        ff(11, n - 1, m - 1) = std::stod(dummyWord);
                    }
                }
                else
                {
                    // nExtraFlavours = 0
                    if (alphaSorder == 2)
                    {
                        // NNLO
                        ff(11, n - 1, m - 1) = 0.0; // photon
                        for (int ip{ 1 }; ip <= 11; ++ip)
                        {
                            file >> dummyWord;
                            ff(ip - 1, n - 1, m - 1) = std::stod(dummyWord);
                        }
                    }
                    else
                    {
                        // LO or NLO
                        ff(9, n - 1, m - 1) = 0.0; // chm-cbar
                        ff(10, n - 1, m - 1) = 0.0; // bot-bbar
                        ff(11, n - 1, m - 1) = 0.0; // photon
                        for (int ip{ 1 }; ip <= 9; ++ip)
                        {
                            file >> dummyWord;
                            ff(ip - 1, n - 1, m - 1) = std::stod(dummyWord);
                        }
                    }
                }

                if (!file.is_open()) {
                    std::cerr << "Error in GetOnePDF reading " << filename << '\n';
                    return -1;
                }
                // std::cout << n << " " << m << " ff " << ff(0, n - 1, m - 1) << '\n';
            }
        }

        // Check that ALL the file contents have been read in.
        file >> dummyWord;
        if (file)
        {
            std::cerr << "Error in GetOnePDF: not at the end of " << filename << '\n';
            return -1;
        }

        file.close();

        // PDFs are identically zero at x = 1.
        for (int m{ 1 }; m <= nq; ++m)
        {
            for (int ip{ 1 }; ip <= np; ++ip)
            {
                ff(ip - 1, nx - 1, m - 1) = 0.;
            }
        }


        for (int n{ 1 }; n <= nx; ++n)
        {
            xxl(n - 1) = log10(xx(n - 1));
            //std::cout << xxl(n - 1) << ',';
        }
        //std::cout << '\n' << x << "x\n";

        for (int m{ 1 }; m <= nq; ++m)
        {
            qql(m - 1) = log10(qq(m - 1));
            //std::cout << qql(m - 1) << ',';
        }
        //std::cout << '\n';


        //   Initialise all parton flavours.
        for (int ip{ 1 }; ip <= np; ++ip)
        {
            InitialisePDF(ip, np, ih, nhess, nx, nq, nqc0, nqb0,
                xxl, qql, ff, &cc);
        }

        oldprefix[ih] = prefix;

        //   ... End of initialisation for eigenvector set "ih".
    }

    //if (ih == 64 && f==9 && x==0.199)
    //{
      //  std::cout << "xx: " << xxl(28 - 1) << ' ' << xx(28 - 1) << '\n';
      //  for (int blip{ 1 }; blip <= nq; ++blip)
        //{
            //std::cout << qql(blip-1) << '\n';
        //}
    //}
    

    double qsq{ q * q };
    //  If mc2 < qsq < mc2 + eps, then qsq = mc2 + eps.
    if (qsq > qq(nqc0 - 1) && qsq < qq(nqc0)) qsq = qq(nqc0);
    //   If mb2 < qsq < mb2 + eps, then qsq = mb2 + eps.
    if (qsq > qq(nqb0 - 1) && qsq < qq(nqb0)) qsq = qq(nqb0);

    double xlog{ log10(x) };
    double qsqlog{ log10(qsq) };

    double res{ 0. };
    double res1{ 0. };
    int ip{};


    if (f == 0)          //gluon
    {
        ip = 1;
    }
    else if (f >= 1 && f <= 5) //quarks
    {
        ip = f + 1;
    }
    else if (f <= -1 && f >= -5)  //antiquarks
    {
        ip = -f + 1;
    }
    else if (f >= 7 && f <= 11) //valence quarks
    {
        ip = f;
    }
    else if (f == 13)    //photon
    {
        ip = 12;
    }
    else if (abs(f) != 6 && f != 12)
    {
        if (warn || fatal) std::cout << "Error in GetOnePDF: f = " << f << '\n';
        if (fatal) return -1;
    }

    // std::cout << "cc(): " << cc(ip - 1, ih - 1, 5, 0, 3, 0) << '\n';


    if (x <= 0. || x > xmax || q <= 0.)
    {
        if (warn || fatal) std::cout << "Error in GetOnePDF: x,qsq = " << x << ',' << qsq << '\n';
        if (fatal) return -1;
    }

    else if (qsq < qsqmin) //extrapolate to low Q ^ 2
    {
        if (warn)
        {
            std::cout << "Warning in GetOnePDF, extrapolating: f = "
                << f << ", x = " << x << ", q = " << q << '\n';
        }

        if (x < xmin)    //extrapolate to low x
        {
            res = ExtrapolatePDF(ip, np, ih, nhess, xlog,
                log10(qsqmin), nx, nq, &xxl, &qql, &cc);
            res1 = ExtrapolatePDF(ip, np, ih, nhess, xlog,
                log10(1.01 * qsqmin), nx, nq, &xxl, &qql, &cc);

            if (f <= -1 && f >= -5)  //antiquark = quark - valence
            {
                res = res - ExtrapolatePDF(ip + 5, np, ih, nhess, xlog,
                    log10(qsqmin), nx, nq, &xxl, &qql, &cc);
                res1 = res1 - ExtrapolatePDF(ip + 5, np, ih, nhess, xlog,
                    log10(1.01 * qsqmin), nx, nq, &xxl, &qql, &cc);
            }
        }


        else                   //do usual interpolation
        {
            res = InterpolatePDF(ip, np, ih, nhess, xlog,
                log10(qsqmin), nx, nq, &xxl, &qql, &cc);
            res1 = InterpolatePDF(ip, np, ih, nhess, xlog,
                log10(1.01 * qsqmin), nx, nq, &xxl, &qql, &cc);
            if (f <= -1 && f >= -5)  //antiquark = quark - valence
            {
                res = res - InterpolatePDF(ip + 5, np, ih, nhess, xlog,
                    log10(qsqmin), nx, nq, &xxl, &qql, &cc);
                res1 = res1 - InterpolatePDF(ip + 5, np, ih, nhess, xlog,
                    log10(1.01 * qsqmin), nx, nq, &xxl, &qql, &cc);
            }
        }
    
        /*   Calculate the anomalous dimension, dlog(xf) / dlog(qsq),
            C--   evaluated at qsqmin.Then extrapolate the PDFs to low
            C--   qsq < qsqmin by interpolating the anomalous dimenion between
            C--   the value at qsqminand a value of 1 for qsq << qsqmin.
            C--   If value of PDF at qsqmin is very small, just set
            C--   anomalous dimension to 1 to prevent rounding errors.
        */
        double anom{};

        if (abs(res) >= 1e-5)
        {
            anom = (res1 - res) / res / 0.01;
        }
        else
        {
            anom = 1.;
        }
        res = std::pow(res * (qsq / qsqmin), (anom * qsq / qsqmin + 1. - qsq / qsqmin));
    }
    else if (x<xmin || qsq > qsqmax)  //extrapolate
    {
        if (warn)
        {
            std::cout << "Warning in GetOnePDF, extrapolating: f = "
                << f << ", x = " << x << ", q = " << q << '\n';
        }

        res = ExtrapolatePDF(ip, np, ih, nhess, xlog,
            qsqlog, nx, nq, &xxl, &qql, &cc);

        if (f <= -1 && f >= -5)  //antiquark = quark - valence
        {
            res = res - ExtrapolatePDF(ip + 5, np, ih, nhess, xlog,
                qsqlog, nx, nq, &xxl, &qql, &cc);

        }
    }
    else if (abs(f) != 6 && f != 12)  //do usual interpolation
    {

        res = InterpolatePDF(ip, np, ih, nhess, xlog,
            qsqlog, nx, nq, &xxl, &qql, &cc);



        if (f <= -1 && f >= -5)  //antiquark = quark - valence
        {
            res = res - InterpolatePDF(ip + 5, np, ih, nhess, xlog,
                qsqlog, nx, nq, &xxl, &qql, &cc);
        }
    }

    // if (f==9) std::cout << f << " " << res << '\n';


    //if (x == 0.199 && f==9)
    //{
      //  std::cout << f << " res " << res << '\n';
    //}
    

    return res;

}


void GetAllPDFs32(std::string prefix, int ih, double x, double q,
    double* upv, double* dnv, double* usea, double* dsea, double* str,
    double* sbar, double* chm, double* cbar, double* bot, double* bbar,
    double* glu, double* phot)
{

    //   Quarks.
    double dn{ GetOnePDF32(prefix, ih, x, q, 1) };
    double up{ GetOnePDF32(prefix, ih, x, q, 2) };
    *str = GetOnePDF32(prefix, ih, x, q, 3) ;
    *chm = GetOnePDF32(prefix, ih, x, q, 4) ;
    *bot = GetOnePDF32(prefix, ih, x, q, 5) ;

    // Valence quarks.
    *dnv = GetOnePDF32(prefix, ih, x, q, 7);
    *upv = GetOnePDF32(prefix, ih, x, q, 8);
    double sv{ GetOnePDF32(prefix, ih, x, q, 9) };
    double cv{ GetOnePDF32(prefix, ih, x, q, 10) };
    double bv{ GetOnePDF32(prefix, ih, x, q, 11) };

    //if (x == 0.199)
    //{
      //  std::cout << *str << " " << sv << "\n\n";
    //}


    // Antiquarks = quarks - valence quarks.
    *dsea = dn - *dnv;
    *usea = up - *upv;
    *sbar = *str - sv;
    *cbar = *chm - cv;
    *bbar = *bot - bv;

    // std::cout << *dsea << ' ' << *usea << ' ' << *sbar << '\n';

    // Gluon.
    *glu = GetOnePDF32(prefix, ih, x, q, 0);

    // Photon(= zero unless considering QED contributions).
    *phot = GetOnePDF32(prefix, ih, x, q, 13);

    return;
}


int main()
{
    std::ofstream file("../data/partonfrac.out");

    int q2{};
    double upv7{};
    double dnv7{};
    double usea7{};
    double dsea7{};
    double str7{};
    double sbar7{};
    double chm7{};
    double cbar7{};
    double bot7{};
    double bbar7{};
    double glu7{};
    double phot7{};

    std::string prefix{};

    double sea7, splus7, sminus7, seam7;

    double ERRSUMg1{};
    double ERRSUMg2{};
    double ERRSUMupv1{};
    double ERRSUMupv2{};
    double ERRSUMdnv1{};
    double ERRSUMdnv2{};
    double ERRSUMsea1{};
    double ERRSUMsea2{};
    double ERRSUMseam1{};
    double ERRSUMseam2{};
    double ERRSUMsplus1{};
    double ERRSUMsplus2{};
    double ERRSUMsminus1{};
    double ERRSUMsminus2{};

    int iset1{};
    int iset2{};

    double upv8{}, dnv8{}, usea8{}, dsea8{}, str8{},
        sbar8{}, chm8{}, cbar8{}, bot8{}, bbar8{}, glu8{}, phot8{};
    double sea8{}, splus8{}, sminus8{}, seam8{};
    double upv9{}, dnv9{}, usea9{}, dsea9{}, str9{},
        sbar9{}, chm9{}, cbar9{}, bot9{}, bbar9{}, glu9{}, phot9{};
    double sea9{}, splus9{}, sminus9{}, seam9{};


    double errg1, errg2, errupv1, errupv2, errdnv1, errdnv2,
        errsea1, errsea2, errseam1, errseam2,
        errsplus1, errsplus2, errsminus1, errsminus2;

    double ERRTOTg1, ERRTOTg2, ERRPCg1, ERRPCg2, ERRTOTupv1, ERRTOTupv2,
        ERRPCupv1, ERRPCupv2, ERRTOTdnv1, ERRTOTdnv2, ERRPCdnv1, ERRPCdnv2,
        ERRTOTsea1, ERRTOTsea2, ERRPCsea1, ERRPCsea2, ERRTOTseam1,
        ERRTOTseam2, ERRPCseam1, ERRPCseam2, ERRTOTsplus1, ERRTOTsplus2,
        ERRPCsplus1, ERRPCsplus2, ERRTOTsminus1, ERRTOTsminus2,
        ERRPCsminus1, ERRPCsminus2;

    double ifix{};

    double upv10{}, dnv10{}, usea10{}, dsea10{}, str10{},
        sbar10{}, chm10{}, cbar10{}, bot10{}, bbar10{}, glu10{}, phot10{};
    double sea10{}, splus10{}, sminus10{}, seam10{};
    double upv11{}, dnv11{}, usea11{}, dsea11{}, str11{},
        sbar11{}, chm11{}, cbar11{}, bot11{}, bbar11{}, glu11{}, phot11{};
    double sea11{}, splus11{}, sminus11{}, seam11{};


    std::vector<double> xczbin{ 0.00001,
        0.00003, 0.00007, 0.00018, 0.00035, 0.0006,
        0.001, 0.0015, 0.003, 0.006, 0.012, 0.02, 0.030,
        0.050, 0.080, 0.10, 0.199, 0.3, 0.4, 0.5, 0.7,
        0.85, 0.90, 0.95, 0.98 };


    double q{};
    for (int nq{ 1 }; nq <= 6; ++nq)
    {
        if (nq == 1) q2 = 1;
        else if (nq == 2) q2 = 2;
        else if (nq == 3) q2 = 12;
        else if (nq == 4) q2 = 60;
        else if (nq == 5) q2 = 200;
        else if (nq == 6) q2 = 650;
        q = sqrt(q2);

        int iset{ 0 };

        

        for (int nxch{ 0 }; nxch < 22; ++nxch)
        {
            double x{ xczbin[nxch] };
            // Prefix for the grid files.
            prefix = "../data/msht/msht/msht20nnlo_as118_internal";
            GetAllPDFs32(prefix, iset, x, q, &upv7, &dnv7, &usea7, &dsea7, &str7,
                &sbar7, &chm7, &cbar7, &bot7, &bbar7, &glu7, &phot7);
            sea7 = 2. * usea7 + 2. * dsea7 + str7 + sbar7;
            splus7 = str7 + sbar7;
            sminus7 = str7 - sbar7;
            seam7 = -usea7 + dsea7;

            ERRSUMg1 = 0.;
            ERRSUMg2 = 0.;
            ERRSUMupv1 = 0.;
            ERRSUMupv2 = 0.;
            ERRSUMdnv1 = 0.;
            ERRSUMdnv2 = 0.;
            ERRSUMsea1 = 0.;
            ERRSUMsea2 = 0.;
            ERRSUMseam1 = 0.;
            ERRSUMseam2 = 0.;
            ERRSUMsplus1 = 0.;
            ERRSUMsplus2 = 0.;
            ERRSUMsminus1 = 0.;
            ERRSUMsminus2 = 0.;

            for (int nmode{ 1 }; nmode <= 32; ++nmode)
            {
                //if (x==0.199) std::cout << nmode << '\n';

                prefix = "../data/msht/msht/msht20nnlo_as118_internal";  // prefix for the grid files
                iset1 = 2 * nmode - 1;
                iset2 = 2 * nmode;
                GetAllPDFs32(prefix, iset1, x, q, &upv8, &dnv8, &usea8, &dsea8, &str8,
                    &sbar8, &chm8, &cbar8, &bot8, &bbar8, &glu8, &phot8);

                //std::cout << upv7 << " " << upv8 << '\n';
                sea8 = 2. * usea8 + 2. * dsea8 + str8 + sbar8;
                seam8 = -usea8 + dsea8;
                splus8 = str8 + sbar8;
                sminus8 = str8 - sbar8;
                GetAllPDFs32(prefix, iset2, x, q, &upv9, &dnv9, &usea9, &dsea9, &str9,
                    &sbar9, &chm9, &cbar9, &bot9, &bbar9, &glu9, &phot9);
                sea9 = 2. * usea9 + 2. * dsea9 + str9 + sbar9;
                splus9 = str9 + sbar9;
                sminus9 = str9 - sbar9;
                seam9 = -usea9 + dsea9;

                errg1 = max(max(glu8 - glu7, glu9 - glu7), 0.);
                errg2 = max(max(glu7 - glu8, glu7 - glu9), 0.);
                ERRSUMg1 = ERRSUMg1 + errg1 * errg1;
                ERRSUMg2 = ERRSUMg2 + errg2 * errg2;
                errupv1 = max(max(upv8 - upv7, upv9 - upv7), 0.);
                errupv2 = max(max(upv7 - upv8, upv7 - upv9), 0.);
                //std::cout << errupv1 << " " << errupv2 << '\n';

                ERRSUMupv1 = ERRSUMupv1 + errupv1 * errupv1;
                ERRSUMupv2 = ERRSUMupv2 + errupv2 * errupv2;
                errdnv1 = max(max(dnv8 - dnv7, dnv9 - dnv7), 0.);
                errdnv2 = max(max(dnv7 - dnv8, dnv7 - dnv9), 0.);
                ERRSUMdnv1 = ERRSUMdnv1 + errdnv1 * errdnv1;
                ERRSUMdnv2 = ERRSUMdnv2 + errdnv2 * errdnv2;
                errsea1 = max(max(sea8 - sea7, sea9 - sea7), 0.);
                errsea2 = max(max(sea7 - sea8, sea7 - sea9), 0.);
                ERRSUMsea1 = ERRSUMsea1 + errsea1 * errsea1;
                ERRSUMsea2 = ERRSUMsea2 + errsea2 * errsea2;
                errseam1 = max(max(seam8 - seam7, seam9 - seam7), 0.);
                errseam2 = max(max(seam7 - seam8, seam7 - seam9), 0.);
                ERRSUMseam1 = ERRSUMseam1 + errseam1 * errseam1;
                ERRSUMseam2 = ERRSUMseam2 + errseam2 * errseam2;
                errsplus1 = max(max(splus8 - splus7, splus9 - splus7), 0.);
                errsplus2 = max(max( splus7 - splus8, splus7 - splus9), 0. );
                // std::cout << "diff: " << errsplus2 << '\n';
                ERRSUMsplus1 = ERRSUMsplus1 + errsplus1 * errsplus1;
                ERRSUMsplus2 = ERRSUMsplus2 + errsplus2 * errsplus2;
                errsminus1 = max(max(sminus8 - sminus7, sminus9 - sminus7), 0.);
                errsminus2 = max(max(sminus7 - sminus8, sminus7 - sminus9), 0.);
                ERRSUMsminus1 = ERRSUMsminus1 + errsminus1 * errsminus1;
                ERRSUMsminus2 = ERRSUMsminus2 + errsminus2 * errsminus2;
            }
            ERRTOTg1 = sqrt(ERRSUMg1);
            ERRTOTg2 = sqrt(ERRSUMg2);
            ERRPCg1 = ERRTOTg1 / (glu7) * 100.;
            ERRPCg2 = ERRTOTg2 / (glu7) * 100.;
            ERRTOTupv1 = sqrt(ERRSUMupv1);
            ERRTOTupv2 = sqrt(ERRSUMupv2);
            ERRPCupv1 = ERRTOTupv1 / (upv7) * 100.;
            ERRPCupv2 = ERRTOTupv2 / (upv7) * 100.;
            ERRTOTdnv1 = sqrt(ERRSUMdnv1);
            ERRTOTdnv2 = sqrt(ERRSUMdnv2);
            ERRPCdnv1 = ERRTOTdnv1 / (dnv7) * 100.;
            ERRPCdnv2 = ERRTOTdnv2 / (dnv7) * 100.;
            ERRTOTsea1 = sqrt(ERRSUMsea1);
            ERRTOTsea2 = sqrt(ERRSUMsea2);
            ERRPCsea1 = ERRTOTsea1 / (sea7) * 100.;
            ERRPCsea2 = ERRTOTsea2 / (sea7) * 100.;
            ERRTOTseam1 = sqrt(ERRSUMseam1);
            ERRTOTseam2 = sqrt(ERRSUMseam2);
            ERRPCseam1 = ERRTOTseam1 / (seam7) * 100.;
            ERRPCseam2 = ERRTOTseam2 / (seam7) * 100.;
            ERRTOTsplus1 = sqrt(ERRSUMsplus1);
            ERRTOTsplus2 = sqrt(ERRSUMsplus2);
            //std::cout << ERRTOTsplus2 << '\n';
            ERRPCsplus1 = ERRTOTsplus1 / (splus7) * 100.;
            ERRPCsplus2 = ERRTOTsplus2 / (splus7) * 100.;
            ERRTOTsminus1 = sqrt(ERRSUMsminus1);
            ERRTOTsminus2 = sqrt(ERRSUMsminus2);
            ERRPCsminus1 = ERRTOTsminus1 / (sminus7) * 100.;
            ERRPCsminus2 = ERRTOTsminus2 / (sminus7) * 100.;

            ifix = 2;



            GetAllPDFs32(prefix, ifix - 1, x, q, &upv10, &dnv10, &usea10, &dsea10,
                &str10, &sbar10,
                &chm10, &cbar10, &bot10, &bbar10, &glu10, &phot10);
            sea10 = 2. * usea10 + 2. * dsea10 + str10 + sbar10;
            splus10 = str10 + sbar10;
            sminus10 = str10 - sbar10;
            seam10 = -usea10 + dsea10;
            GetAllPDFs32(prefix, ifix, x, q, &upv11, &dnv11, &usea11, &dsea11,
                &str11, &sbar11,
                &chm11, &cbar11, &bot11, &bbar11, &glu11, &phot11);
            sea11 = 2. * usea11 + 2. * dsea11 + str11 + sbar11;
            splus11 = str11 + sbar11;
            sminus11 = str11 - sbar11;
            seam11 = -usea11 + dsea11;

            if (x == 0.199)
            {
                std::cout << nxch << ": " << ERRTOTsplus1 << " " << ERRTOTsplus2 << '\n';
                std::cout << "> " << splus10 << ' ' << splus11 << '\n';
            }

            file << std::setprecision(15) << x << " "
                << 10. * pow((glu10 - glu11) / (ERRTOTg1 + ERRTOTg2),2) << " "
                << 10. * pow((upv10 - upv11) / (ERRTOTupv1 + ERRTOTupv2),2) << " "
                << 10. * pow((dnv10 - dnv11) / (ERRTOTdnv1 + ERRTOTdnv2),2) << " "
                << 10. * pow((sea10 - sea11) / (ERRTOTsea1 + ERRTOTsea2),2) << " "
                << 10. * pow((seam10 - seam11) / (ERRTOTseam1 + ERRTOTseam2),2) << " "
                << 10. * pow((splus10 - splus11) / (ERRTOTsplus1 + ERRTOTsplus2),2) << " "
                << 10. * pow((sminus10 - sminus11) / (ERRTOTsminus1 + ERRTOTsminus2),2) << '\n';
        }
    }

    file.close();
    return 0;

}
