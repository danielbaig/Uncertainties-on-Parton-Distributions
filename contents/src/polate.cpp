#include "polate.h"


double ExtrapolatePDF(int ip, int np, int ih, int nhess, double x, double y,
    int nx, int my, Eigen::VectorXd* xx, Eigen::VectorXd* yy, Eigen::Tensor<double, 6>* cc)
    
    //integer ih, nx, my, nhess, locx, n, m, ip, np
    //double precision xx(nx), yy(my), cc(np, 0:nhess, nx, my, 4, 4),
    //& x, y, z, f0, f1, z0, z1, InterpolatePDF
    
{

    double z{};
    double f0{};
    double f1{};
    double z0{};
    double z1{};

    int n{ locx(xx, nx, x) };           //0: below xmin, nx : above xmax
    int m{ locx(yy, my, y) };           //0 : below qsqmin, my : above qsqmax

    //   If extrapolation in small x only :
    if (n == 0 && m > 0 && m < my)
    {
        f0 = InterpolatePDF(ip, np, ih, nhess, (*xx)(0), y, nx, my, xx, yy, cc);
        f1 = InterpolatePDF(ip, np, ih, nhess, (*xx)(1), y, nx, my, xx, yy, cc);
        if (f0 > 0. && f1 > 0.)
        {
            z = exp(log(f0) + (log(f1) - log(f0)) / ((*xx)(1) - (*xx)(0)) * (x - (*xx)(0)));
        }
        else
        {
            z = f0 + (f1 - f0) / ((*xx)(1) - (*xx)(0)) * (x - (*xx)(0));
        }

    }

    //  If extrapolation into large q only :
    else if (n > 0 && m == my)
    {
        f0 = InterpolatePDF(ip, np, ih, nhess, x, (*yy)(my-1), nx, my, xx, yy, cc);
        f1 = InterpolatePDF(ip, np, ih, nhess, x, (*yy)(my - 2), nx, my, xx, yy, cc);

        if (f0 > 0. && f1 > 0.)
        {
            z = exp(log(f0) + (log(f0) - log(f1)) / ((*yy)(my-1) - (*yy)(my - 2)) *
                (y - (*yy)(my-1)));
        }
        else
        {
            z = f0 + (f0 - f1) / ((*yy)(my-1) - (*yy)(my - 2)) * (y - (*yy)(my-1));
        }
    }
    //   If extrapolation into large q AND small x :
    else if (n == 0 && m == my)
    {
        f0 = InterpolatePDF(ip, np, ih, nhess, (*xx)(0), (*yy)(my-1), nx, my, xx, yy, cc);
        f1 = InterpolatePDF(ip, np, ih, nhess, (*xx)(0), (*yy)(my - 2), nx, my, xx, yy,
            cc);
        if (f0 > 0. && f1 > 0.)
        {
            z0 = exp(log(f0) + (log(f0) - log(f1)) / ((*yy)(my-1) - (*yy)(my - 2)) *
                (y - (*yy)(my-1)));
        }
        else
        {
            z0 = f0 + (f0 - f1) / ((*yy)(my-1) - (*yy)(my - 2)) * (y - (*yy)(my-1));
        }
        f0 = InterpolatePDF(ip, np, ih, nhess, (*xx)(1), (*yy)(my-1), nx, my, xx, yy, cc);
        f1 = InterpolatePDF(ip, np, ih, nhess, (*xx)(1), (*yy)(my - 2), nx, my, xx, yy,
            cc);
        if (f0 > 0. && f1 > 0.)
        {
            z1 = exp(log(f0) + (log(f0) - log(f1)) / ((*yy)(my-1) - (*yy)(my - 2)) *
                (y - (*yy)(my-1)));
        }
        else
        {
            z1 = f0 + (f0 - f1) / ((*yy)(my-1) - (*yy)(my - 2)) * (y - (*yy)(my-1));
        }
        if (z0 > 0. && z1 > 0.)
        {
            z = exp(log(z0) + (log(z1) - log(z0)) / ((*xx)(1) - (*xx)(0)) * (x - (*xx)(0)));
        }
        else
        {
            z = z0 + (z1 - z0) / ((*xx)(1) - (*xx)(0)) * (x - (*xx)(0));
        }
    }
    else
    {
        std::cout << "Error in ExtrapolatePDF";
        return -1;
    }

    return z;

}







double InterpolatePDF(int ip, int np, int ih, int nhess, double x, double y,
    int nx, int my, Eigen::VectorXd* xx, Eigen::VectorXd* yy, Eigen::Tensor<double, 6>* cc)
    //integer ih, nx, my, nhess, locx, l, m, n, ip, np
    //double precision xx(nx), yy(my), cc(np, 0:nhess, nx, my, 4, 4),
    //& x, y, z, t, u

{

    static int n{};
    static int m{};

    

    n = locx(xx, nx, x);
    m = locx(yy, my, y);

    //std::cout << "n,m=" << n << "," << m << '\n';

    double t{ (x - (*xx)(n-1)) / ((*xx)(n) - (*xx)(n-1)) };
    double u{ (y - (*yy)(m-1)) / ((*yy)(m) - (*yy)(m-1)) };

    
    //std::cout << "t: " << t << '\n';
    //std::cout << "u: " << u << '\n';
    
    double z{ 0. };
    for (int l{ 4 }; l >= 1; --l)
    {
        //std::cout << "cc..."
          //  << cc(ip - 1, ih, n - 1, m - 1, l - 1, 0) << '\n';

        z = t * z + (((*cc)(ip-1, ih, n-1, m-1, l-1, 3) * u + (*cc)(ip-1, ih, n-1, m-1, l-1, 2)) * u
            + (*cc)(ip-1, ih, n-1, m-1, l-1, 1)) * u + (*cc)(ip-1, ih, n-1, m-1, l-1, 0);
    }

    //std::cout << z << '\n';


    return z;
}


