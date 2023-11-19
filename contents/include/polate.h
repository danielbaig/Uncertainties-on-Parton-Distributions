#ifndef POLATE_H
#define POLATE_H

#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "deriv_locx.h"

double ExtrapolatePDF(int ip, int np, int ih, int nhess, double x, double y,
    int nx, int my, Eigen::VectorXd* xx, Eigen::VectorXd* yy, Eigen::Tensor<double, 6>* cc);

double InterpolatePDF(int ip, int np, int ih, int nhess, double x, double y,
    int nx, int my, Eigen::VectorXd* xx, Eigen::VectorXd* yy, Eigen::Tensor<double, 6>* cc);







#endif POLATE_H
