#ifndef DERIV_LOCX_H
#define DERIV_LOCX_H

#include <Eigen/Dense>

int locx(Eigen::VectorXd* xx, int nx, double x);

double polderiv1(double x1, double x2, double x3, double y1, double y2, double y3);

double polderiv2(double x1, double x2, double x3, double y1, double y2, double y3);

double polderiv3(double x1, double x2, double x3, double y1, double y2, double y3);


#endif DERIV_LOCX_H
