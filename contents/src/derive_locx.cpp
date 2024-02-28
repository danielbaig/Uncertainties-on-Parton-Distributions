#include "deriv_locx.h"

#include <iostream>
int locx(Eigen::VectorXd* xx, int nx, double x)
{
	//  returns an integer j such that x lies inbetween xx(j) and xx(j + 1).
	//   nx is the length of the array with xx(nx) the highest element.

	int jl{};
	int ju{};
	int jm{};
	if (x == (*xx)(0))
	{
		return 1;
	}
	if (x == (*xx)(nx-1))
	{
		return nx - 1;

	}
	ju = nx + 1;
	jl = 0;
	while ((ju - jl) > 1)
	{
		jm = (ju + jl) / 2;
		if (x >= (*xx)(jm-1))
		{
			jl = jm;
		}
		else
		{
			ju = jm;
		}

	}
	
	assert(jl != 0);

	return jl;	
}




double polderiv1(double x1, double x2, double x3, double y1, double y2, double y3)
{
	//   returns the estimate of the derivative at x1 obtained by a
	//   polynomial interpolation using the three points(x_i, y_i).
	return (x3 * x3 * (y1 - y2) + 2. * x1 * (x3 * (-y1 + y2) + x2 * (y1 - y3))
		+ x2 * x2 * (-y1 + y3) + x1 * x1 * (-y2 + y3)) / ((x1 - x2) * (x1 - x3) * (x2 - x3));
}

double polderiv2(double x1, double x2, double x3, double y1, double y2, double y3)
{
	//   returns the estimate of the derivative at x2 obtained by a
	//   polynomial interpolation using the three points(x_i, y_i).
		return (x3 * x3 * (y1 - y2) - 2. * x2 * (x3 * (y1 - y2) + x1 * (y2 - y3))
			+ x2 * x2 * (y1 - y3) + x1 * x1 * (y2 - y3)) / ((x1 - x2) * (x1 - x3) * (x2 - x3));
}

double polderiv3(double x1, double x2, double x3, double y1, double y2, double y3)
{
	//   returns the estimate of the derivative at x3 obtained by a
	//   polynomial interpolation using the three points(x_i, y_i).
	return (x3 * x3 * (-y1 + y2) + 2. * x2 * x3 * (y1 - y3) + x1 * x1 * (y2 - y3)
		+ x2 * x2 * (-y1 + y3) + 2. * x1 * x3 * (-y2 + y3)) /
		((x1 - x2) * (x1 - x3) * (x2 - x3));
}