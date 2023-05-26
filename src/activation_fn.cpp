#include "../include/activation_fn.h"

#include <cmath>

namespace acitvation_fn 
{
	double sigmoid(double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}

    double tanh(double x)
	{
		return std::tanh(x);
	}

    double relu(double x)
	{
		return std::fmax(0.0, x);
	}

	double sigmoid_derivative(double val)
	{
		return val * (1.0 - val);
	}

	double tanh_derivative(double val)
	{
		return 1.0 - (val * val);
	}

	double relu_derivative(double val)
	{
		if (val > 0)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}

} // end activation_fn


