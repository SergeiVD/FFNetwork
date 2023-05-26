#include "../include/activation_fn.h"

#include <cmath>

namespace activation_fn {
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double relu(double x)
    {
        return std::fmax(0.0, x);
    }

    double tanh(double x)
    {
        return std::tanh(x);
    }

    double sigmoidDerivative(double val)
    {
        return val * (1.0 - val);
    }

    double reluDerivative(double val)
    {
        if (val > 0.0) {
            return 1.0;
        }
        else {
            return 0.0;
        }
    }

    double tanhDerivative(double val)
    {
        return 1.0 - (val * val);
    }
}
