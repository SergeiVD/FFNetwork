#ifndef ACTIVATION_FN_H
#define ACTIVATION_FN_H

namespace activation_fn
{
    double sigmoid(double x);
    double relu(double x);
    double tanh(double x);
	
    double sigmoidDerivative(double val);
    double reluDerivative(double val);
    double tanhDerivative(double val);
}

#endif /* ACTIVATION_FN_H */
