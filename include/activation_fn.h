#ifndef ACTIVATION_FN_H
#define ACTIVATION_FN_H


namespace acitvation_fn 
{
	double sigmoid(double x);
	double tanh(double x);
	double relu(double x);

	double sigmoid_derivative(double val);
	double tanh_derivative(double val);
	double relu_derivative(double val);
}


#endif /* ACTIVATION_FN_H */
