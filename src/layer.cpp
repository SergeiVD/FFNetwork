#include "../include/layer.h"
#include "../include/activation_fn.h"

#include <cmath>
#include <random>

Layer::Layer(int input_size, int output_size, activate_fn activ_fn) :
        input_size_{input_size},
        output_size_{output_size},
        weights_(output_size, std::vector<double>(input_size)),
        biases_(output_size),
		activ_fn_{activ_fn}
{
	// Инициализация весов и смещений
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-1.0, 1.0);
        
	// for (int i = 0; i < input_size_; i++)
	// {
	// 	for (int j = 0; j < output_size_; j++)
	// 	{
	// 		weights_[i][j] = dis(gen); // Инициализация случайными значениями от -1 до 1
	// 	}
	// }

	for (int i = 0; i < output_size_; i++)
	{
		for (int j = 0; j < input_size_; j++)
		{
			weights_[i][j] = dis(gen); // Инициализация случайными значениями от -1 до 1
		}
	}


	for (int i = 0; i < output_size_; i++)
	{
		biases_[i] = dis(gen); // Инициализация случайными значениями от -1 до 1
	}
}

Layer::Layer(const Layer &layer) :
	input_size_{layer.input_size_},
	output_size_{layer.output_size_},
	weights_(layer.weights_),
	biases_(layer.biases_),
	activ_fn_(layer.activ_fn_)
{
}

Layer::Layer(const Layer&& layer)noexcept :
	input_size_(layer.input_size_),
	output_size_(layer.output_size_),
	weights_(std::move(layer.weights_)),
	biases_(std::move(layer.biases_)),
	activ_fn_(std::move(layer.activ_fn_))
	
{	
}

Layer& Layer::operator=(const Layer& layer)
{
	input_size_ = layer.input_size_;
	output_size_ = layer.output_size_;
	weights_ = layer.weights_;
	biases_ = layer.biases_;
	activ_fn_ = layer.activ_fn_;

	return *this;
}

Layer& Layer::operator=(Layer&& layer)
{
	input_size_ = std::move(layer.input_size_);
	output_size_ = std::move(layer.output_size_);
	weights_ = std::move(layer.weights_);
	biases_ = std::move(layer.biases_);
	activ_fn_ = std::move(layer.activ_fn_);

	return *this;
	
}

std::vector<double> Layer::activate(const std::vector<double>& input)
{
	std::vector<double> output(output_size_);
		
	// Умножение входа на веса и добавление смещений
	for (int j = 0; j < output_size_; j++)
	{
		double weighted_sum = 0.0;
		for (int i = 0; i < input_size_; i++)
		{
			weighted_sum += input[i] * weights_[j][i];
		}
		
		switch (activ_fn_)
		{
		case activate_fn::sigmoid: output[j] = acitvation_fn::sigmoid(weighted_sum + biases_[j]);
			break;
		case activate_fn::relu: output[j] = acitvation_fn::relu(weighted_sum + biases_[j]);
			break;
		case activate_fn::tanh: output[j] = acitvation_fn::tanh(weighted_sum + biases_[j]);
			break;
		}

	}
	return output;
}

// void Layer::activate(const std::vector<double>& input)
// {
// 	input_ = input;
// 	output_.resize(output_size_);
// 	// Умножение входа на веса и добавление смещений
// 	for (int j = 0; j < output_size_; j++)
// 	{
// 		double weighted_sum = 0.0;
// 		for (int i = 0; i < input_size_; i++)
// 		{
// 			weighted_sum += input[i] * weights_[j][i];
// 		}
		
// 		switch (activ_fn_)
// 		{
// 		case activate_fn::sigmoid: output_[j] = acitvation_fn::sigmoid(weighted_sum + biases_[j]);
// 			break;
// 		case activate_fn::relu: output_[j] = acitvation_fn::relu(weighted_sum + biases_[j]);
// 			break;
// 		case activate_fn::tanh: output_[j] = acitvation_fn::tanh(weighted_sum + biases_[j]);
// 			break;
// 		}
// 	}
// }

void Layer::set_weights(const int& num_neuron, const int& num_weigth, const double& weigth)
{
	weights_.at(num_neuron).at(num_weigth) = weigth;
}

void Layer::set_bias(const int& num_neuron, const double& bias)
{
	biases_.at(num_neuron) = bias;
}

std::vector<double> Layer::derive_activation(const std::vector<double>& val)
{
	std::vector<double> activ_derivative(val.size());
	
	switch (activ_fn_)
	{
	case activate_fn::sigmoid:
	{
		for(int i = 0; i < val.size(); ++i)
		{
			activ_derivative[i] = acitvation_fn::sigmoid_derivative(val.at(i));
		}
		return activ_derivative;
		break;
	}

	case activate_fn::relu:
	{
		for(int i = 0; i < val.size(); ++i)
		{
			activ_derivative[i] = acitvation_fn::relu_derivative(val.at(i));
		}
		return activ_derivative;
		break;
	}

	case activate_fn::tanh:
	{
		for(int i = 0; i < val.size(); ++i)
		{
			activ_derivative[i] = acitvation_fn::relu_derivative(val.at(i));
		}
		return activ_derivative;
		break;
	}

}

}




