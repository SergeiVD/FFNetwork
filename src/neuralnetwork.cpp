#include "../include/neuralnetwork.h"
#include "../include/activation_fn.h"
#include <vector>

NeuralNetwork::NeuralNetwork(std::vector<int> layer_sizes, std::vector<activate_fn> functions) :
	layer_sizes_{layer_sizes},
	functions_{functions}
{
	for (int i = 1; i < layer_sizes.size(); ++i)
	{
		layers_.push_back(Layer(layer_sizes[i-1], layer_sizes[i], functions_[i-1]));
	}
	output_layers_.resize(layers_.size());
}

std::vector<double> NeuralNetwork::feed_forward(double* input)
{
	std::vector<double> output(input, input + layer_sizes_[0]);

	for (size_t i = 0; i < layers_.size(); ++i)
	{
		output = layers_.at(i).activate(output);
		output_layers_.at(i) = output;
	}

	return output;
}

// std::vector<double> NeuralNetwork::feed_forward(double* input)
// {
// 	std::vector<double> output(input, input + layer_sizes_[0]);

// 	for (auto& layer : layers_)
// 	{
// 		output = layer.activate(output);
// 	}
// 	return output;
// }


std::vector<double> NeuralNetwork::feed_forward(const std::vector<double>& input)
{
	std::vector<double> output(input.data(), input.data() + layer_sizes_[0]);
	
	for (size_t i = 0; i < layers_.size(); ++i)
	{
		output = layers_.at(i).activate(output);
		output_layers_.at(i) = output;
	}

	return output;
}


// std::vector<double> NeuralNetwork::feed_forward(const std::vector<double>& input)
// {
// 	std::vector<double> output(input.data(), input.data() + layer_sizes_[0]);
	
// 	for (auto& layer : layers_)
// 	{
// 		output = layer.activate(output);
// 	}
	
// 	return output;
// }

void NeuralNetwork::set_weights(const int& num_layer, const int& num_neuron, const int& num_weight, const double& weight)
{
	layers_[num_layer].set_weights(num_neuron, num_weight, weight);
}

void NeuralNetwork::set_biases(const int& num_layer, const int& num_neuron, const double& bias)
{
	layers_[num_layer].set_bias(num_neuron, bias);
}

std::vector<std::vector<std::vector<double>>> NeuralNetwork::get_weights()
{
	std::vector<std::vector<std::vector<double>>> weights;
	for(auto& layer : layers_)
	{
		weights.push_back(layer.get_weights());
	}
	return weights;
}

std::vector<std::vector<double>> NeuralNetwork::get_biases()
{
	std::vector<std::vector<double>> biases;
	for(auto& layer : layers_)
	{
		biases.push_back(layer.get_biases());
	}
	return biases;
}

std::vector<double> NeuralNetwork::layer_derivative(const int& num_layer)
{

	switch (layers_.at(num_layer).get_name_activate_fn())
	{
	case activate_fn::sigmoid:
	{
		std::cout<<"sigmoid"<<std::endl;
		break;
	}
	case activate_fn::relu:
	{
		std::cout<<"relu"<<std::endl;
		break;
	}
	case activate_fn::tanh:
	{
		std::cout<<"tanh"<<std::endl;
		break;
	}

default:
		break;
	}
	
	return 	layers_.at(num_layer).derive_activation(output_layers_.at(num_layer));

	// std::vector<double> activ_derivative(layers_.at(num_layer).get_output().size());
	// switch (functions_.at(num_layer))
	// {
	// case activate_fn::sigmoid:
	// {
	// 	for(int i = 0; layers_.at(num_layer).get_output().size(); ++i)
	// 	{
	// 		activ_derivative[i] = acitvation_fn::sigmoid_derivative(layers_.at(num_layer).get_output().at(i));
	// 	}
	// 	return activ_derivative;
	// 	break;
	// }

	// case activate_fn::relu:
	// {
	// 	for(int i = 0; layers_.at(num_layer).get_output().size(); ++i)
	// 	{
	// 		activ_derivative[i] = acitvation_fn::relu_derivative(layers_.at(num_layer).get_output().at(i));
	// 	}
	// 	return activ_derivative;
	// 	break;
	// }

	// case activate_fn::tanh:
	// {
	// 	for(int i = 0; layers_.at(num_layer).get_output().size(); ++i)
	// 	{
	// 		activ_derivative[i] = acitvation_fn::tanh_derivative(layers_.at(num_layer).get_output().at(i));
	// 	}
	// 	return activ_derivative;
	// 	break;
	// }


          // default:
	// 	break;
	// }

	

}




