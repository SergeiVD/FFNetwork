#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <vector>
#include "layer.h"

class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<int> layer_sizes, std::vector<activate_fn> functions);

    /* * functions feed forward*/
    std::vector<double> feed_forward(double* input);
    std::vector<double> feed_forward(const std::vector<double>& input);

    /* * setter for weights*/
	void set_weights(const int& num_layer, const int& num_neuron, const int& num_weight, const double& weight);

    /* * setter for biases*/
	void set_biases(const int& num_layer, const int& num_neuron, const double& bias);

	/* * getter for neuron weights */
	std::vector<std::vector<std::vector<double>>> get_weights();

	/* * getter for neuron weights */
	std::vector<std::vector<double>> get_biases();

	/* * getter for neuron net layers */
	std::vector<Layer> get_layers() {return layers_;}

	/* * getter for output layer */
	std::vector<double> get_output_layers(const int& num_layer){return output_layers_.at(num_layer);}
	
	std::vector<double> layer_derivative(const int& num_layer);

private:
    std::vector<int> layer_sizes_;
    std::vector<activate_fn> functions_;
    std::vector<Layer> layers_;

	std::vector<std::vector<double>> output_layers_;
};


#endif /* NEURALNETWORK_H */
