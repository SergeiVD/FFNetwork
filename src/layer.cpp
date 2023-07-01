#include "../include/layer.h"
#include "../include/activation_fn.h"

#include <bits/c++config.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <stdexcept>

Layer::Layer(std::size_t inputSize, std::size_t outputSize, activate_fn activationFunction)
    :
	num_input_(inputSize),
	num_out_(outputSize),
	input_(inputSize),
	output_(outputSize),
	weights_(outputSize, inputSize),
	biases_(outputSize),
	activationFunction_(activationFunction)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < num_out_; i++)
	{
        for (int j = 0; j < num_input_; j++)
		{
            weights_[i][j] = dis(gen); // Initialize weights with random values between -1 and 1
        }
    }

    for (int i = 0; i < num_out_; i++)
	{
        biases_[i] = dis(gen); // Initialize biases with random values between -1 and 1
    }
	
}

Layer::Layer(const Layer& layer)
    :
	num_input_(layer.num_input_),
	num_out_(layer.num_out_),
	input_(layer.input_),
	output_(layer.output_),
	weights_(layer.weights_),
	biases_(layer.biases_),
	activationFunction_(layer.activationFunction_)
{
}

Layer::Layer(Layer&& layer) noexcept
    :
	num_input_(layer.num_input_),
	num_out_(layer.num_out_),
	input_(std::move(layer.input_)),
	output_(std::move(layer.output_)),
	weights_(std::move(layer.weights_)),
	biases_(std::move(layer.biases_)),
	activationFunction_(std::move(layer.activationFunction_))
{
}

Layer& Layer::operator=(const Layer& layer)
{
    num_input_ = layer.num_input_;
    num_out_ = layer.num_out_;
	input_ = layer.input_;
	output_ = layer.output_;
    weights_ = layer.weights_;
    biases_ = layer.biases_;
    activationFunction_ = layer.activationFunction_;

    return *this;
}

Layer& Layer::operator=(Layer&& layer) noexcept
{
    num_input_ = std::move(layer.num_input_);
    num_out_ = std::move(layer.num_out_);
	input_ = std::move(layer.input_);
	output_ = std::move(layer.output_);
    weights_ = std::move(layer.weights_);
    biases_ = std::move(layer.biases_);
    activationFunction_ = std::move(layer.activationFunction_);

    return *this;
}

Layer::~Layer()
{
	
}

const mlm::vecd& Layer::activate(const mlm::vecd& input)
{
    input_ = input;

    for (int j = 0; j < num_out_; j++)
	{
        double weightedSum = 0.0;
        for (int i = 0; i < num_input_; i++)
		{
            weightedSum += input[i] * weights_(j,i);
        }
		
        switch (activationFunction_)
		{
        case activate_fn::sigmoid:
            output_[j] = activation_fn::sigmoid(weightedSum + biases_[j]);
            break;
        case activate_fn::relu:
            output_[j] = activation_fn::relu(weightedSum + biases_[j]);
            break;
        case activate_fn::tanh:
            output_[j] = activation_fn::tanh(weightedSum + biases_[j]);
            break;
        }
    }
	
    return output_;
}

void Layer::update_weights(const int& neuronNum, const int& weightNum, const double& weight_gradient)
{
    weights_(neuronNum,weightNum) += weight_gradient;
}

void Layer::update_bias(const int& neuronNum, const double& bias_gradient)
{
    biases_[neuronNum] += bias_gradient;
}

mlm::vecd Layer::deriveActivation(const mlm::vecd& val)
{
	mlm::vecd activationDerivative(val.size());

    switch (activationFunction_)
	{
    case activate_fn::sigmoid:
        for (int i = 0; i < val.size(); ++i)
		{
            activationDerivative[i] = activation_fn::sigmoidDerivative(val[i]);
        }
        break;

    case activate_fn::relu:
        for (int i = 0; i < val.size(); ++i)
		{
            activationDerivative[i] = activation_fn::reluDerivative(val[i]);
        }
        break;

    case activate_fn::tanh:
        for (int i = 0; i < val.size(); ++i)
		{
            activationDerivative[i] = activation_fn::tanhDerivative(val[i]);
        }
        break;
    }

    return activationDerivative;
}


const double Layer::derive_activation(const std::size_t& num_output)const
{
	if(num_output >= output_.size())throw std::invalid_argument("error! derive_activation num_output biggest then output size");
    switch (activationFunction_)
	{
    case activate_fn::sigmoid:
        return activation_fn::sigmoidDerivative(output_[num_output]);
        break;

    case activate_fn::relu:
		return activation_fn::reluDerivative(output_[num_output]);
        break;

    case activate_fn::tanh:
		return activation_fn::tanhDerivative(output_[num_output]);
        break;
    }
	
}

const mlm::vecd Layer::derive_activation()const
{
	mlm::vecd activationDerivative(output_.size());
    switch (activationFunction_)
	{
    case activate_fn::sigmoid:
        for (int i = 0; i < output_.size(); ++i)
		{
            activationDerivative[i] = activation_fn::sigmoidDerivative(output_[i]);
        }
        break;

    case activate_fn::relu:
        for (int i = 0; i < output_.size(); ++i)
		{
            activationDerivative[i] = activation_fn::reluDerivative(output_[i]);
        }
        break;

    case activate_fn::tanh:
        for (int i = 0; i < output_.size(); ++i)
		{
            activationDerivative[i] = activation_fn::tanhDerivative(output_[i]);
        }
        break;
    }
	return activationDerivative;
}


void Layer::save_weights_to_file(const std::string& filename) const
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        for(size_t i = 0; i < weights_.rows(); ++i)
        {
			for (size_t j = 0; j < weights_.cols(); ++j)
			{
				file << weights_(i,j) << " ";
			}

            file << std::endl;
        }
        file.close();
    }
    else
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}


void Layer::load_weights_from_file(const std::string& filename)
{
    std::ifstream file(filename);
    if (file.is_open())
    {
        std::string line;
        int row = 0;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::string weight_str;
            int col = 0;
            while (iss >> weight_str)
            {
                double weight = std::stod(weight_str);
                weights_(row,col) = weight;
                col++;
            }
            row++;
        }
        file.close();
    }
    else
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}
