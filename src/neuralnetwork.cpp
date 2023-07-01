#include "../include/neuralnetwork.h"
#include "../include/activation_fn.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

#include "../lib/nlohmann/json.hpp"


NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes, const std::vector<activate_fn>& activationFunctions)
	:
	layer_sizes_(layerSizes),
	layers_(layerSizes.size() - 1)
{
	for (size_t i = 1; i < layerSizes.size(); ++i)
	{
		layers_[i-1] = Layer(layerSizes[i - 1], layerSizes[i], activationFunctions[i - 1]);
	}
}

mlm::vecd NeuralNetwork::feedForward(const std::vector<double>& input)
{	
    mlm::vecd output = input;

    for (size_t i = 0; i < layers_.size(); ++i)
	{
        output = layers_[i].activate(output);
    }

    return output;
}

Layer& NeuralNetwork::getLayer(const int& num_layer)
{
	if(num_layer >= layers_.size())throw std::runtime_error("fun getLayer: no correct num layer");
	return layers_[num_layer];
}

void NeuralNetwork::saveToFile(const std::string& filename) 
{
    nlohmann::json jsonNetwork;

    jsonNetwork["layer_sizes"] = nlohmann::json::parse(layer_sizes_.data(), layer_sizes_.data()+layer_sizes_.size());

    std::vector<std::string> weightFiles;
    for (int i = 0; i < layers_.size(); ++i)
    {
        std::string weightFilename = "weights_layer_" + std::to_string(i) + ".txt";
        weightFiles.push_back(weightFilename);
        layers_[i].save_weights_to_file(weightFilename);
    }
    jsonNetwork["weight_files"] = weightFiles;

    std::ofstream file(filename);
    file << jsonNetwork;
    file.close();
}

void NeuralNetwork::loadFromFile(const std::string& filename)
{
    nlohmann::json jsonNetwork;

    std::ifstream file(filename);
    file >> jsonNetwork;
    file.close();

    layer_sizes_ = jsonNetwork["layer_sizes"].get<std::vector<int>>();

    std::vector<std::string> weightFiles = jsonNetwork["weight_files"].get<std::vector<std::string>>();
    for (int i = 0; i < weightFiles.size(); ++i)
    {
        layers_[i].load_weights_from_file(weightFiles[i]);
    }
}
