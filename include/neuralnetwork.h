#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"
#include <bits/c++config.h>
#include <vector>

class NeuralNetwork
{
public:
	NeuralNetwork()=default;
    NeuralNetwork(const std::vector<int>& layerSizes, const std::vector<activate_fn>& activationFunctions);
	~NeuralNetwork()=default;
	
    mlm::vecd feedForward(const std::vector<double>& input);

	mlm::vec<Layer> getLayers()const{return layers_;}
	Layer& getLayer(const int& num_layer);
	const std::size_t get_num_layers()const {return layer_sizes_.size() - 1;} // Колличество слоев без учета входного слоя

    void saveToFile(const std::string& filename);
    void loadFromFile(const std::string& filename);
	
private:
	mlm::veci layer_sizes_;
    mlm::vec<Layer> layers_;
};

#endif /* NEURALNETWORK_H */



	// void setWeights(const int& layerNum, const int& neuronNum, const int& weightNum, const double& weight);
    // void setBiases(const int& layerNum, const int& neuronNum, const double& bias);

