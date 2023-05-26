#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layerSizes, const std::vector<activate_fn>& activationFunctions);
    std::vector<double> feedForward(const std::vector<double>& input);
    void setWeights(const int& layerNum, const int& neuronNum, const int& weightNum, const double& weight);
    void setBiases(const int& layerNum, const int& neuronNum, const double& bias);
    const std::vector<std::vector<std::vector<double>>>& getWeights() const { return layers_; }
    const std::vector<std::vector<double>>& getBiases() const { return biases_; }

    void NeuralNetwork::saveToFile(const std::string& filename) const;
    void NeuralNetwork::loadFromFile(const std::string& filename);

private:
    std::vector<Layer> layers_;
    std::vector<std::vector<double>> biases_;
    std::vector<std::vector<double>> outputLayers_;
};

#endif /* NEURALNETWORK_H */
