#ifndef LAYER_H
#define LAYER_H

#include "activation_fn.h"
#include <vector>

class Layer {
public:
    Layer(int inputSize, int outputSize, activate_fn activationFunction);
    Layer(const Layer& layer);
    Layer(Layer&& layer) noexcept;
    Layer& operator=(const Layer& layer);
    Layer& operator=(Layer&& layer) noexcept;
    std::vector<double> activate(const std::vector<double>& input);
    void setWeights(const int& neuronNum, const int& weightNum, const double& weight);
    void setBias(const int& neuronNum, const double& bias);
    std::vector<double> deriveActivation(const std::vector<double>& val);
    const std::vector<std::vector<double>>& getWeights() const { return weights_; }
    const std::vector<double>& getBiases() const { return biases_; }


    void saveWeightsToFile(const std::string& filename) const;
    void loadWeightsFromFile(const std::string& filename);

private:
    int inputSize_;
    int outputSize_;
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;
    activate_fn activationFunction_;
};

#endif /* LAYER_H */
