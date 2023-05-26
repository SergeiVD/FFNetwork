#include "../include/l_techer.h"
#include "../include/activation_fn.h"

L_Teacher::L_Teacher(NeuralNetwork& network, const std::vector<std::vector<double>>& trainingData, const std::vector<std::vector<double>>& targetData, double learningRate, int numEpochs)
    : network_(network), trainingData_(trainingData), targetData_(targetData), learningRate_(learningRate), numEpochs_(numEpochs)
{
}

void L_Teacher::train()
{
    for (int epoch = 0; epoch < numEpochs_; ++epoch) {
        double epochError = 0.0;
        for (size_t i = 0; i < trainingData_.size(); ++i) {
            auto output = network_.feedForward(trainingData_[i]);

            std::vector<double> error(output.size());
            for (size_t j = 0; j < targetData_[i].size(); ++j) {
                error[j] = targetData_[i][j] - output[j];
            }

            epochError += std::inner_product(error.begin(), error.end(), error.begin(), 0.0);

            auto gradients = backpropagate(error);
            updateWeights(gradients);
        }
        epochError /= trainingData_.size();
        std::cout << "Epoch " << epoch + 1 << ", Error = " << epochError << std::endl;
    }
}

std::vector<std::vector<std::vector<double>>> L_Teacher::backpropagate(const std::vector<double>& error)
{
    std::vector<std::vector<std::vector<double>>> gradients(network_.getWeights().size());

    int lastLayerIndex = network_.getWeights().size() - 1;
    auto delta = error;

    std::vector<double> lastLayerOutput = network_.getOutputLayers()[lastLayerIndex - 1];

    std::vector<std::vector<double>> weightGradients;
    for (size_t i = 0; i < delta.size(); ++i) {
        std::vector<double> weightGradient(lastLayerOutput.size());
        for (size_t j = 0; j < lastLayerOutput.size(); ++j) {
            weightGradient[j] = delta[i] * lastLayerOutput[j];
        }
        weightGradients.push_back(weightGradient);
    }
    gradients[lastLayerIndex] = weightGradients;

    std::vector<std::vector<double>> biasGradients;
    for (size_t i = 0; i < delta.size(); ++i) {
        std::vector<double> biasGradient(delta.size());
        biasGradient[i] = delta[i];
        biasGradients.push_back(biasGradient);
    }
    gradients[lastLayerIndex + 1] = biasGradients;

    for (int layerIndex = lastLayerIndex - 1; layerIndex >= 0; --layerIndex) {
        const std::vector<std::vector<double>>& weights = network_.getWeights()[layerIndex + 1];
        const std::vector<double>& layerDerivative = network_.getLayers()[layerIndex].deriveActivation(network_.getOutputLayers()[layerIndex]);

        std::vector<double> deltaNextLayer(network_.getLayers()[layerIndex + 1].getBiases().size());

        for (size_t i = 0; i < deltaNextLayer.size(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < weights[i].size(); ++j) {
                sum += weights[i][j] * delta[j];
            }
            deltaNextLayer[i] = sum * layerDerivative[i];
        }

        std::vector<std::vector<double>> weightGradients;
        for (size_t i = 0; i < deltaNextLayer.size(); ++i) {
            std::vector<double> weightGradient(network_.getLayers()[layerIndex].getBiases().size());
            for (size_t j = 0; j < network_.getLayers()[layerIndex].getBiases().size(); ++j) {
                weightGradient[j] = deltaNextLayer[i] * network_.getOutputLayers()[layerIndex][j];
            }
            weightGradients.push_back(weightGradient);
        }
        gradients[layerIndex] = weightGradients;

        std::vector<std::vector<double>> biasGradients;
        for (size_t i = 0; i < deltaNextLayer.size(); ++i) {
            std::vector<double> biasGradient(deltaNextLayer.size());
            biasGradient[i] = deltaNextLayer[i];
            biasGradients.push_back(biasGradient);
        }
        gradients[layerIndex + 1] = biasGradients;

        delta = deltaNextLayer;
    }

    return gradients;
}

void L_Teacher::updateWeights(const std::vector<std::vector<std::vector<double>>>& gradients)
{
    const std::vector<std::vector<std::vector<double>>>& weights = network_.getWeights();
    const std::vector<std::vector<double>>& biases = network_.getBiases();

    for (size_t i = 0; i < weights.size(); ++i) {
        const std::vector<std::vector<double>>& weightGradients = gradients[i];
        std::vector<std::vector<double>>& layerWeights = network_.getLayers()[i].getWeights();

        for (size_t j = 0; j < weightGradients.size(); ++j) {
            const std::vector<double>& weightGradient = weightGradients[j];
            std::vector<double>& neuronWeights = layerWeights[j];

            for (size_t k = 0; k < weightGradient.size(); ++k) {
                neuronWeights[k] += learningRate_ * weightGradient[k];
            }
        }
    }

    for (size_t i = 0; i < biases.size(); ++i) {
        const std::vector<double>& biasGradients = gradients[i + 1];
        std::vector<double>& layerBiases = network_.getLayers()[i].getBiases();

        for (size_t j = 0; j < biasGradients.size(); ++j) {
            const std::vector<double>& biasGradient = biasGradients[j];
            layerBiases[j] += learningRate_ * biasGradient[0];
        }
    }
}
