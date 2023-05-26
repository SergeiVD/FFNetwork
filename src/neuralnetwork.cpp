#include "../include/neuralnetwork.h"
#include "../include/activation_fn.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes, const std::vector<activate_fn>& activationFunctions)
{
    layers_.reserve(layerSizes.size() - 1);
    outputLayers_.resize(layerSizes.size() - 1);

    for (int i = 1; i < layerSizes.size(); ++i) {
        layers_.emplace_back(layerSizes[i - 1], layerSizes[i], activationFunctions[i - 1]);
    }
}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double>& input)
{
    std::vector<double> output = input;

    for (size_t i = 0; i < layers_.size(); ++i) {
        output = layers_[i].activate(output);
        outputLayers_[i] = output;
    }

    return output;
}

void NeuralNetwork::setWeights(const int& layerNum, const int& neuronNum, const int& weightNum, const double& weight)
{
    layers_[layerNum].setWeights(neuronNum, weightNum, weight);
}

void NeuralNetwork::setBiases(const int& layerNum, const int& neuronNum, const double& bias)
{
    layers_[layerNum].setBias(neuronNum, bias);
}


// Метод сохранения структуры сети и ссылок на файлы с весами в файл JSON
void NeuralNetwork::saveToFile(const std::string& filename) const
{
    nlohmann::json jsonNetwork;

    // Сохранение структуры сети
    jsonNetwork["layer_sizes"] = layer_sizes_;

    // Сохранение ссылок на файлы с весами
    std::vector<std::string> weightFiles;
    for (int i = 0; i < layers_.size(); ++i)
    {
        std::string weightFilename = "weights_layer_" + std::to_string(i) + ".txt";
        weightFiles.push_back(weightFilename);
        layers_[i].saveWeightsToFile(weightFilename);
    }
    jsonNetwork["weight_files"] = weightFiles;

    // Сохранение в файл
    std::ofstream file(filename);
    file << jsonNetwork;
    file.close();
}

// Метод загрузки структуры сети и ссылок на файлы с весами из файла JSON
void NeuralNetwork::loadFromFile(const std::string& filename)
{
    nlohmann::json jsonNetwork;

    // Чтение файла
    std::ifstream file(filename);
    file >> jsonNetwork;
    file.close();

    // Загрузка структуры сети
    layer_sizes_ = jsonNetwork["layer_sizes"].get<std::vector<int>>();

    // Загрузка ссылок на файлы с весами
    std::vector<std::string> weightFiles = jsonNetwork["weight_files"].get<std::vector<std::string>>();
    for (int i = 0; i < weightFiles.size(); ++i)
    {
        layers_[i].loadWeightsFromFile(weightFiles[i]);
    }
}