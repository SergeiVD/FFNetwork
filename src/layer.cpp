#include "../include/layer.h"
#include "../include/activation_fn.h"

#include <cmath>
#include <random>

Layer::Layer(int inputSize, int outputSize, activate_fn activationFunction)
    : inputSize_(inputSize), outputSize_(outputSize), activationFunction_(activationFunction)
{
    weights_.resize(outputSize_, std::vector<double>(inputSize_));
    biases_.resize(outputSize_);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < outputSize_; i++) {
        for (int j = 0; j < inputSize_; j++) {
            weights_[i][j] = dis(gen); // Initialize weights with random values between -1 and 1
        }
    }

    for (int i = 0; i < outputSize_; i++) {
        biases_[i] = dis(gen); // Initialize biases with random values between -1 and 1
    }
}

Layer::Layer(const Layer& layer)
    : inputSize_(layer.inputSize_), outputSize_(layer.outputSize_), weights_(layer.weights_), biases_(layer.biases_), activationFunction_(layer.activationFunction_)
{
}

Layer::Layer(Layer&& layer) noexcept
    : inputSize_(layer.inputSize_), outputSize_(layer.outputSize_), weights_(std::move(layer.weights_)), biases_(std::move(layer.biases_)), activationFunction_(std::move(layer.activationFunction_))
{
}

Layer& Layer::operator=(const Layer& layer)
{
    inputSize_ = layer.inputSize_;
    outputSize_ = layer.outputSize_;
    weights_ = layer.weights_;
    biases_ = layer.biases_;
    activationFunction_ = layer.activationFunction_;

    return *this;
}

Layer& Layer::operator=(Layer&& layer) noexcept
{
    inputSize_ = std::move(layer.inputSize_);
    outputSize_ = std::move(layer.outputSize_);
    weights_ = std::move(layer.weights_);
    biases_ = std::move(layer.biases_);
    activationFunction_ = std::move(layer.activationFunction_);

    return *this;
}

std::vector<double> Layer::activate(const std::vector<double>& input)
{
    std::vector<double> output(outputSize_);

    for (int j = 0; j < outputSize_; j++) {
        double weightedSum = 0.0;
        for (int i = 0; i < inputSize_; i++) {
            weightedSum += input[i] * weights_[j][i];
        }

        switch (activationFunction_) {
        case activate_fn::sigmoid:
            output[j] = activation_fn::sigmoid(weightedSum + biases_[j]);
            break;
        case activate_fn::relu:
            output[j] = activation_fn::relu(weightedSum + biases_[j]);
            break;
        case activate_fn::tanh:
            output[j] = activation_fn::tanh(weightedSum + biases_[j]);
            break;
        }
    }

    return output;
}

void Layer::setWeights(const int& neuronNum, const int& weightNum, const double& weight)
{
    weights_[neuronNum][weightNum] = weight;
}

void Layer::setBias(const int& neuronNum, const double& bias)
{
    biases_[neuronNum] = bias;
}

std::vector<double> Layer::deriveActivation(const std::vector<double>& val)
{
    std::vector<double> activationDerivative(val.size());

    switch (activationFunction_) {
    case activate_fn::sigmoid:
        for (int i = 0; i < val.size(); ++i) {
            activationDerivative[i] = activation_fn::sigmoidDerivative(val[i]);
        }
        break;

    case activate_fn::relu:
        for (int i = 0; i < val.size(); ++i) {
            activationDerivative[i] = activation_fn::reluDerivative(val[i]);
        }
        break;

    case activate_fn::tanh:
        for (int i = 0; i < val.size(); ++i) {
            activationDerivative[i] = activation_fn::tanhDerivative(val[i]);
        }
        break;
    }

    return activationDerivative;
}



// Метод сохранения значения весов слоя в отдельный файл
void Layer::saveWeightsToFile(const std::string& filename) const
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        for (const auto& weights : weights_)
        {
            for (double weight : weights)
            {
                file << weight << " ";
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

// Метод загрузки значения весов слоя из файла
void Layer::loadWeightsFromFile(const std::string& filename)
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
                weights_[row][col] = weight;
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
