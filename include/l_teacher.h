#ifndef L_TEACHER_H
#define L_TEACHER_H

#include "neuralnetwork.h"
#include <vector>

class L_Teacher {
public:
    L_Teacher(NeuralNetwork& network, const std::vector<std::vector<double>>& trainingData, const std::vector<std::vector<double>>& targetData, double learningRate = 0.1, int numEpochs = 1000);
    void train();

private:
    NeuralNetwork& network_;
    const std::vector<std::vector<double>>& trainingData_;
    const std::vector<std::vector<double>>& targetData_;
    double learningRate_;
    int numEpochs_;

    std::vector<std::vector<std::vector<double>>> backpropagate(const std::vector<double>& error);
    void updateWeights(const std::vector<std::vector<std::vector<double>>>& gradients);
};

#endif /* L_TEACHER_H */
