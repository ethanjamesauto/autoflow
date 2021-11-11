#include "neural_network.h"
#include <cmath>

Operation::Operation(Tensor* input) {
    this->input = input;
}

Operation::Operation() {}

WeightedOperation::WeightedOperation(Tensor* input)
    : Operation(input) {
}

WeightedOperation::WeightedOperation()
    : Operation() {
}

void WeightedOperation::updateWeights(Tensor& factor) {
    Tensor learningRate(.1, factor.shape);
    Tensor update(weights.shape);  //TODO: possibly optimize??
    Tensor::elementmult(factor, learningRate, update);
    Tensor::scalarMult(update, -1, update);
    Tensor::add(weights, update, weights);
}

void WeightedOperation::RMSProp(Tensor& factor) {
    float learning_rate = .001;
    float b = .90;
    for (int i = 0; i < weights.length; i++) {
        learningRate.array[i] = b * learningRate.array[i] + (1 - b) * pow(factor.array[i], 2);
        weights.array[i] -= learning_rate * factor.array[i] / sqrt(learningRate.array[i] + 1e-8);
    }
}

NeuralNetwork::NeuralNetwork(std::vector<Operation> sequence) {
    this->sequence = sequence;
}
