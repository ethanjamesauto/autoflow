#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

/**
 * @brief Implementation of a tensor with variable shape. Essentially equivalent to a TensorFlow tensor.
 */
class Tensor {
   public:
    float** array;
    std::vector<int> shape;

    std::vector<int> shape();
    float** reshape(std::vector<int> shape);
    float** matmult(Tensor one, Tensor two);
    float** matmult(Tensor one, Tensor two, Tensor result_container);
    float** add(Tensor other);
    float** relu();
};

#endif