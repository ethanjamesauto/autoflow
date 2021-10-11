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

    Tensor(float**, std::vector<int> shape);
    Tensor(float, std::vector<int> shape);

    Tensor();
    ~Tensor();

    std::vector<int> shape();
    void reshape(std::vector<int> shape);
    void relu();
    void add(Tensor other);
    Tensor matmult(Tensor one, Tensor two);
    Tensor matmult(Tensor one, Tensor two, Tensor result_container);
};

#endif