#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

/**
 * @brief Implementation of a tensor with variable shape. Essentially equivalent to a TensorFlow tensor.
 */
class Tensor {
   public:
    float* array;
    std::vector<int> shape;
    int length;

    Tensor(float* array, std::vector<int> shape);
    Tensor(float val, std::vector<int> shape);
    ~Tensor();

    Tensor clone();

    void reluMutable();
    void addMutable(Tensor& other);

    Tensor reshape(std::vector<int> shape);
    Tensor relu();
    Tensor matmult(Tensor& one, Tensor& two);
    Tensor matmult(Tensor& one, Tensor& two, Tensor& result_container);
};

#endif
