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

    Tensor(float* array, std::vector<int>& shape);
    Tensor(float val, std::vector<int>& shape);
    ~Tensor();

    std::vector<int> getShape();
    void reshape(std::vector<int>& shape);
    void relu();
    void add(Tensor& other);
    Tensor matmult(Tensor& one, Tensor& two);
    Tensor matmult(Tensor& one, Tensor& two, Tensor& result_container);
};

#endif
