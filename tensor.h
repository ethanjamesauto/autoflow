#include <memory>
#include <vector>

#ifndef TENSOR_H
#define TENSOR_H

/**
 * @brief Implementation of a tensor with variable shape. Essentially equivalent to a TensorFlow tensor.
 */
class Tensor {
   public:
    std::shared_ptr<float[]> array;
    std::vector<int> shape;
    int length;

    Tensor();
    Tensor(std::shared_ptr<float[]> array, std::vector<int> shape);
    Tensor(std::vector<float> array, std::vector<int> shape);
    Tensor(float val, std::vector<int> shape);
    Tensor(const Tensor& t);

    void print();

    void reluMutable();
    Tensor relu() const;

    void softmaxMutable();
    Tensor softmax() const;

    void addMutable(const Tensor& other);
    Tensor add(const Tensor& other) const;

    void scalarMultMutable(float scalar);
    Tensor scalarMult(float scalar);

    static void matmult(const Tensor& one, const Tensor& two, Tensor& out);
    static Tensor matmult(const Tensor& one, const Tensor& two);

    static void outer_product(const Tensor& one, const Tensor& two, Tensor& out);
    
    static float mse(const Tensor& exp, const Tensor& actual);
};

#endif
