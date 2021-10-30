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
    Tensor(std::shared_ptr<float[]> array, std::vector<int>& shape);
    Tensor(std::vector<float>& array, std::vector<int>& shape);
    Tensor(float val, std::vector<int> shape);
    Tensor(const Tensor& t);
    Tensor(std::vector<int> shape);  //TODO: find out why this can't pass by reference

    void print();

    static void relu(const Tensor& t, Tensor& out);
    Tensor relu() const;

    static void softmax(const Tensor& t, Tensor& out);
    Tensor softmax() const;

    static void add(const Tensor& t, const Tensor& add, Tensor& out);
    Tensor add(const Tensor& other) const;

    static void scalarMult(const Tensor& t, float scalar, Tensor& out);
    Tensor scalarMult(float scalar);

    static void matmult(const Tensor& one, const Tensor& two, Tensor& out);
    static Tensor matmult(const Tensor& one, const Tensor& two);

    static void elementmult(const Tensor& one, const Tensor& two, Tensor& out);

    static void outer_product(const Tensor& one, const Tensor& two, Tensor& out);

    static float mse(const Tensor& exp, const Tensor& actual);

    static Tensor random(const std::vector<int>& shape);
};

#endif
