#include <gtest/gtest.h>

#include <minitensor/Tensor.hpp>

#include <iostream>
std::vector<float> makeVec()
{
    std::vector<float> vec;
    for (size_t i = 0; i < 20; ++i)
    {
        vec.push_back(i);
    }
    return vec;
}

TEST(tensor, construct)
{
    std::vector<float> vec = makeVec();

    mt::Tensor<float, 2> tensor(vec.data(), {5, 4});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape[0], 5);
    ASSERT_EQ(shape[1], 4);

    ASSERT_EQ(tensor(0, 0), 0);
    ASSERT_EQ(tensor(0, 1), 1);
    ASSERT_EQ(tensor(0, 2), 2);
    ASSERT_EQ(tensor(0, 3), 3);
    ASSERT_EQ(tensor(1, 0), 4);
    ASSERT_EQ(tensor(1, 1), 5);
    ASSERT_EQ(tensor(1, 2), 6);
    ASSERT_EQ(tensor(1, 3), 7);
}

TEST(tensor, reverse_index)
{
    std::vector<float> vec = makeVec();

    mt::Tensor<float, 2> tensor(vec.data(), {5, 4});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape[0], 5);
    ASSERT_EQ(shape[1], 4);

    ASSERT_EQ(tensor(-1, 0), 16);
    ASSERT_EQ(tensor(-1, 1), 17);
    ASSERT_EQ(tensor(-1, 2), 18);
    ASSERT_EQ(tensor(-1, 3), 19);
}

TEST(tensor, iterate)
{
    std::vector<float> vec = makeVec();
    {
        mt::Tensor<float, 3> tensor(vec.data(), {5, 2, 2});
        std::cout << tensor << std::endl;
    }
    {
        mt::Tensor<float, 2> tensor(vec.data(), {5, 4});
        std::cout << tensor << std::endl;
    }
}

TEST(tensor, type_erasure)
{
    std::vector<float> vec = makeVec();
    mt::Tensor<float, 3> a(vec.data(), {5, 2, 2});

    mt::Tensor<void, 3> b = a;

    mt::Tensor<const float, 3> c = b;
    std::cout << c << std::endl;
}