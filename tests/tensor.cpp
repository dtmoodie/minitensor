#include <gtest/gtest.h>

#include <minitensor/Tensor.hpp>

#include <iostream>
std::vector<float> makeVec(size_t N = 20)
{
    std::vector<float> vec;
    for (size_t i = 0; i < N; ++i)
    {
        vec.push_back(i);
    }
    return vec;
}

TEST(tensor, construct1d)
{
    std::vector<float> vec = makeVec();

    mt::Tensor<float, 1> tensor(vec.data(), {20});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape[0], 20);

    ASSERT_EQ(tensor(0), 0);
    ASSERT_EQ(tensor(1), 1);
    ASSERT_EQ(tensor(2), 2);
    ASSERT_EQ(tensor(3), 3);
    ASSERT_EQ(tensor(4), 4);
    ASSERT_EQ(tensor(5), 5);
    ASSERT_EQ(tensor(6), 6);
    ASSERT_EQ(tensor(7), 7);
}

TEST(tensor, construct2d)
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

TEST(tensor, construct3d)
{
    std::vector<float> vec = makeVec(5*4*3);

    mt::Tensor<float, 3> tensor(vec.data(), {5, 4, 3});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape[0], 5);
    ASSERT_EQ(shape[1], 4);
    ASSERT_EQ(shape[2], 3);

    ASSERT_EQ(tensor(0, 0, 0), 0);
    ASSERT_EQ(tensor(0, 0, 1), 1);
    ASSERT_EQ(tensor(0, 0, 2), 2);
    ASSERT_EQ(tensor(1, 0, 0), 12);
    ASSERT_EQ(tensor(1, 0, 1), 13);
    ASSERT_EQ(tensor(1, 0, 2), 14);
}

TEST(tensor, array_indexing3d)
{
    std::vector<float> vec = makeVec(5*4*3);

    mt::Tensor<float, 3> tensor(vec.data(), {5, 4, 3});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape[0], 5);
    ASSERT_EQ(shape[1], 4);
    ASSERT_EQ(shape[2], 3);

    mt::Tensor<float, 2> tensor2d = tensor[0];
    ASSERT_EQ(tensor2d(0,0), tensor(0,0,0));
    ASSERT_EQ(tensor2d(1,0), tensor(0,1,0));
    ASSERT_EQ(tensor2d(1,1), tensor(0,1,1));
    ASSERT_EQ(tensor2d(2,0), tensor(0,2,0));
}

TEST(tensor, array_indexing2d)
{
    std::vector<float> vec = makeVec(5*4);

    mt::Tensor<float, 2> tensor(vec.data(), {5, 4});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape[0], 5);
    ASSERT_EQ(shape[1], 4);

    mt::Tensor<float, 1> tensor2d = tensor[0];
    ASSERT_EQ(tensor2d(0), tensor(0,0));
    ASSERT_EQ(tensor2d(1), tensor(0,1));
    ASSERT_EQ(tensor2d(2), tensor(0,2));
    ASSERT_EQ(tensor2d(0), tensor(0,0));
    ASSERT_EQ(tensor2d(1), tensor(0,1));
    ASSERT_EQ(tensor2d(2), tensor(0,2));
}

TEST(tensor, array_indexing3D)
{
    std::vector<float> vec = makeVec(5*4*3);

    mt::Tensor<float, 3> tensor(vec.data(), {5,4,3});
    auto shape = tensor.getShape();
    ASSERT_EQ(shape[0], 5);
    ASSERT_EQ(shape[1], 4);
    ASSERT_EQ(shape[2], 3);

    mt::Tensor<float, 2> tensor2d = tensor[0];
    ASSERT_EQ(tensor2d(0,0), tensor(0,0,0));
    ASSERT_EQ(tensor2d(1), tensor(0,1));
    ASSERT_EQ(tensor2d(2), tensor(0,2));
    ASSERT_EQ(tensor2d(0), tensor(0,0));
    ASSERT_EQ(tensor2d(1), tensor(0,1));
    ASSERT_EQ(tensor2d(2), tensor(0,2));
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

TEST(tesnro, copy)
{
    std::vector<float> vec1 = makeVec();
    std::vector<float> vec2(vec1.size());
    mt::Tensor<float, 2> t1(vec1.data(), {5,4});

    mt::Tensor<float, 2> t2(vec2.data(), {5,4});
    t2 = t1;
    for(size_t i = 0; i < vec1.size(); ++i)
    {
        ASSERT_EQ(vec1[i], vec2[i]);
    }
}


TEST(tensor, wrap_vec_1d)
{
    std::vector<float> vec = makeVec();
    mt::Tensor<float, 1> tensor(vec.data(), {5*2*2});
    ASSERT_EQ(tensor.data(), vec.data());
    for(size_t i = 0; i < tensor.getShape()[0]; ++i)
    {
        ASSERT_EQ(tensor[i], vec[i]);
    }
}

TEST(tensor, print)
{
    std::vector<float> vec = makeVec();
    {
        mt::Tensor<float, 3> tensor(vec.data(), {5, 2, 2});
        ASSERT_EQ(tensor.getShape(), mt::Shape<3>(5, 2, 2));
        std::stringstream ss;
        ss << tensor;
        ASSERT_EQ(ss.str(),
                  "size: 5 2 2\nstride: 4 2 1\nDataType: f\n   0  1\n   2  3\n\n   4  5\n   6  7\n\n   8  9\n   10  "
                  "11\n\n   12  13\n   14  15\n\n   16  17\n   18  19\n\n\n");
    }
    {
        mt::Tensor<float, 2> tensor(vec.data(), {5, 4});
        std::stringstream ss;
        ss << tensor;
        ASSERT_EQ(
            ss.str(),
            "size: 5 4\nstride: 4 1\nDataType: f\n 0 1 2 3\n 4 5 6 7\n 8 9 10 11\n 12 13 14 15\n 16 17 18 19\n\n");
    }
}

TEST(tensor, implicit_unsqueeze)
{
    std::vector<float> vec = makeVec();
    mt::Tensor<float, 2> tensor(vec.data(), {5,4});

    mt::Tensor<float, 3> t3 = tensor;

    ASSERT_EQ(t3.getShape()[0], 1);
    ASSERT_EQ(t3.getShape()[1], 5);
    ASSERT_EQ(t3.getShape()[2], 4);
}

TEST(tensor, implicit_unsqueeze_alot)
{
    std::vector<float> vec = makeVec();
    mt::Tensor<float, 2> tensor(vec.data(), {5,4});

    mt::Tensor<float, 5> t3 = tensor;

    ASSERT_EQ(t3.getShape()[0], 1);
    ASSERT_EQ(t3.getShape()[1], 1);
    ASSERT_EQ(t3.getShape()[2], 1);
    ASSERT_EQ(t3.getShape()[3], 5);
    ASSERT_EQ(t3.getShape()[4], 4);
}

TEST(tensor_conversion, float1d)
{
    float val = 1.0F;
    auto tensor = mt::tensorWrap(val);

    float v2 = tensor;
    ASSERT_EQ(val, v2);
}

TEST(tensor_conversion, const_float1d)
{
    const float val = 1.0F;
    auto tensor = mt::tensorWrap(val);

    float v2 = tensor;
    ASSERT_EQ(val, v2);
}


TEST(tensor, type_erasure_construct)
{
    std::vector<float> vec = makeVec();
    mt::Tensor<float, 3> a(vec.data(), {5, 2, 2});

    mt::Tensor<void, 3> b = a;

    mt::Tensor<const float, 3> c = b;
    ASSERT_EQ(c.getShape(), a.getShape());
    std::stringstream ss;
    ss << c;
    ASSERT_EQ(ss.str(),
              "size: 5 2 2\nstride: 4 2 1\nDataType: f\n   0  1\n   2  3\n\n   4  5\n   6  7\n\n   8  9\n   10  11\n\n "
              "  12  13\n   14  15\n\n   16  17\n   18  19\n\n\n");
    ASSERT_EQ(c.data(), vec.data());
}

#include <ct/reflect.hpp>
#include <ct/reflect_macros.hpp>
struct Point
{
    REFLECT_INTERNAL_BEGIN(Point)
        REFLECT_INTERNAL_MEMBER(float, x)
        REFLECT_INTERNAL_MEMBER(float, y)
        REFLECT_INTERNAL_MEMBER(float, z)
    REFLECT_INTERNAL_END;
};

TEST(tensor, aggregate_type_conversion)
{
    std::vector<Point> points(5*4);
    mt::Tensor<Point, 2> tensor(points.data(), {5,4});
    mt::Tensor<float, 3> raw_view = tensor;

    raw_view(0,0,0) = 4.0F;
    raw_view(0,0,1) = 5.0F;
    raw_view(0,0,2) = 6.0F;

    ASSERT_EQ(points[0].x, 4.0F);
    ASSERT_EQ(points[0].y, 5.0F);
    ASSERT_EQ(points[0].z, 6.0F);

    raw_view(1,0,0) = 4.0F;
    raw_view(1,0,1) = 5.0F;
    raw_view(1,0,2) = 6.0F;

    ASSERT_EQ(points[4].x, 4.0F);
    ASSERT_EQ(points[4].y, 5.0F);
    ASSERT_EQ(points[4].z, 6.0F);

    mt::Tensor<Point, 2> tensor2 = raw_view;

    tensor2(0,0).x = 0.0F;
    tensor2(0,0).y = 0.0F;
    tensor2(0,0).z = 0.0F;

    ASSERT_EQ(points[0].x, 0.0F);
    ASSERT_EQ(points[0].y, 0.0F);
    ASSERT_EQ(points[0].z, 0.0F);
}
