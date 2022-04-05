#include <gtest/gtest.h>

#include <minitensor/Shape.hpp>

#include <iostream>
#include <vector>

TEST(shape, construct)
{
    mt::Shape<4> shape(5, 4, 3, 2);
    ASSERT_EQ(shape[0], 5);
    ASSERT_EQ(shape[1], 4);
    ASSERT_EQ(shape[2], 3);
    ASSERT_EQ(shape[3], 2);

    ASSERT_EQ(shape.getStride(0), 4 * 3 * 2);
    ASSERT_EQ(shape.getStride(1), 3 * 2);
    ASSERT_EQ(shape.getStride(2), 2);
    ASSERT_EQ(shape.getStride(3), 1);
}

TEST(shape, index)
{
    mt::Shape<4> shape(5, 4, 3, 2);

    ASSERT_EQ(shape.index(1, 1, 1, 1), 33);
}

TEST(shape, reverse_index)
{
    mt::Shape<4> shape(5, 4, 3, 2);

    ASSERT_EQ(shape.index(-1, -1, -1, -1), 119);
}

TEST(shape, linear_index)
{
    mt::Shape<2> shape(5, 4);

    ASSERT_EQ(shape.index(1), 4);
    ASSERT_EQ(shape.index(1, 1), 5);
    ASSERT_EQ(shape.index(5), 5*4);

    ASSERT_EQ(shape.linearIndex(5), 5);
    ASSERT_EQ(shape.linearIndex(6), 6);
}

TEST(shape, unsqueezing)
{
    mt::Shape<2> shape(5,4);
    mt::Shape<3> shape3 = shape;
    ASSERT_EQ(shape3[0], 1);
    ASSERT_EQ(shape3[1], 5);
    ASSERT_EQ(shape3[2], 4);

    ASSERT_EQ(shape3.getStride(2), shape.getStride(1));
    ASSERT_EQ(shape3.getStride(1), shape.getStride(0));
    ASSERT_EQ(shape3.getStride(0), 5*4);

}
