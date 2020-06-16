#include <gtest/gtest.h>

#include <minitensor/Array.hpp>

TEST(array, construct)
{
    mt::Array<float, 5> arr(0,1,2,3,4);
    ASSERT_EQ(arr[0], 0);
    ASSERT_EQ(arr[1], 1);
    ASSERT_EQ(arr[2], 2);
    ASSERT_EQ(arr[3], 3);
    ASSERT_EQ(arr[4], 4);
}

TEST(array, reverse_index)
{
    mt::Array<float, 5> arr(0,1,2,3,4);
    ASSERT_EQ(arr[-5], 0);
    ASSERT_EQ(arr[-4], 1);
    ASSERT_EQ(arr[-3], 2);
    ASSERT_EQ(arr[-2], 3);
    ASSERT_EQ(arr[-1], 4);
}

TEST(array, print)
{
    mt::Array<float, 5> arr(0,1,2,3,4);
    std::stringstream ss;
    ss << arr;
    std::string str = ss.str();
    ASSERT_EQ(str, "0 1 2 3 4");
}

    