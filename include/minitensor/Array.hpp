#ifndef MINITENSOR_ARRAY_HPP
#define MINITENSOR_ARRAY_HPP
#include "defines.hpp"

#include <cstddef>

#include <cstdint>
#include <utility>

namespace mt
{
    template <class T, class U>
    constexpr auto revIndex(T val, U) -> typename std::enable_if<std::is_unsigned<T>::value, size_t>::type
    {
        return val;
    }

    template <class T, class U>
    constexpr auto revIndex(T val, U n) -> typename std::enable_if<!std::is_unsigned<T>::value, size_t>::type
    {
        return val < 0 ? n + val : val;
    }

    template <class T, uint8_t N>
    class Array
    {
        T m_data[N];

      public:
        MT_XINLINE Array(const Array& other)
        {
            for(uint8_t i = 0; i < N; ++i)
            {
                m_data[i] = other.m_data[i];
            }
        }
        MT_XINLINE Array(Array& other)
        {
            for(uint8_t i = 0; i < N; ++i)
            {
                m_data[i] = other.m_data[i];
            }
        }
        MT_XINLINE Array(Array&& other)
        {
            for(uint8_t i = 0; i < N; ++i)
            {
                m_data[i] = other.m_data[i];
            }
        }

        template <class... ARGS>
        MT_XINLINE constexpr Array(ARGS&&... args) : m_data{static_cast<T>(std::forward<ARGS>(args))...}
        {
        }

        MT_XINLINE Array<T, N>& operator=(const Array<T, N>& other) = default;
        MT_XINLINE Array<T, N>& operator=(Array<T, N>&& other) = default;

        MT_XINLINE T& operator[](int16_t idx) { return m_data[revIndex(idx, N)]; }
        MT_XINLINE const T& operator[](int16_t idx) const { return m_data[revIndex(idx, N)]; }

        MT_XINLINE constexpr uint8_t size() const { return N; }

        T* begin() { return m_data; }
        const T* begin() const { return m_data; }

        T* end() { return m_data + N; }
        const T* end() const { return m_data + N; }
    };
} // namespace mt

#include <ostream>

namespace std
{
    template <class T, uint8_t N>
    ostream& operator<<(ostream& os, const mt::Array<T, N>& arr)
    {
        for (uint8_t i = 0; i < N; ++i)
        {
            if (i != 0)
            {
                os << ' ';
            }
            os << arr[i];
        }
        return os;
    }
} // namespace std

#endif // MINITENSOR_ARRAY_HPP