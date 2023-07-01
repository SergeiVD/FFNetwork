#ifndef VEC_H
#define VEC_H

#include <algorithm>
#include <bits/c++config.h>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <cmath>

#include <vector>

namespace mlm
{
	template<typename T>
	class vec
	{
	public:
		vec() = default;

		explicit vec(const std::size_t& size) : size_{size}, data_ { new T[size] }
			{
				for (size_t i = 0; i < size_; i++)
				{
					data_[i] = T();
				}
			}

		vec(const std::size_t& size, const T& val) : size_{size}, data_{new T[size]}
			{
				for (size_t i = 0; i < size_; ++i)
				{
					data_[i] = val;
				}

			}

		vec(const std::vector<T>& v) : size_{ v.size() }, data_{new T[size_]}
			{
				std::copy(v.begin(), v.end(), data_);
			}
		
		vec(const mlm::vec<T>& other) : size_{other.size_}, data_{new T[size_] }
			{
				std::copy(other.data_, other.data_ + other.size_, data_);
			}

		mlm::vec<T>& operator=(const mlm::vec<T>& v)
			{
				if (this != &v)
				{
					delete[] data_;
					data_ = new T[v.size_];
					size_ = v.size_;
					std::copy(v.data_, v.data_ + v.size_, data_);
					return *this;
				}
				return *this;
			}

		vec(vec&& other) noexcept : size_ (other.size_), data_(std::move(other.data_))
			{
				other.size_ = 0;
				other.data_ = nullptr;
			}

		vec& operator=(vec&& other) noexcept
			{
				if (this != &other)
				{
					delete[] data_;
					size_ = other.size_;
					data_ = std::move(other.data_);
					other.size_ = 0;
					other.data_ = nullptr;
				}
				return *this;
			}

		virtual ~vec() { delete[] data_; size_= 0; };
		
		T& operator[](size_t index) {return data_[index];}
		const T& operator[](size_t index) const{return data_[index];}

		vec<T>& operator+=(const vec<T>& other);
        vec<T>& operator-=(const vec<T>& other);
        vec<T>& operator*=(T scalar);
        vec<T>& operator/=(T scalar);

		const std::size_t& size()const{return size_;}
		const T* data()const {return data_;}
		T* begin()const{return data_;}
		T* end()const{return data_+size_;}
		void resize(const std::size_t& new_size)
			{
				delete[] data_;
				data_ = new T[new_size];
				size_ = new_size;
			}
		const T& back()const {return data_[size_ -1];}
		

		T length();

		void list()const
			{
				std::cout<<"vec([";
				for (size_t i = 0; i < size_; ++i)
				{
					std::cout<<data_[i];
					if(i != size_-1)std::cout<<", ";
				}
				std::cout<<"])"<<std::endl;
			}

        void printAsTensor() const
			{
				std::cout << "tensor([";
				for (size_t i = 0; i < size(); ++i)
				{
					if (i == 0)std::cout << "[";
					std::cout << data_[i];
					if (i < size() - 1)
					{
						std::cout << ", ";
					}
					else
					{
						std::cout << "]";
					}
					//std::cout << "]";
				}
				std::cout << "])" << std::endl;

			}
		
	private:
		std::size_t size_{0};
		T* data_{nullptr};

	};


    template<typename T>
    vec<T> operator+(const vec<T>& v1, const vec<T>& v2) { return vec<T>(v1) += v2; }

    template<typename T>
    vec<T> operator-(const vec<T>& v1, const vec<T>& v2) { return vec<T>(v1) -= v2; }

    template<typename T>
    vec<T> operator*(const vec<T>& v, T scalar) { return vec<T>(v) *= scalar; }

    template<typename T>
    vec<T> operator*(T scalar, const vec<T>& v) { return vec<T>(v) *= scalar; }

    template<typename T>
    vec<T> operator/(const vec<T>& v, T scalar) { return vec<T>(v) /= scalar; }

	
	template<typename T>
	vec<T>& vec<T>::operator+=(const vec<T>& other)
	{
		if (size() != other.size())
		{
			throw std::invalid_argument("Vectors must have the same size");
		}
		for (size_t i = 0; i < size(); ++i)
		{
			data_[i] += other[i];
		}
		return *this;
	}

	template<typename T>
	vec<T>& vec<T>::operator-=(const vec<T>& other)
	{
		if (size() != other.size()) {
			throw std::invalid_argument("Vectors must have the same size");
		}
		for (size_t i = 0; i < size(); ++i)
		{
			data_[i] -= other[i];
		}
		return *this;
	}

	template<typename T>
	vec<T>& vec<T>::operator*=(T scalar)
	{
		for (size_t i = 0; i < size(); ++i)
		{
			data_[i] *= scalar;
		}
		return *this;
	}

	template<typename T>
	vec<T>& vec<T>::operator/=(T scalar)
	{
		for (size_t i = 0; i < size(); ++i)
		{
			data_[i] /= scalar;
		}
		return *this;
	}

	template<typename T>
	T vec<T>::length()
	{
		T sum = T();
		for (size_t i = 0; i < this->size_; ++i)
		{
			sum += std::pow(data_[i], 2);
		}
		return std::sqrt(sum);
	}

	template<typename T>
    T dot_product(const vec<T>& v1, const vec<T>& v2)
	{
        if (v1.size() != v2.size())
		{
            throw std::invalid_argument("Vectors must have the same size");
        }
        T result = 0;
        for (size_t i = 0; i < v1.size(); ++i)
		{
            result += v1[i] * v2[i];
        }
        return result;
    }

	template<typename T>
	vec<T> normalize(const vec<T>& v)
	{
		T mag = v.length();
		if(mag == 0)throw std::invalid_argument("Cannot normalize the zero vector");
		return v/mag;
	}

	template<typename T>
	vec<T> cross_product(const vec<T>& v1,const vec<T>& v2)
	{
		if (v1.size() != v2.size())throw std::invalid_argument("error size mlm::vec cross_product mlm::vec");
		if (v1.size() < 2 || v1.size() > 3)throw std::invalid_argument("error cross_product is defined only 2D and 3D vectors");

		vec<T> result(v1.size());
		if(v1.size() == 2)
		{
			result[0] = T();
			result[1] = (v1[0] * v2[1]) - (v1[1] * v2[0]);
		}
		else
		{
			result[0] = v1[1]*v2[2]-v1[2]*v2[1];
			result[1] = v1[2]*v2[0]-v1[0]*v2[2];
			result[2] = v1[0]*v2[1]-v1[1]*v2[0];

			for (size_t i = 3; i < v1.size(); ++i)
			{
				result[i] = T();
				for (size_t j = 0; j < v1.size(); ++j)
				{
					size_t k = (j + 1) % v1.size();
					size_t l = (j + 2) % v1.size();
					result[i] = v1[j] * v2[k] - v1[k] * v2[j];
				}
			}
		}
		return result;
	}

		


	using vecd = vec<double>;
	using veci = vec<int>;
	

};


#endif /* VEC_H */
