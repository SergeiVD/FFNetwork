#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <bits/c++config.h>
#include <iterator>
#include <initializer_list>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <sys/types.h>
#include <vector>
#include <tuple>
#include "vec.h"

namespace mlm
{
	
	template<typename T>
	class matrix
	{
	public:
		matrix()=default;

		matrix(const int& rows, const int& cols) : rows_{ rows }, cols_{ cols }
			{
				data_ = new T * [rows_];
				for (size_t i = 0; i < rows_; i++)
				{
					data_[i] = new T[cols_];
					for (size_t j = 0; j < cols_; j++)
					{
						data_[i][j] = T();
					}
				}
			}

		// matrix(const std::initializer_list<std::initializer_list<T>> data) : rows_{ data.size() }, cols_{ data[0].size() }
		// 	{
		// 		data_ = new T * [rows_];
		// 		for (size_t i = 0; i < rows_; i++)
		// 		{
		// 			data_[i] = new T[cols_];
		// 			for (size_t j = 0; j < cols_; j++)
		// 			{
		// 				data_[i][j] = data[i][j];
		// 			}
		// 		}
		// 	}


		matrix(const std::vector<std::vector<T>> data) : rows_{ data.size() }, cols_{ data[0].size() }
			{
				data_ = new T * [rows_];
				for (size_t i = 0; i < rows_; i++)
				{
					data_[i] = new T[cols_];
					for (size_t j = 0; j < cols_; j++)
					{
						data_[i][j] = data[i][j];
					}
				}
			}


		matrix(const matrix& other) : rows_{ other.rows_ }, cols_{ other.cols_ }
			{
				data_ = new T * [rows_];
				for (size_t i = 0; i < rows_; i++)
				{
					data_[i] = new T[cols_];
					for (size_t j = 0; j < cols_; j++)
					{
						data_[i][j] = other.data_[i][j];
					}
				}
			}

		matrix& operator=(const matrix& other)
			{
				if(this != &other)
				{
					for (size_t i = 0; i < rows_; i++)
					{
						delete[] data_[i];
					}
					delete[] data_;

					rows_ = other.rows();
					cols_ = other.cols();

					data_ = new T * [other.rows_];
					for (size_t i = 0; i < other.rows_; i++)
					{
						data_[i] = new T[other.cols_];
						for (size_t j = 0; j < other.cols_; j++)
						{
							data_[i][j] = other.data_[i][j];
						}
					}
					
				}
				return *this;
			}

		virtual ~matrix()
			{
				for (size_t i = 0; i < rows_; i++)
				{
					delete[] data_[i];
				}
				delete[] data_;
			}

		
		std::size_t rows()const { return rows_; }
		std::size_t cols()const { return cols_; }
		T* operator[](const int& row) { return data_[row]; }
		const T* operator[](const int& row)const { return data_[row]; }

		// const T** data()const{return data_;}

		matrix<T>& operator+=(const matrix<T>& other);
        matrix<T>& operator-=(const matrix<T>& other);
		matrix<T>& operator*=(const matrix<T>& other);
		
        matrix<T>& operator*=(T scalar);
        matrix<T>& operator/=(T scalar);
		matrix<T>& operator-=(T scalar);
		matrix<T>& operator+=(T scalar);
		
		// Print matrix to console
		void list()const
			{
				std::cout<<"matrix[";
				for (size_t i = 0; i < rows_; ++i)
				{
					if(i != 0)std::cout<<"       ";
					for (size_t j = 0; j < cols_; ++j)
					{
						std::cout<<data_[i][j];
						if(j != cols_-1)std::cout<<", ";
					}

					if(i == rows_-1)
					{
						std::cout<<"]";	
					}
					else
					{
						std::cout<<std::endl;
					}
				}
				std::cout<<std::endl;
			}

		//* Matrix filling
		void fill(const T& val)
			{
				for (size_t i = 0; i < rows_; i++)
				{
					for (size_t j = 0; j < cols_; j++)
					{
						data_[i][j] = val;
					}
				}

			}

		//* Transpose matrix
		matrix transpose()const
			{
				matrix result(cols_, rows_);
				for (size_t i = 0; i < rows_; i++)
				{
					for (size_t j = 0; j < cols_; j++)
					{
						result[j][i] = data_[i][j];
					}
				}
				return result;
			}

		//* Creating an identity matrix
        static matrix<T> eye(size_t n)
			{
				matrix<T> result(n, n);
				result.fill(T());
				for (size_t i = 0; i < n; ++i)
				{
					result[i][i] = T(1);
				}
				return result;
			}

		T trace() const
			{
				T result = T();
				for (size_t i = 0; i < std::min(rows_, cols_); ++i)
				{
					result += (*this)(i, i);
				}
				return result;
			}

        T determinant() const
			{
				if (rows_ != cols_)
				{
					throw std::invalid_argument("matrix must be square to calculate determinant");
				}
				matrix<T> lu = this->lu_decomposition();
				T det = static_cast<T>(1);
				for (size_t i = 0; i < rows_; ++i)
				{
					det *= lu(i, i);
				}
				return det;
			}

		matrix<T> inverse() const
			{
			if (rows_ != cols_)
			{
				throw std::invalid_argument("Matrix must be square to invert");
			}

			// Create augmented matrix [A|I]
			matrix<T> inv(rows_, 2 * cols_);
			
			for (size_t i = 0; i < rows_; ++i)
			{
				std::copy_n(data_[i], cols_, inv[i]);
				inv[i][ i + cols_] = T(1);
			}

			// Gaussian-Jordan elimination
			for (size_t i = 0; i < rows_; ++i)
			{
				if (inv[i][i] == T(0)) throw std::runtime_error("Matrix is singular and cannot be inverted");
				
				for (size_t j = 0; j < rows_; ++j)
				{
					if (i != j)
					{
						T factor = inv[j][i] / inv[i][i];
						
						for (size_t k = i; k < 2 * cols_; ++k)
						{
							inv[j][k] = inv[j][k] - factor * inv[i][k];
						}
					}
				}
				
				T divisor = inv[i][i];
				for (size_t k = 0; k < 2*cols_; ++k)
				{
					inv[i][k] = inv[i][k]/divisor;
				}

			}

			// Extract inverse matrix [I|A^-1]
			matrix<T> matrix_result(rows_, cols_);
			
			for (size_t i = 0; i < rows_; ++i)
			{
				for (size_t j = 0; j < cols_; ++j)
				{
					matrix_result[i][j] = inv[i][cols_ + j];
				}

			}

			return matrix_result;
		}

		
        vec<T> solve(const vec<T>& b) const
			{
				if (rows_ != cols_)
				{
					throw std::invalid_argument("matrix must be square to solve linear system");
				}
				matrix<T> lu = this->lu_decomposition();
				vec<T> y = forward_substitution(lu, b);
				vec<T> x = backward_substitution(lu.transpose(), y);
				return x;
			}


		matrix<T> lu_decomposition() const
			{
				if (rows_ != cols_)
				{
					throw std::invalid_argument("matrix must be square to perform LU decomposition");
				}
				matrix<T> lu(rows_, cols_);
				for (size_t i = 0; i < rows_; ++i)
				{
					for (size_t j = 0; j < cols_; ++j)
					{
						if (i <= j)
						{
							T sum = T();
							for (size_t k = 0; k < i; ++k)
							{
								sum += lu[i][k] * lu[k][j];
							}
							lu[i][j] = data_[i][j] - sum;
						}
						else
						{
							T sum = T();
							for (size_t k = 0; k < j; ++k)
							{
								sum += lu[i][k] * lu[k][j];
							}
							lu[i][j] = (data_[i][j] - sum) / lu[j][j];			
						}
					}
				}
				return lu;
			}

		std::tuple<matrix<T>, matrix<T>> qr_decomposition() const 
			{
				matrix<T> q(rows_, cols_);
				q.fill(T());
				matrix<T> r = *this;
				for (size_t j = 0; j < cols_; ++j)
				{
					vec<T> a(rows_);
					for (size_t i = 0; i < rows_; ++i)
					{
						a[i] = r(i, j);
					}
					for (size_t i = 0; i < j; ++i)
					{
						T dot_product = T();
						for (size_t k = 0; k < rows_; ++k)
						{
							dot_product += r(k, j) * q(k, i);
						}
						for (size_t k = 0; k < rows_; ++k)
						{
							a[k] -= dot_product * q(k, i);
						}
					}
					T norm_squared = T();
					for (size_t i = 0; i < rows_; ++i)
					{
						norm_squared += a[i] * a[i];
					}
					T norm = std::sqrt(norm_squared);
					if (norm == T())
					{
						throw std::runtime_error("QR decomposition failed: matrix is rank-deficient");
					}
					for (size_t i = 0; i < rows_; ++i)
					{
						q(i, j) = a[i] / norm;
					}
					for (size_t i = j; i < cols_; ++i)
					{
						T dot_product = T();
						for (size_t k = 0; k < rows_; ++k)
						{
							dot_product += r(k, j) * q(k, i);
						}
						for (size_t k = 0; k < rows_; ++k)
						{
							r(k, i) -= dot_product * q(k, j);
						}
					}
				}
				return std::make_tuple(q, r);
			}

		
		size_t rank() const
			{
				matrix<T> rref = this->rref();
				size_t rank = 0;
				for (size_t i = 0; i < rows_; ++i)
				{
					bool is_zero_row = true;
					for (size_t j = 0; j < cols_; ++j)
					{
						if (rref(i, j) != T())
						{
							is_zero_row = false;
							break;
						}
					}
					if (!is_zero_row)
					{
						++rank;
					}
				}
				return rank;
			}


		
	private:
		T** data_{ nullptr };
		int rows_{ 0 };
		int cols_{ 0 };


		static T mod_inverse(const T& a, const T& p)
			{
				T t = T();
				T newt = T(1);
				T r = p;
				T newr = a;
				while (newr != 0) 
				{
					T quotient = r / newr;
					T temp = t;
					t = newt;
					newt = temp - quotient * newt;
					temp = r;
					r = newr;
					newr = temp - quotient * newr;
				}
				if (r > 1)
				{
					throw std::invalid_argument("Element has no inverse");
				}
				if (t < 0)
				{	
					t += p;
				}
				return t;
			}


		matrix<T> rref() const
			{
				matrix<T> result = *this;
				size_t lead = 0;
				for (size_t r = 0; r < rows_; ++r)
				{
					if (cols_ <= lead)
					{
						break;
					}
					size_t i = r;
					while (result(i, lead) == T())
					{
						++i;
						if (rows_ == i)
						{
							i = r;
							++lead;
							if (cols_ == lead)
							{
								break;
							}
						}
					}
					for (size_t j = 0; j < cols_; ++j)
					{
						std::swap(result(i, j), result(r, j));
					}
					T lv = result(r, lead);
					for (size_t j = 0; j < cols_; ++j)
					{
						result(r, j) /= lv;
					}
					for (size_t i = 0; i < rows_; ++i)
					{
						if (i != r)
						{
							T lv = result(i, lead);
							for (size_t j = 0; j < cols_; ++j)
							{
								result(i, j) -= lv * result(r, j);
							}
						}
					}
					++lead;
				}
				return result;
			}


        vec<T> forward_substitution(const matrix<T>& l, const vec<T>& b) const
			{
				vec<T> x(rows_);
				for (size_t i = 0; i < rows_; ++i)
				{
					T sum = T();
					for (size_t j = 0; j < i; ++j)
					{
						sum += l[i][j] * x[j];
					}
					x[i] = b[i] - sum;
				}
				return x;
			}

        vec<T> backward_substitution(const matrix<T>& u, const vec<T>& y) const
			{
				vec<T> x(rows_);
				for (ptrdiff_t i = rows_ - 1; i >= 0; --i)
				{
					T sum = T();
					for (size_t j = i + 1; j < cols_; ++j)
					{
						sum += u[i][j] * x[j];
					}
					x[i] = (y[i] - sum) / u[i][i];
				}
				return x;
			}
    };



	template<typename T>
    matrix<T> operator+(const matrix<T>& m1, const matrix<T>& m2) { return matrix<T>(m1) += m2; }

    template<typename T>
    matrix<T> operator-(const matrix<T>& m1, const matrix<T>& m2) { return matrix<T>(m1) -= m2; }

    template<typename T>
    matrix<T> operator*(const matrix<T>& m1, const matrix<T>& m2) { return matrix<T>(m1) *= m2; }

    template<typename T>
    matrix<T> operator*(const matrix<T>& m, T scalar) { return matrix<T>(m) *= scalar; }

    template<typename T>
    matrix<T> operator*(T scalar, const matrix<T>& m) { return matrix<T>(m) *= scalar; }

    template<typename T>
    matrix<T> operator/(const matrix<T>& m, T scalar) { return matrix<T>(m) /= scalar; }

	template<typename T>
    matrix<T> operator-(const matrix<T>& m, T scalar) { return matrix<T>(m) -= scalar; }

	template<typename T>
    matrix<T> operator+(const matrix<T>& m, T scalar) { return matrix<T>(m) += scalar; }

	
	template<typename T>
	matrix<T>& matrix<T>::operator+=(const matrix<T>& other)
	{
		if (rows_ != other.rows_ || cols_ != other.cols_)throw std::invalid_argument("Matrices must have the same size to be added");

		for (size_t i = 0; i < rows_; i++)
		{
			for (size_t j = 0; j < cols_; j++)
			{
				data_[i][j] += other[i][j];
			}
		}
		return *this;

	}

	template<typename T>
	matrix<T>& matrix<T>::operator-=(const matrix<T>& other)
	{
		if (rows_ != other.rows_ || cols_ != other.cols_)throw std::invalid_argument("Matrices must have the same size to be added");
		for (size_t i = 0; i < rows_; i++)
		{
			for (size_t j = 0; j < cols_; j++)
			{
				data_[i][j] -= other[i][j];
			}
		}
		return *this;
	}

	template<typename T>
	matrix<T>& matrix<T>::operator*=(const matrix<T>& other)
	{
		if (cols_ != other.rows_)throw std::invalid_argument("Matrices must have the same size to be added");
		matrix<T> result(rows_, other.cols());
		for (size_t i = 0; i < rows_; ++i)
		{
			for (size_t j = 0; j < other.cols(); ++j)
			{
				for (size_t k = 0; k < cols_; ++k)
				{
					result[i][j] += data_[i][k] * other[k][j];
				}
			}
		}

		*this = result;

		return *this;
	}


	template<typename T>
	matrix<T>& matrix<T>::operator-=(T scalar)
	{
		for (size_t i = 0; i < rows_; i++)
		{
			for (size_t j = 0; j < cols_; j++)
			{
				data_[i][j] -= scalar;
			}
		}
		return *this;
	}

	template<typename T>
	matrix<T>& matrix<T>::operator+=(T scalar)
	{
		for (size_t i = 0; i < rows_; i++)
		{
			for (size_t j = 0; j < cols_; j++)
			{
				data_[i][j] += scalar;
			}
		}
		return *this;
	}


	template<typename T>
	matrix<T>& matrix<T>::operator*=(T scalar)
	{
		for (size_t i = 0; i < rows_; i++)
		{
			for (size_t j = 0; j < cols_; j++)
			{
				data_[i][j] *=  scalar;
			}
		}
		return *this;
	}

	template<typename T>
	matrix<T>& matrix<T>::operator/=(T scalar)
	{
		for (size_t i = 0; i < rows_; i++)
		{
			for (size_t j = 0; j < cols_; j++)
			{
				data_[i][j] /=  scalar;
			}
		}
		return *this;		
	}

	template<typename T>
    vec<T> operator*(const matrix<T>& m, const vec<T>& v)
    {
        if (m.cols() != v.size()) throw std::invalid_argument("matrix and vector dimensions do not match");

        vec<T> result(m.rows());
        for (size_t i = 0; i < m.rows(); ++i)
		{
            T sum = T();
            for (size_t j = 0; j < m.cols(); ++j)
			{
                sum += m[i][j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    template<typename T>
    vec<T> operator*(const vec<T>& v, const matrix<T>& m)
    {
        if (v.size() != m.rows()) throw std::invalid_argument("Vector and matrix dimensions do not match");
		
        vec<T> result(m.cols());
        for (size_t j = 0; j < m.cols(); ++j)
		{
            T sum = T();
            for (size_t i = 0; i < m.rows(); ++i)
			{
                sum += v[i] * m[i][j];
            }
            result[j] = sum;
        }
        return result;
    }

	
	using matrixd = matrix<double>;
	


}    

#endif /* MATRIX_H */


















		// // Sum matrix
		// matrix operator+(const matrix& other)const
		// 	{
		// 		if (rows_ != other.rows_ || cols_ != other.cols_)throw std::invalid_argument("Matrices must have the same size to be added");
		// 		matrix result(rows_, cols_);
		// 		for (size_t i = 0; i < rows_; i++)
		// 		{
		// 			for (size_t j = 0; j < cols_; j++)
		// 			{
		// 				result[i][j] = data_[i][j] + other.data_[i][j];
		// 			}
		// 		}
		// 		return result;
		// 	}

		// // Minus matrix
		// matrix operator-(const matrix& other)const
		// 	{
		// 		if (rows_ != other.rows_ || cols_ != other.cols_)throw std::invalid_argument("Matrices must have the same size to be added");
		// 		matrix result(rows_, cols_);
		// 		for (size_t i = 0; i < rows_; i++)
		// 		{
		// 			for (size_t j = 0; j < cols_; j++)
		// 			{
		// 				result[i][j] = data_[i][j] - other.data_[i][j];
		// 			}
		// 		}
		// 		return result;
		// 	}

		// matrix operator*(const matrix& other)const
		// 	{
		// 		if(cols_ != other.rows_)throw std::invalid_argument("Number of columns in first matrix must match number of rows in second matrix to be multiplied");
				
		// 		matrix result(rows_,other.cols_);

		// 		for (size_t i = 0; i < rows_; ++i)
		// 		{
		// 			for (size_t j = 0; j < other.cols_; ++j)
		// 			{
		// 				int sum = 0;
		// 				for (size_t k = 0; k < cols_; ++k)
		// 				{
		// 					sum += data_[i][k] * other[k][j];
		// 				}
		// 				result[i][j] = sum;
		// 			}
		// 		}
		// 		return result;
		// 	}

		// mlm::vec<T> operator*(const mlm::vec<T>& vec)const
		// 	{
		// 		if(cols_ != vec.size())throw std::invalid_argument("Number of columns in first matrix must match number of rows in second matrix to be multiplied");
				
		// 		mlm::vec<T> result(rows_);

		// 		for (size_t i = 0; i < rows_; ++i)
		// 		{
		// 			T sum = T();
		// 			for (size_t j = 0; j < cols_; ++j)
		// 			{
		// 				sum += data_[i][j] * vec[j];
		// 			}
		// 			result[i] = sum;
		// 		}
		// 		return result;
		// 	}
