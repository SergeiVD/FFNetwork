#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <algorithm>
#include <vector>

namespace mlm
{
	template <typename T>
	class matrix
	{
	public:
		matrix() = default;
		matrix(const int& rows, const int& cols) : rows_{rows}, cols_{cols}, data_{new T[rows* cols]()}{}
	
		matrix(const std::vector<std::vector<T>> data) : rows_{ data.size() }, cols_{ data[0].size() }
			{
				data_ = new T[rows_ * cols_]();
				for (size_t i = 0; i < rows_; i++)
				{
					for (size_t j = 0; j < cols_; j++)
					{
						data_[cols_ * i + j] = data[i][j];
					}
				}
			}


		matrix(const matrix& other) : rows_(other.rows_), cols_(other.cols_), data_(new T[rows_ * cols_])
			{
				std::copy(other.data_, other.data_ + (rows_ * cols_), data_);
			}
		matrix(matrix&& other) noexcept : rows_(other.rows_), cols_(other.cols_), data_(other.data_)
			{
				other.rows_ = 0;
				other.cols_ = 0;
				other.data_ = nullptr;
			}

		virtual ~matrix()
			{
				delete[] data_;
			}

		matrix& operator=(const matrix& other)
			{
				if(this != &other)
				{
					matrix temp(other);
					std::swap(rows_, temp.rows_);
					std::swap(cols_, temp.cols_);
					std::swap(data_, temp.data_);
				}
				return *this;
			}

		matrix& operator=(matrix&& other) noexcept
			{
				if(this != &other)
				{
					delete[] data_;
					rows_ = other.rows_;
					cols_ = other.cols_;
					data_ = other.data_;
					other.rows_ = 0;
					other.cols_ = 0;
					other.data_ = nullptr;
				}
				return * this;
			}

		T& operator()(int row, int col)
			{
				return data_[rows_ * row + col];
			}

		const T& operator()(int row, int col)const
			{
				return data_[rows_ * row + col];
			}


		struct Proxy
		{
			T* row;
			int cols;

			T& operator[](int col)
				{
					if(col >= 0 && col < cols)
					{
						return row[col];
					}
					else
					{
						std::cout<<"Invalid index"<<std::endl;
						static T dummy;
						return dummy;
					}
				}
		};

		Proxy operator[](int row)
			{
				if(row >= 0 && row < rows_)
				{
				
					return {data_ + row * cols_, cols_};
				}
				else
				{
					std::cout<<"Invalid index"<<std::endl;
					static T dummy_row;
					return {&dummy_row, cols_};
				}
			}

		const std::size_t rows()const {return rows_;}
		const std::size_t cols()const {return cols_;}

	private:
		int rows_{0};
		int cols_{0};
		T* data_{nullptr};
	};


	using matrixd = matrix<double>;
	
}



#endif /* MATRIX_H */
