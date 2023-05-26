#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>

enum class activate_fn
{
	sigmoid,
	relu,
	tanh
};


class Layer
{
public:
	Layer()=default;
	Layer(int input_size, int output_size, activate_fn activ_fn);

	Layer(const Layer& layer);
	Layer(const Layer&& layer)noexcept;

	Layer& operator=(const Layer& layer);
	Layer& operator=(Layer&& layer);

	std::vector<double> activate(const std::vector<double>& input);
	// void activate(const std::vector<double>& input);
	void set_weights(const int& num_neuron, const int& num_weigth, const double& weigth);
	void set_bias(const int& num_neuron, const double& bias);

	std::vector<std::vector<double>> get_weights(){return weights_;}
	std::vector<double> get_biases(){return biases_;}
	std::vector<double> get_input(){return input_;}
	int get_num_outputs(){return output_size_;};
	int get_num_inputs(){return input_size_;}

	std::vector<double> derive_activation(const std::vector<double>& val);
	
	activate_fn get_name_activate_fn(){return activ_fn_;}

private:
    int input_size_;
    int output_size_;
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;

	std::vector<double> input_;
	
	activate_fn activ_fn_;
};


//     Layer(int input_size, int output_size) :
//         input_size_{input_size},
//         output_size_{output_size},
//         weights_(output_size, std::vector<double>(input_size)),
//         biases_(output_size)
//     {
//         // Инициализация весов и смещений
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_real_distribution<> dis(-1.0, 1.0);
        
//         for (int i = 0; i < input_size_; i++)
//         {
//             for (int j = 0; j < output_size_; j++)
//             {
//                 weights_[j][i] = dis(gen); // Инициализация случайными значениями от -1 до 1
//             }
//         }

//         for (int i = 0; i < output_size_; i++)
//         {
//             biases_[i] = dis(gen); // Инициализация случайными значениями от -1 до 1
//         }
//     }

//     std::vector<double> activate(const std::vector<double>& input, const activate_fun& function)
//     {
//         // std::unique_ptr<double[]> output(new double[output_size_]);
// 		std::vector<double> output(output_size_);
		
//         // Умножение входа на веса и добавление смещений
//         for (int j = 0; j < output_size_; j++)
//         {
//             double weighted_sum = 0.0;
//             for (int i = 0; i < input_size_; i++)
//             {
//                 weighted_sum += input[i] * weights_[j][i];
//             }
//             switch (function)
//             {
//                 case activate_fun::sigmoid:
//                     output[j] = sigmoid(weighted_sum + biases_[j]);
//                     break;
//                 case activate_fun::tanh:
//                     output[j] = tanh(weighted_sum + biases_[j]);
//                     break;
//                 case activate_fun::relu:
//                     output[j] = relu(weighted_sum + biases_[j]);
//                     break;
//             }
//         }

//         return output;
//     }

// 	void set_weigths(const int& num_neuron, const int& num_weigth, const double& weigth)
// 		{
// 			weights_.at(num_neuron).at(num_weigth) = weigth;
// 		}

// 	void set_bias(const int& num_neuron, const double& bias)
// 		{
// 			biases_.at(num_neuron) = bias;
// 		}

// 	std::vector<std::vector<double>> get_weigths(){return weights_;}
// 	std::vector<double> get_biases(){return biases_;}

// private:
//     int input_size_;
//     int output_size_;
//     std::vector<std::vector<double>> weights_;
//     std::vector<double> biases_;

//     double sigmoid(double x)
// 		{
// 			return 1.0 / (1.0 + exp(-x));
// 		}

//     double tanh(double x)
// 		{
// 			return std::tanh(x);
// 		}

//     double relu(double x)
// 		{
// 			return std::fmax(0.0, x);
// 		}

// 	double sigmoid_derivative(double x)
// 		{
// 			double s = sigmoid(x);
// 			return s * (1.0 - s);
// 		}

// 	double tanh_derivative(double x)
// 		{
// 			double t = std::tanh(x);
// 			return 1.0 - (t * t);
// 		}

// 	double relu_derivative(double x)
// 		{
// 			if (x > 0) {
// 				return 1;
// 			} else {
// 				return 0;
// 			}
// 		}

// };



#endif /* LAYER_H */
