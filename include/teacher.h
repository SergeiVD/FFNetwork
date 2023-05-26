#ifndef TEACHER_H
#define TEACHER_H

#include "layer.h"
#include "neuralnetwork.h"
#include "../include/activation_fn.h"
#include <stdexcept>
#include <vector>

#include <string>

class Teacher {
public:
  Teacher(NeuralNetwork& network,
		  std::vector<std::vector<double>>& training_data,
		  std::vector<std::vector<double>>& target_data,
          double learning_rate = 0.1,
		  int num_epochs = 1000) :
    network_{network},
    training_data_{training_data},
    target_data_{target_data},
    learning_rate_{learning_rate},
    num_epochs_{num_epochs}
  {}

	void train()
		{
			for (int epoch = 0; epoch < num_epochs_; ++epoch)
			{
				double epoch_error = 0.0;
				for (int i = 0; i < training_data_.size(); ++i)
				{
					auto output = network_.feed_forward(training_data_[i]);

					// output_ = network_.feed_forward(training_data_[i]);
					
					// auto error = target_data_[i] - output;
					auto error = subtraction_vec(target_data_[i], output);  // вычитание векторов
					epoch_error += calculate_error(error);

					// auto gradients = backpropagate(error);
					auto gradients = backprop(error);
					update_weights(gradients);
				}
				epoch_error /= training_data_.size();
				std::cout << "Epoch " << epoch+1 << ", Error = " << epoch_error << std::endl;
			}
		}

private:
	NeuralNetwork& network_;
  
	std::vector<std::vector<double>>& training_data_;
	std::vector<std::vector<double>>& target_data_;
	double learning_rate_;
	int num_epochs_;

	// Вычитание векторов
	std::vector<double> subtraction_vec(const std::vector<double>& vec_1, const std::vector<double>& vec_2)
		{
			if (vec_1.size() != vec_2.size())throw std::runtime_error("no correct size vector1: " + std::to_string(vec_1.size()) + " size vector 2: " + std::to_string(vec_2.size()));
			std::vector<double> rez(vec_1.size());
			for (size_t i = 0; i < vec_1.size(); ++i)
			{
				rez[i] = vec_1[i]-vec_2[i];
			}
			return rez;
		}

	// Умножение векторов
	std::vector<double> multiplication_vec(const std::vector<double>& vec_1, const std::vector<double>& vec_2)
		{
			if (vec_1.size() != vec_2.size())throw std::runtime_error("multiplicate fun no correct size vector1: " + std::to_string(vec_1.size()) + " size vector 2: " + std::to_string(vec_2.size()));
			
			std::vector<double> rez(vec_1.size());
			
			for (size_t i = 0; i < vec_1.size(); ++i)
			{
				rez[i] = vec_1[i] * vec_2[i];
			}
			return rez;
			
		}

	// Умножение матрицы на вектор
	std::vector<double> multiplication_mat_vec(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec)
		{
			if (mat.size() != vec.size())throw std::runtime_error("No correct length matrix: " + std::to_string(mat.at(0).size()) + " and length vector: " + std::to_string(vec.size()));
			
			std::vector<double>rez(vec.size());
			
			for (size_t i = 0; i < mat.size(); ++i)
			{
				double sum = 0;
				for (size_t j = 0; j < vec.size(); ++j)
				{
					sum += mat[i][j] * vec[j];
				}
				rez[i]=sum;
			}
			return rez;
		}
	
	// Расчет функции ошибки
	double calculate_error(const std::vector<double>& error)
		{
			double sum_sq = 0.0;
			for (auto e : error) {
				sum_sq += e * e;
			}
			return 0.5 * sum_sq;
		}

  // Обратное распространение ошибки
	// std::vector<std::vector<std::vector<double>>> backpropagate(const std::vector<double>& error)
	// 	{
	// 		std::vector<std::vector<std::vector<double>>> gradients;
	// 		auto last_layer = network_.get_layers().back();

	// 		// Вычисление градиента для последнего слоя
	// 		// auto delta = last_layer.derive_activation(last_layer.get_output()) * error;
			
	// 		auto grad_weights = calculate_grad_weights(delta, last_layer.get_input());
	// 		auto grad_bias = calculate_grad_bias(delta);
			
	// 		gradients.push_back(grad_weights);
	// 		gradients.push_back(grad_bias);

	// 		// Вычисление градиентов для остальных слоев
	// 		for (int i = network_.get_layers().size()-2; i >= 0; --i) {
	// 			auto layer = network_.get_layers()[i];
	// 			auto next_layer = network_.get_layers()[i+1];

	// 			delta = layer.derive_activation(layer.get_output()) * (next_layer.get_weights().transpose() * delta);
	// 			grad_weights = calculate_grad_weights(delta, layer.get_input());
	// 			grad_bias = calculate_grad_bias(delta);
	// 			gradients.insert(gradients.begin(), grad_bias);
	// 			gradients.insert(gradients.begin(), grad_weights);
	// 		}

	// 		return gradients;
	// 	}

	std::vector<std::vector<std::vector<double>>> backprop(const std::vector<double>& error)
		{	
			std::vector<std::vector<std::vector<double>>> gradients;
			
			int last_layer = network_.get_layers().size()-1;

			// Вычисление градиента для последнего слоя
			std::vector<double> delta = multiplication_vec(network_.layer_derivative(last_layer), error);

			std::vector<double> input_last_layer{network_.get_output_layers(last_layer-1)}; // Выход предполседного слоя является входом последнего!

			auto grad_weights = calc_grad_weights(delta, input_last_layer);
			auto grad_bias = calculate_grad_bias(delta);
			
			gradients.push_back(grad_weights);
			gradients.push_back(grad_bias);


			std::cout<<"net_size: "<<network_.get_layers().size() << std::endl;
			std::cout<<"net_size-2: "<<network_.get_layers().size()-2 << std::endl;
			
		// 	// Вычисление градиентов для остальных слоев
			// for (int i = network_.get_layers().size()-2; i >= 0; --i)
			for (int i = last_layer - 1; i >= 0; --i)
			{
				Layer layer = network_.get_layers()[i];
				Layer next_layer = network_.get_layers()[i+1];

				std::vector<std::vector<double>> weights_next_layer = network_.get_weights().at(i+1);

				std::cout<<"num_neurons: "<<weights_next_layer.size()<<std::endl;
				std::cout<<"num_weights: "<<weights_next_layer.at(i).size()<<std::endl;

				// delta = layer.derive_activation(layer.get_output()) * (next_layer.get_weights().transpose() * delta);
				// delta = layer.derive_activation(multiplication_vec(layer.get_output(), multiplication_mat_vec(next_layer.get_weights(), delta)));
				
				// delta =  multiplication_vec(network_.layer_derivative(i),multiplication_mat_vec(next_layer.get_weights(),delta));

				delta = multiplication_vec(network_.layer_derivative(i), multiplication_mat_vec(weights_next_layer, delta));


				// delta = layer.derive_activation(layer.get_output()) * multiplication_mat_vec(next_layer.get_weights(), delta);

				grad_weights = calc_grad_weights(delta, network_.get_output_layers(i));
				grad_bias = calculate_grad_bias(delta);
				gradients.insert(gradients.begin(), grad_bias);
				gradients.insert(gradients.begin(), grad_weights);
			}

			// return gradients;
			return gradients;
		}

	// Вычисление градиента для весов
	std::vector<std::vector<double>> calc_grad_weights(const std::vector<double>& delta, const std::vector<double>& input)
		{
			std::vector<std::vector<double>> grad_weights;

			for (int i = 0; i < delta.size(); ++i)
			{
				std::vector<double> grad_ij;
				for (int j = 0; j < input.size(); ++j)
				{
					grad_ij.push_back(delta[i] * input[j]);
				}
				grad_weights.push_back(grad_ij);
			}

			return grad_weights;
		}


	
  // // Вычисление градиента для весов
  // 	std::vector<std::vector<std::vector<double>>> calculate_grad_weights(const std::vector<double>& delta, const std::vector<double>& input)
  // 		{
  // 			std::vector<std::vector<std::vector<double>>> grad_weights;

  // 			for (int i = 0; i < delta.size(); ++i)
  // 			{
  // 				std::vector<std::vector<double>> grad_weight_i;

  // 				for (int j = 0; j < input.size(); ++j)
  // 				{
  // 					std::vector<double> grad_ij;
  // 					grad_ij.push_back(delta[i] * input[j]);
  // 					grad_weight_i.push_back(grad_ij);
  // 				}

  // 				grad_weights.push_back(grad_weight_i);
  // 			}

  // 			return grad_weights;
  // 		}

  // Вычисление градиента для смещений
	std::vector<std::vector<double>> calculate_grad_bias(const std::vector<double>& delta)
		{
			std::vector<std::vector<double>> grad_bias;

			for (int i = 0; i < delta.size(); ++i)
			{
				std::vector<double> grad_i;
				grad_i.push_back(delta[i]);
				grad_bias.push_back(grad_i);
			}

			return grad_bias;
		}

  // Обновление весов и смещений
	void update_weights(const std::vector<std::vector<std::vector<double>>>& gradients)
		{
			auto layers = network_.get_layers();
			for (int i = 0; i < layers.size(); ++i)
			{
				auto layer = layers[i];
				auto grad_weights = gradients[i*2];
				auto grad_bias = gradients[i*2 + 1];

				// Обновление весов
				for (int j = 0; j < layer.get_num_outputs(); ++j)
				{
					for (int k = 0; k < layer.get_num_inputs(); ++k) {
						layer.set_weights(j, k, layer.get_weights().at(j).at(k) + learning_rate_ * grad_weights[j][k]);
					}
				}

				// Обновление смещений
				for (int j = 0; j < layer.get_num_outputs(); ++j)
				{
					layer.set_bias(j, layer.get_biases().at(j) + learning_rate_ * grad_bias[j][0]);
				}
			}

		}
};


#endif /* TEACHER_H */
