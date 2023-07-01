#include "../include/l_teacher.h"
#include "../include/activation_fn.h"

#include <iostream>
#include <chrono>


L_Teacher::L_Teacher(NeuralNetwork& network,
					 const std::vector<std::vector<double>>& trainingData,
					 const std::vector<std::vector<double>>& targetData,
					 double learningRate,
					 int numEpochs)
    :
	network_(network),
	trainingData_(trainingData),
	targetData_(targetData),
	learningRate_(learningRate),
	numEpochs_(numEpochs)
{
}

void L_Teacher::train()
{
	
	for (int epoch = 0; epoch < numEpochs_; ++epoch)
	{
		auto start_time = std::chrono::high_resolution_clock::now(); // Измеряем время обучения одной эпохи (для теста)

		double epochError = 0.0;

		for (size_t i = 0; i < trainingData_.size(); ++i)
		{
			mlm::vecd output = network_.feedForward(trainingData_[i]);

			mlm::vecd error(output.size());
            // Вычисляем ошибку для каждого элемента выходного вектора
			for (size_t j = 0; j < targetData_[i].size(); ++j)
			{
				error[j] = targetData_[i][j] - output[j];
			}
            // Вычисляем сумму квадратов ошибок элементов выходного вектора
			for (size_t j = 0; j < error.size(); ++j)
			{
				epochError += error[j] * error[j];
			}
            // Применяем обратное распространение ошибки и получаем градиенты
			mlm::vec<mlm::matrixd> gradients = backpropagate(error);
			// Обновляем веса сети на основе градиентов
			updateWeights(gradients);
		}
		// Нормализуем суммарную ошибку на размер обучающего набора данных
		epochError /= trainingData_.size();

		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

		// Выводим номер эпохи, значение ошибки и время обучения
		std::cout << "Epoch " << epoch + 1 << ", Error = " << epochError << " : "<<duration.count()<<" milliseconds"<< std::endl;
	}

}

mlm::vec<mlm::matrixd> L_Teacher::backpropagate(const mlm::vecd& error)
{
	const std::size_t num_layers = network_.get_num_layers();

	mlm::vec<mlm::matrixd> gradients(num_layers);

	mlm::vecd delta = error;

	const int out_layer_index = num_layers - 1;

	const Layer& out_layer = network_.getLayer(out_layer_index);

	// Вычисляем градиенты для выходного слоя

	for (size_t i = 0; i < delta.size(); ++i)
	{
		delta[i] *= out_layer.derive_activation(i); // Умножаем ошибку на производную активационной функции выходного слоя
	}

	const mlm::matrixd& weights = out_layer.get_weights();

	mlm::matrixd weight_gradient(weights.rows(), weights.cols() + 1);

	for (size_t i = 0; i < weight_gradient.rows(); ++i)
	{
		for (size_t j = 0; j < weight_gradient.cols(); ++j)
		{
			if (j < weight_gradient.cols() - 1)
			{
				weight_gradient(i, j) = delta[i] * out_layer.get_input()[j] + 0.1 * weight_gradient(i, j); // Вычисляем частные производные весов выходного слоя
			}
			else
			{
				weight_gradient(i, j) = delta[i]; // Вычисляем частные производные смещений выходного слоя
			}
		}
	}

	gradients[out_layer_index] = weight_gradient; // Сохраняем градиенты для выходного слоя

	// Вычисляем градиенты для скрытых слоев

	for (int layer_index = out_layer_index - 1; layer_index >= 0; --layer_index)
	{
		const Layer& layer = network_.getLayer(layer_index);
		const Layer& previous_layer = network_.getLayer(layer_index + 1);
		mlm::vecd delta_layer(layer.get_output().size());
		const mlm::matrixd& weights = previous_layer.get_weights();

		mlm::matrixd weight_gradient(layer.get_weights().rows(), layer.get_weights().cols() + 1);

		// Вычисляем градиенты для текущего слоя

		for (size_t i = 0; i < delta_layer.size(); ++i)
		{
			double sum{ 0 };

			// Суммируем взвешенные ошибки из следующего слоя

			for (size_t j = 0; j < previous_layer.get_output().size(); ++j)
			{
				sum += delta[j] * weights(j, i);
			}

			delta_layer[i] = sum * layer.derive_activation(i); // Умножаем сумму ошибок на производную активационной функции текущего слоя
		}

		for (size_t i = 0; i < weight_gradient.rows(); ++i)
		{
			for (size_t j = 0; j < weight_gradient.cols(); ++j)
			{
				if (j < weight_gradient.cols() - 1)
				{
					weight_gradient(i, j) = delta_layer[i] * layer.get_input()[j] + 0.1 * weight_gradient(i, j); // Вычисляем частные производные весов текущего слоя
				}
				else
				{
					weight_gradient(i, j) = delta_layer[i]; // Вычисляем частные производные смещений текущего слоя
				}
			}
		}

		gradients[layer_index] = weight_gradient; // Сохраняем градиенты для текущего слоя

		delta = delta_layer; // Обновляем ошибку для следующей итерации
	}

	return gradients; // Возвращаем вычисленные градиенты для всех слоев сети
}


void L_Teacher::updateWeights(const mlm::vec<mlm::matrixd>& gradients)
{
	size_t network_num_layers{ network_.get_num_layers() };

	for (size_t i = 0; i < network_num_layers; ++i)
	{
		Layer& layer = network_.getLayer(i);

		const mlm::matrixd& weight_gradients = gradients[i];

		// Обновляем веса и смещения для каждого слоя

		for (size_t j = 0; j < weight_gradients.rows(); ++j)
		{
			for (size_t k = 0; k < weight_gradients.cols(); ++k)
			{
				if (k < weight_gradients.cols() - 1)
				{
					layer.update_weights(j, k, learningRate_ * weight_gradients(j,k));
				}
				else
				{
					layer.update_bias(j, learningRate_ * weight_gradients(j,k));
				}

			}
		}
	}
}
