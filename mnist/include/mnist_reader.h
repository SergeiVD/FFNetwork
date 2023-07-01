#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <bits/stdint-uintn.h>

#include <iostream>
 #include <opencv2/opencv.hpp>

#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <cstdint>


#include "mnist_reader_common.h"

namespace mnist
{

/*!
 * \brief Represents a complete mnist dataset
 * \tparam Pixel The type of a pixel
 * \tparam Label The type of a label
 */
	template <typename Pixel = uint8_t, typename Label = uint8_t>
	struct MNIST_dataset
	{
		std::vector<std::vector<Pixel>> training_images; ///< The training images
		std::vector<std::vector<Pixel>> test_images;     ///< The test images
		std::vector<Label> training_labels;              ///< The training labels
		std::vector<Label> test_labels;                  ///< The test labels
	};

/*!
 * \brief Read a MNIST image file and return a container filled with the images
 * \param path The path to the image file
 * \return A std::vector filled with the read images
 */
	template <typename Pixel = uint8_t, typename Label = uint8_t>
	std::vector<std::vector<Pixel>> read_mnist_image_file(const std::string& path)
	{
		auto buffer = read_mnist_file(path, 0x803);

		if (buffer)
		{
			auto count   = read_header(buffer, 1);
			auto rows    = read_header(buffer, 2);
			auto columns = read_header(buffer, 3);

			//Skip the header
			//Cast to unsigned char is necessary cause signedness of char is
			//platform-specific
			auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

			std::vector<std::vector<Pixel>> images;
			images.reserve(count);

			for (size_t i = 0; i < count; ++i)
			{
				images.emplace_back(rows * columns);

				for (size_t j = 0; j < rows * columns; ++j)
				{
					auto pixel   = *image_buffer++;
					images[i][j] = static_cast<Pixel>(pixel);
				}
			}

			return images;
		}

		return {};
	}

/*!
 * \brief Read a MNIST label file and return a container filled with the labels
 * \param path The path to the image file
 * \return A std::vector filled with the read labels
 */
	template <typename Label = uint8_t>
	std::vector<Label> read_mnist_label_file(const std::string& path)
	{
		auto buffer = read_mnist_file(path, 0x801);

		if (buffer)
		{
			auto count = read_header(buffer, 1);

			//Skip the header
			//Cast to unsigned char is necessary cause signedness of char is
			//platform-specific
			auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

			std::vector<Label> labels(count);

			for (size_t i = 0; i < count; ++i)
			{
				auto label = *label_buffer++;
				labels[i]  = static_cast<Label>(label);
			}

			return labels;
		}

		return {};
	}

/*!
 * \brief Read all training images and return a container filled with the images.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \return Container filled with the images
 */
	template <typename Pixel = uint8_t>
	std::vector<std::vector<Pixel>> read_training_images()
	{
		return read_mnist_image_file<Pixel>("../mnist/train-images-idx3-ubyte");
	}

/*!
 * \brief Read all test images and return a container filled with the images.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \return Container filled with the images
 */
	template <typename Pixel = uint8_t>
	std::vector<std::vector<Pixel>> read_test_images()
	{
		return read_mnist_image_file<Pixel>("../mnist/t10k-images-idx3-ubyte");
	}

/*!
 * \brief Read all training labels and return a container filled with the labels.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \return Container filled with the labels
 */
	template <typename Label = uint8_t>
	std::vector<Label> read_training_labels()
	{
		return read_mnist_label_file<Label>("../mnist/train-labels-idx1-ubyte");
	}

/*!
 * \brief Read all test labels and return a container filled with the labels.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \return Container filled with the labels
 */
	template <typename Label = uint8_t>
	std::vector<Label> read_test_labels()
	{
		return read_mnist_label_file<Label>("../mnist/t10k-labels-idx1-ubyte");
	}

/*!
 * \brief Read dataset.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \return The dataset
 */
	template <typename Pixel = uint8_t, typename Label = uint8_t>
	MNIST_dataset<Pixel, Label> read_dataset()
	{
		MNIST_dataset<Pixel, Label> dataset;

		dataset.training_images = read_training_images<Pixel>();
		dataset.training_labels = read_training_labels<Label>();

		dataset.test_images = read_test_images<Pixel>();
		dataset.test_labels = read_test_labels<Label>();

		return dataset;
	}


	enum class type
	{
		training,
		test
	};
	
	template<typename Pixel = uint8_t, typename Label = uint8_t>
	void show_dataset(const MNIST_dataset<Pixel,Label>& dataset, const type& type)
	{
		uint32_t rows{28};
		uint32_t cols{28};

		auto data_image = std::vector<std::vector<Pixel>>{};
		auto data_label = std::vector<Label>{};
		
		if(type == type::training)
		{
			data_image = dataset.training_images;
			data_label = dataset.training_labels;
		}			

		if(type == type::test)
		{
			data_image = dataset.test_images;
			data_label = dataset.test_labels;
		}			
		std::string label;
		for (size_t item_id = 0; item_id < dataset.test_images.size(); ++item_id)
		{
			label = "label: " + std::to_string(static_cast<int>(data_label[item_id]));
			auto* pixels = data_image[item_id].data();

			cv::Mat image;
			switch (sizeof(Pixel))
			{
			case 8: {image =  cv::Mat(rows, cols, CV_64FC1, pixels); break;}
			case 4: {image =  cv::Mat(rows, cols, CV_32FC1, pixels); break;}
			case 1: {image =  cv::Mat(rows, cols, CV_8UC1, pixels); break;}
			}

			// resize bigger for showing
			cv::resize(image, image, cv::Size(200,200));
			
			cv::setWindowTitle("MNIST_VIEWER", label);
			cv::imshow("MNIST_VIEWER", image);

			int key = cv::waitKey(0);
			if(key == 27 || key == 'q')break;

		}
			cv::destroyWindow("MNIST_VIEWER");
	}

	// template<typename Pixel = uint8_t, typename Label = uint8_t, typename Net>
	// void test_net_image(const MNIST_dataset<Pixel,Label>& dataset, const type& type, const Net& net)
	// {
	// 	uint32_t rows{28};
	// 	uint32_t cols{28};

	// 	auto data_image = std::vector<std::vector<Pixel>>{};
	// 	auto data_label = std::vector<Label>{};
		
	// 	if(type == type::training)
	// 	{
	// 		data_image = dataset.training_images;
	// 		data_label = dataset.training_labels;
	// 	}			

	// 	if(type == type::test)
	// 	{
	// 		data_image = dataset.test_images;
	// 		data_label = dataset.test_labels;
	// 	}			
	// 	std::string label;
	// 	for (size_t item_id = 0; item_id < dataset.test_images.size(); ++item_id)
	// 	{
			
	// 		label = "label: " + std::to_string(static_cast<int>(data_label[item_id])) + " net answer: " + std::to_string(net.feedForward(data_image[item_id]));
	// 		auto* pixels = data_image[item_id].data();

	// 		cv::Mat image;
	// 		switch (sizeof(Pixel))
	// 		{
	// 		    case 8: {image =  cv::Mat(rows, cols, CV_64FC1, pixels); break;}
	// 		    case 4: {image =  cv::Mat(rows, cols, CV_32FC1, pixels); break;}
	// 		    case 1: {image =  cv::Mat(rows, cols, CV_8UC1, pixels); break;}
	// 		}

	// 		// resize bigger for showing
	// 		cv::resize(image, image, cv::Size(200,200));
			
	// 		cv::setWindowTitle("MNIST_VIEWER", label);
	// 		cv::imshow("MNIST_VIEWER", image);

	// 		int key = cv::waitKey(0);
	// 		if(key == 27 || key == 'q')break;

	// 	}
	// 	cv::destroyWindow("MNIST_VIEWER");
	// }

} //end of namespace mnist


#endif /* MNIST_READER_H */
