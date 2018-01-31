#include "NN.h"

NN::NN(std::string input_file, int label_col=-1)
{
    this->input_file = input_file;
    this->data = std::vector <double>();
    this->predictions = std::vector <std::string>();

    //TODO Read input file and populate data
}

void NN::train_test_split()
{
}

void NN::addData(std::vector <double>& data)
{
}

void NN::addDataPoint(double data_point)
{
}

void NN::train(std::string file_name="")
{
}

std::vector <std::string> NN::predict(std::vector <double>& data)
{
    std::vector <std::string> predictions;
    return predictions;
}

std::string NN::predict(double data_point)
{
    std::string prediction;
    return prediction;
}

double NN:sigmoid(double n)
{
}

double NN:sigmoid_derivative(double n)
{
}

double NN::cost(std::vector <double>& targets, std::vector <double>& predictions)
{
    double cost;
    return cost;
}

double NN::cost(double target, double prediction)
{
    double cost;
    return cost;
}
