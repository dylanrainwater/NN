#ifndef D_Neural_Network
#define D_Neural_Network

#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <iostream>

struct Label {
    std::string label;
    double lower_bound;
    double upper_bound;
    double perfect_value;
};

class NN {
    private:
        std::string input_file;
        std::vector <std::vector <double>> data;
        std::vector <std::string> known_labels;
        std::vector <std::string> predictions;
        std::vector <double> weights;
        std::vector <Label> labels;
        double original_weights;
        double original_bias;
        double alpha;
        double bias;
        double randomDouble();
    public:
        NN(std::string input_file="");

        double getPerfectVal(Label& label);
        double getPerfectVal(std::string& label_name);
        void populateData();
        void train_test_split();
        void createLabel(double perfect, double lower, double upper, std::string& name);
        std::string getLabel(double prediction);
        void addData(std::vector <std::vector <double>>& data); /* To set data without the need of a file */
        void addDataPoint(std::vector <double>& data_point);
        void train(int iterations=1000); /* Train neural network on data */
        std::vector <double> predict(std::vector <std::vector <double>>& data);
        double predict(std::vector <double>& data_point);
        double sigmoid(double n);
        double sigmoid_derivative(double n);
        double cost(std::vector <double>& targets, std::vector <double>& predictions);
        double cost(double target, double prediction);
        void reset(bool keep_weights);
        double test(int num);
        void setAlpha(double alpha);
        void populateWeights();
};

#endif
