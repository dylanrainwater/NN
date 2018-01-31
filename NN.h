#ifndef D_Neural_Network
#define D_Neural_Network
#include <string>
#include <vector>

class NN {
    private:
        std::string input_file;
        std::vector <double> data;
        std::vector <std::string> predictions;
    public:
        NN(std::string input_file, int label_col);
        void train_test_split();
        void addData(std::vector <double>& data); /* To set data without the need of a file */
        void addDataPoint(double data_point);
        void train(std::string file_name); /* Train neural network on data */
        std::vector <std::string> predict(std::vector <double>& data);
        std::string predict(double data_point);
        double sigmoid(double n);
        double sigmoid_derivative(double n);
        double cost(std::vector <double>& targets, std::vector <double>& predictions);
        double cost(double target, double prediction);
        /* TODO: Add hyperparameter tuning options */
};

#endif
