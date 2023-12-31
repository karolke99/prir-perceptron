#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define FEATURES 8

typedef struct {
    double features[FEATURES];
    int label;
} DataPoint;

typedef struct {
    double weights[FEATURES];
    double bias;
} Perceptron;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void trainPerceptron(Perceptron *perceptron, DataPoint data, double learningRate) {
    double weightedSum = perceptron->bias;
    for (int i = 0; i < FEATURES; i++) {
        weightedSum += perceptron->weights[i] * data.features[i];
    }

    double prediction = sigmoid(weightedSum);
    double error = data.label - prediction;

    perceptron->bias += learningRate * error * prediction * (1.0 - prediction);

    for (int i = 0; i < FEATURES; i++) {
        perceptron->weights[i] += learningRate * error * prediction * (1.0 - prediction) * data.features[i];
    }
}

int predict(Perceptron perceptron, double *features) {
    double weightedSum = perceptron.bias;
    for (int i = 0; i < FEATURES; i++) {
        weightedSum += perceptron.weights[i] * features[i];
    }

    double prediction = sigmoid(weightedSum);
    return (prediction >= 0.5) ? 1 : 0;
}

void shuffleData(DataPoint *dataSet, int dataSize, int rank) {
    if (rank == 0) {
        for (int i = dataSize - 1; i > 0; i--) {
            int j = rand() % (i + 1);

            DataPoint temp = dataSet[i];
            dataSet[i] = dataSet[j];
            dataSet[j] = temp;
        }
    }
}

double calculateError(Perceptron perceptron, DataPoint *dataSet, int dataSize) {
    double totalError = 0.0;
    for (int i = 0; i < dataSize; i++) {
        double predictedLabel = predict(perceptron, dataSet[i].features);
        totalError += fabs(dataSet[i].label - predictedLabel);
    }
    return totalError / dataSize;
}

void standardScaler(DataPoint *dataSet, int dataSize) {
    double mean[FEATURES] = {0.0};
    double variance[FEATURES] = {0.0};

    for (int i = 0; i < dataSize; i++) {
        for (int j = 0; j < FEATURES; j++) {
            mean[j] += dataSet[i].features[j];
        }
    }

    for (int j = 0; j < FEATURES; j++) {
        mean[j] /= dataSize;
    }

    for (int i = 0; i < dataSize; i++) {
        for (int j = 0; j < FEATURES; j++) {
            variance[j] += pow(dataSet[i].features[j] - mean[j], 2);
        }
    }

    for (int j = 0; j < FEATURES; j++) {
        variance[j] /= dataSize;
        variance[j] = sqrt(variance[j]);
    }

    for (int i = 0; i < dataSize; i++) {
        for (int j = 0; j < FEATURES; j++) {
            dataSet[i].features[j] = (dataSet[i].features[j] - mean[j]) / variance[j];
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(42);

    FILE *file = fopen("diab.csv", "r");
    if (file == NULL) {
        perror("Unable to open file");
        MPI_Finalize();
        return 1;
    }

    int dataSize = 768;
    DataPoint *dataSet = malloc(dataSize * sizeof(DataPoint));

    char buffer[256];
    fgets(buffer, sizeof(buffer), file);

    for (int i = 0; i < dataSize; i++) {
        for (int j = 0; j < FEATURES; j++) {
            fscanf(file, "%lf,", &dataSet[i].features[j]);
        }
        fscanf(file, "%d", &dataSet[i].label);
    }

    fclose(file);

    double start = MPI_Wtime();

    standardScaler(dataSet, dataSize);

    shuffleData(dataSet, dataSize, rank);

    int trainingSize = dataSize * 0.8;
    int testingSize = dataSize - trainingSize;

    int localTrainingSize = trainingSize / size;
    int localTrainingStart = rank * localTrainingSize;
    int localTrainingEnd = (rank == size - 1) ? trainingSize : (rank + 1) * localTrainingSize;

    DataPoint *localTrainingSet = dataSet + localTrainingStart;

    Perceptron perceptron;

    for (int i = 0; i < FEATURES; i++) {
        perceptron.weights[i] = (rand() % 1000) / 1000.0;
    }
    perceptron.bias = (rand() % 1000) / 1000.0;

    double learningRate = 0.01;
    int numEpochs = 1000;

    for (int epoch = 0; epoch < numEpochs; epoch++) {
        for (int i = localTrainingStart; i < localTrainingEnd; i++) {
            trainPerceptron(&perceptron, localTrainingSet[i], learningRate);
        }

        MPI_Allreduce(MPI_IN_PLACE, perceptron.weights, FEATURES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &perceptron.bias, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    int correctPredictions = 0;

    if (rank == 0) {
        printf("Time: %f\n", MPI_Wtime() - start);
        printf("Real Label\tPredicted Label\tEvaluation\n");
        for (int i = trainingSize; i < dataSize; i++) {
            int realLabel = dataSet[i].label;
            int predictedLabel = predict(perceptron, dataSet[i].features);

            printf("%d\t\t%d\t\t", realLabel, predictedLabel);
            if (realLabel == predictedLabel) {
                printf("Correct\n");
                correctPredictions++;
            } else {
                printf("Incorrect\n");
            }
        }

        double accuracy = (double)correctPredictions / testingSize * 100.0;
        printf("Accuracy: %.2f%%\n", accuracy);
        printf("Accuracy: %.2f%%\n", accuracy);
    }

    free(dataSet);
    MPI_Finalize();

    return 0;
}
