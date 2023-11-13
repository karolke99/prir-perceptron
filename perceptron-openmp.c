#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define FEATURES 8  // Liczba cech (atrybutów) w danych

// Struktura reprezentująca pojedynczy punkt danych
typedef struct {
    double features[FEATURES];
    int label;
} DataPoint;

// Struktura reprezentująca perceptron
typedef struct {
    double weights[FEATURES];
    double bias;
} Perceptron;

// Funkcja sigmoidalna
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Funkcja ucząca perceptron na pojedynczym punkcie danych
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

// Funkcja klasyfikująca dane za pomocą perceptronu
int predict(Perceptron perceptron, double *features) {
    double weightedSum = perceptron.bias;
    for (int i = 0; i < FEATURES; i++) {
        weightedSum += perceptron.weights[i] * features[i];
    }

    double prediction = sigmoid(weightedSum);
    return (prediction >= 0.5) ? 1 : 0;
}

// Funkcja do losowego tasowania danych
void shuffleData(DataPoint *dataSet, int dataSize) {
    for (int i = dataSize - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        // Zamień elementy i j miejscami
        DataPoint temp = dataSet[i];
        dataSet[i] = dataSet[j];
        dataSet[j] = temp;
    }
}

// Funkcja do obliczania błędu na zbiorze danych
double calculateError(Perceptron perceptron, DataPoint *dataSet, int dataSize) {
    double totalError = 0.0;
    for (int i = 0; i < dataSize; i++) {
        double predictedLabel = predict(perceptron, dataSet[i].features);
        totalError += fabs(dataSet[i].label - predictedLabel);
    }
    return totalError / dataSize;
}

// Funkcja do normalizacji danych zgodnie ze standard scalerem
void standardScaler(DataPoint *dataSet, int dataSize) {
    double mean[FEATURES] = {0.0};
    double variance[FEATURES] = {0.0};

    // Oblicz średnią dla każdej cechy
    for (int i = 0; i < dataSize; i++) {
        for (int j = 0; j < FEATURES; j++) {
            mean[j] += dataSet[i].features[j];
        }
    }

    for (int j = 0; j < FEATURES; j++) {
        mean[j] /= dataSize;
    }

    // Oblicz wariancję dla każdej cechy
    for (int i = 0; i < dataSize; i++) {
        for (int j = 0; j < FEATURES; j++) {
            variance[j] += pow(dataSet[i].features[j] - mean[j], 2);
        }
    }

    for (int j = 0; j < FEATURES; j++) {
        variance[j] /= dataSize;
        variance[j] = sqrt(variance[j]);
    }

    // Zastosuj standard scaler do każdej cechy
    for (int i = 0; i < dataSize; i++) {
        for (int j = 0; j < FEATURES; j++) {
            dataSet[i].features[j] = (dataSet[i].features[j] - mean[j]) / variance[j];
        }
    }
}

int main() {
    // Inicjalizacja generatora liczb losowych
    srand(time(NULL));

    FILE *file = fopen("diab.csv", "r");
    if (file == NULL) {
        perror("Unable to open file");
        return 1;
    }

    // Wczytanie danych z pliku
    int dataSize = 768;  // Załóżmy, że plik zawiera 768 rekordów danych
    DataPoint *dataSet = malloc(dataSize * sizeof(DataPoint));

    // Pominięcie pierwszej linii (nagłówek)
    char buffer[256];
    fgets(buffer, sizeof(buffer), file);

    for (int i = 0; i < dataSize; i++) {
        for (int j = 0; j < FEATURES; j++) {
            fscanf(file, "%lf,", &dataSet[i].features[j]);
        }
        fscanf(file, "%d", &dataSet[i].label);
    }

    fclose(file);

    // Normalizacja danych zgodnie ze standard scalerem
    standardScaler(dataSet, dataSize);

    // Tasowanie danych
    shuffleData(dataSet, dataSize);

    // Podział na zbiór treningowy i testowy (80% do 20%)
    int trainingSize = dataSize * 0.8;
    int testingSize = dataSize - trainingSize;

    // Przypisanie danych do zbioru treningowego i testowego
    DataPoint *trainingSet = dataSet;
    DataPoint *testingSet = dataSet + trainingSize;

    // Inicjalizacja perceptronu z losowymi wagami
    Perceptron perceptron;
    for (int i = 0; i < FEATURES; i++) {
        perceptron.weights[i] = (rand() % 1000) / 1000.0;  // Losowe liczby z przedziału [0, 1)
    }
    perceptron.bias = (rand() % 1000) / 1000.0;

    // Uczenie perceptronu na zbiorze treningowym przez określoną liczbę epok
    double learningRate = 0.01;
    int numEpochs = 1000;

    for (int epoch = 0; epoch < numEpochs; epoch++) {

        #pragma omp parallel for
        for (int i = 0; i < trainingSize; i++) {
            trainPerceptron(&perceptron, trainingSet[i], learningRate);
        }
    }

    // Testowanie perceptronu na zbiorze testowym
    int correctPredictions = 0;
    printf("Real Label\tPredicted Label\tEvaluation\n");
    for (int i = 0; i < testingSize; i++) {
        int realLabel = testingSet[i].label;
        int predictedLabel = predict(perceptron, testingSet[i].features);
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

    free(dataSet);

    return 0;
}
