#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iterator>
#include <pthread.h>
#include <stdio.h>

using namespace std;

int BLOCK_SIZE = 32;

// Structure for passing thread arguments
struct ThreadData {
    double* A;
    double* B;
    double* C;
    int n;
    int start_row;
    int end_row;
};

// Fill a matrix with random values
void fill_random(double* matrix, int n) {
    if (matrix == nullptr) {
        cerr << "Error: Matrix is not allocated properly!" << endl;
        return;
    }

    for (int i = 0; i < n * n; ++i) {
        double random_value;
        do {
            random_value = static_cast<double>(rand());
        } while (random_value == 0.0);

        matrix[i] = 1.0 / random_value;
    }
}

// Thread function for matrix multiplication
void* matrix_multiply(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    for (int i = data->start_row; i < data->end_row; ++i) {
        for (int j = 0; j < data->n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < data->n; ++k) {
                sum += data->A[i * data->n + k] * data->B[k * data->n + j];
            }
            data->C[i * data->n + j] += sum;
        }
    }
    pthread_exit(nullptr);
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    string input;

    while (true) {
        cout << "\nEnter command (EXIT to quit):" << endl
             << "Format: [SIZE] [BLOCK_SIZE] or THREAD_COUNT (use 'm' for max threads)" << endl
             << "Example: 1000 64 or 4" << endl
             << "> ";

        getline(cin, input);
        if (input == "EXIT") break;

        istringstream iss(input);
        vector<string> tokens{istream_iterator<string>{iss}, istream_iterator<string>{}};

        int n, block_size = 0, thread_count = 0;

        if (tokens.size() < 2) {
            cerr << "Invalid input! Minimum 2 parameters required" << endl;
            continue;
        }

        try {
            n = stoi(tokens[0]);
            if (n <= 0) throw invalid_argument("Size must be positive");

            block_size = stoi(tokens[1]);
            if (block_size <= 0) throw invalid_argument("Block size must be positive");

            if (tokens.size() > 2) {
                string third_param = tokens[2];
                if (third_param == "m" || third_param == "M") {
                    thread_count = std::thread::hardware_concurrency(); // Retrieve max threads
                    cout << "Using maximum threads: " << thread_count << endl;
                } else {
                    thread_count = stoi(third_param);
                    if (thread_count <= 0) throw invalid_argument("Thread count must be positive");
                }
            } else {
                cout << "Using default thread count: 4" << endl;
                thread_count = 4;
            }
        } catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
            continue;
        }

        double* A = new double[n * n];
        double* B = new double[n * n];
        double* C = new double[n * n];

        fill_random(A, n);
        fill_random(B, n);
        fill(C, C + n * n, 0.0);

        pthread_t* threads = new pthread_t[thread_count];
        ThreadData* thread_data = new ThreadData[thread_count];

        auto start = chrono::high_resolution_clock::now();

        // Distributing work among threads
        int rows_per_thread = n / thread_count;
        for (int i = 0; i < thread_count; ++i) {
            thread_data[i].A = A;
            thread_data[i].B = B;
            thread_data[i].C = C;
            thread_data[i].n = n;
            thread_data[i].start_row = i * rows_per_thread;
            thread_data[i].end_row = (i == thread_count - 1) ? n : (i + 1) * rows_per_thread;

            int create_status = pthread_create(&threads[i], NULL, matrix_multiply, (void*)&thread_data[i]);
            if (create_status) {
                fprintf(stderr, "Error - pthread_create() return code: %d
", create_status);
                exit(EXIT_FAILURE);
            }
        }

        // Joining threads
        for (int i = 0; i < thread_count; ++i) {
            pthread_join(threads[i], nullptr);
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Completed multiplication with multithreaded algorithm in " << elapsed.count() << " seconds." << endl;

        delete[] A;
        delete[] B;
        delete[] C;
        delete[] threads;
        delete[] thread_data;
    }

    return 0;
}