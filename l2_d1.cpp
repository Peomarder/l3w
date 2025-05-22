#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <sstream>
#include <iterator>
#include <tbb/tbb.h>
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include <iostream>
#include <vector>

using namespace std; //https://github.com/wjakob/tbb/tree/master
using namespace tbb;

int BLOCK_SIZE = 32;

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

int main() {
    srand(static_cast<unsigned>(time(0)));
    string input;

    while (true) {
        int n, block_size = 0;
        int thread_count = task_scheduler_init::default_num_threads(); // Retrieve max threads
        cout << "\nEnter command (EXIT to quit):" << endl
             << "Format: [SIZE] [BLOCK_SIZE] or " << thread_count << " (use 'm' for max threads)" << endl
             << "Example: 1000 64 or 4" << endl
             << "> ";

        getline(cin, input);
        if (input == "EXIT") break;

        istringstream iss(input);
        vector<string> tokens{istream_iterator<string>{iss}, istream_iterator<string>{}};

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
                    thread_count = task_scheduler_init::default_num_threads(); // Retrieve max threads
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

        auto start = chrono::high_resolution_clock::now();

        // Using Intel TBB for parallel matrix multiplication
        parallel_for(0, n, 1, [&](int i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] += sum;
            }
        });

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Completed multiplication with Intel TBB in " << elapsed.count() << " seconds." << endl;

        delete[] A;
        delete[] B;
        delete[] C;
    }

    return 0;
}