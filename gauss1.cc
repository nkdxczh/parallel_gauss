// Gaussian Elimination Solver
//
// Lehigh University CSE 375/CSE 475
//
// Fall 2017

#include <unistd.h>
#include <iostream>
#include <cmath>
#include <random>
#include <functional>
#include <chrono>
#include "tbb/tbb.h"
#include "tbb/blocked_range.h"

using std::cout;
using std::endl;
using namespace std::chrono;

// A 2d (square) array of doubles
class matrix_t {

    //Q1:
    // We use an array of array rather than an explit matrix.
    // Discuss why in terms of complexity of the operations on this data structure.

    double** M;
    // the # rows / # columns / sqrt(# elements)
    unsigned int size;
public:
    // Construct by allocating the matrix
    matrix_t(unsigned int n)
            : size(n), M(new double*[n])
    {
        for (int i = 0; i < size; ++i)
            M[i] = new double[size];
    }
    // Give the illusion of this being a simple array
    double*& operator[](std::size_t idx) { return M[idx]; };
    double* const& operator[](std::size_t idx) const { return M[idx]; };
    // Since size is private, we need a getter
    unsigned int getSize() { return size; }
};

// A 1d array of doubles
class vector_t {
    // simple array of doubles
    double* V;
    // size
    unsigned int size;
public:
    // Construct by allocating the vector
    vector_t(unsigned int n)
            : size(n), V(new double[n])
    { }
    // Give the illusion of this being a simple array
    double& operator[](std::size_t idx) { return V[idx]; };
    const double& operator[](std::size_t idx) const { return V[idx]; };
    unsigned int getSize() { return size; }
};

// Given a random seed, this method populates the elements of A and then B with a
// sequence of random numbers in the range (-range...range)
void initializeFromSeed(int seed, matrix_t& A, vector_t& B, int range) {
    // Use a Mersenne Twister to create doubles in the requested range
    std::mt19937 seeder(seed);
    auto mt_rand = std::bind(std::uniform_real_distribution<double>(-range,range),
                             std::mt19937(seed));
    // populate A
    for (int i = 0; i < A.getSize(); ++i)
        for (int j = 0; j < A.getSize(); ++j)
            A[i][j] = (double)(mt_rand());
    // populate B
    for (int i = 0; i < B.getSize(); ++i)
        B[i] = (double)(mt_rand());
}

// Print the matrix and array in a form that looks good
void print(matrix_t& A, vector_t& B) {
    for (int i = 0; i < A.getSize(); ++i) {
        for (int j = 0; j < A.getSize(); ++j)
            cout << A[i][j] << "\t";
        cout << " | " << B[i] << "\n";
    }
    cout << endl;
}

// For a system of equations A * x = b, with Matrix A and Vectors B and X,
// and assuming we only know A and b, compute x via the Gaussian Elimination
// technique
void gauss(matrix_t& A, vector_t& B, vector_t& X) {
    // iterate over rows
    for (int i = 0; i < A.getSize(); ++i) {
        // NB: we are now on the ith column

        // For numerical stability, find the largest value in this column
        double big = abs(A[i][i]);
        int row = i;
        for (int k = i + 1; k < A.getSize(); ++k) {
            if (abs(A[k][i]) > big) {
                big = abs(A[k][i]);
                row = k;
            }
        }
        // Given our random initialization, singular matrices are possible!
        if (big == 0.0) {
            cout << "The matrix is singular!" << endl;
            exit(-1);
        }

        // swap so max column value is in ith row
        std::swap(A[i], A[row]);
        std::swap(B[i], B[row]);

        // Eliminate the ith row from all subsequent rows
        //
        // NB: this will lead to all subsequent rows having a 0 in the ith
        // column
        for (int k = i + 1; k < A.getSize(); ++k) {
            double c = -A[k][i] / A[i][i];
            for (int j = i; j < A.getSize(); ++j)
                if (i == j)
                    A[k][j] = 0;
                else
                    A[k][j] += c * A[i][j];
            B[k] += c * B[i];
        }
    }

    // NB: A is now an upper triangular matrix

    // Use back substitution to solve equation A * x = b
    for (int i = A.getSize() - 1; i >= 0; --i) {
        X[i] = B[i] / A[i][i];
        for (int k = i - 1; k >= 0; --k)
            B[k] -= A[k][i] * X[i];
    }
}

// For a system of equations A * x = b, with Matrix A and Vectors B and X,
// and assuming we only know A and b, compute x via the Gaussian Elimination
// technique
class Body{
    int col;
    int row;
    matrix_t& m;
    double big;
public:
    Body(int col_, matrix_t& m_):col(col_), m(m_), big(abs(m[col][col])), row(col_){ }
    Body(Body& b, tbb::split):col(b.col), m(b.m), big(abs(m[col][col])), row(col){}
    void assign(Body& b){big = b.big; row = b.row;}

    template<typename Tag>
    void operator()(const tbb::blocked_range<int>& r, Tag tag){
        for(int k = r.begin(); k != r.end(); ++k){
            if(!Tag::is_final_scan) {
                if (abs(m[k][col]) > big) {
                    big = abs(m[k][col]);
                    row = k;
                }
            }
        }
    }
    int get_row(){return row;}
    void reverse_join(Body& b){
        if(b.big > big){
            big = b.big;
            row = b.row;
        }
    }
    int get_big(){return big;}
};

void parallelGauss(matrix_t& A, vector_t& B, vector_t& X) {
    // iterate over rows
    for (int i = 0; i < A.getSize(); ++i) {
        // NB: we are now on the ith column

        // For numerical stability, find the largest value in this column
        /*double big = abs(A[i][i]);
        int row = i;
        for (int k = i + 1; k < A.getSize(); ++k) {
            if (abs(A[k][i]) > big) {
                big = abs(A[k][i]);
                row = k;
            }
        }*/

        Body body(i, A);
        tbb::parallel_scan(tbb::blocked_range<int>(i + 1, A.getSize()), body);
        double big = body.get_big();
        int row = body.get_row();

        // Given our random initialization, singular matrices are possible!
        if (big == 0.0) {
            cout << "The matrix is singular!" << endl;
            exit(-1);
        }

        // swap so max column value is in ith row
        std::swap(A[i], A[row]);
        std::swap(B[i], B[row]);

        // Eliminate the ith row from all subsequent rows
        //
        // NB: this will lead to all subsequent rows having a 0 in the ith
        // column
        tbb::parallel_for(
                tbb::blocked_range<int>(i + 1, A.getSize()),
                [&](tbb::blocked_range<int> r) {
                    for (int k = r.begin(); k != r.end(); ++k) {
                        double c = -A[k][i] / A[i][i];

                        for (int j = i; j < A.getSize(); ++j)
                            if (i == j)
                                A[k][j] = 0;
                            else
                                A[k][j] += c * A[i][j];
                        B[k] += c * B[i];
                    }
                });

        /*tbb::parallel_for(
        tbb::blocked_range<int>(i, A.getSize()),
                [&](tbb::blocked_range<int> rr) {
                    for (int j = rr.begin(); j != rr.end(); ++j)
                        if (i == j)
                            A[k][j] = 0;
                        else
                            A[k][j] += c * A[i][j];
                }
        );
        B[k] += c * B[i];*/
    }

    /*tbb::parallel_for(
            tbb::blocked_range<int>(i + 1, A.getSize()),
            [&](tbb::blocked_range<int> r) {
                for (int k = r.begin(); k != r.end(); ++k) {
                    double c = -A[k][i] / A[i][i];

                    for (int j = i; j < A.getSize(); ++j)
                        if (i == j)
                            A[k][j] = 0;
                        else
                            A[k][j] += c * A[i][j];
                    B[k] += c * B[i];
                }
            });*/
    // NB: A is now an upper triangular matrix

    // Use back substitution to solve equation A * x = b
    /*for (int i = A.getSize() - 1; i >= 0; --i) {
        X[i] = B[i] / A[i][i];
        tbb::parallel_for(
                tbb::blocked_range<int>(0, i),
                [&](tbb::blocked_range<int> r ) {
                    for (int k = r.begin(); k != r.end(); ++k)
                        B[k] -= A[k][i] * X[i];
                });
    }*/
    // Use back substitution to solve equation A * x = b
    for (int i = A.getSize() - 1; i >= 0; --i) {
        X[i] = B[i] / A[i][i];
        for (int k = i - 1; k >= 0; --k)
            B[k] -= A[k][i] * X[i];
    }
}

void parallelGauss1(matrix_t& A, vector_t& B, vector_t& X) {
    // iterate over rows
    for (int i = 0; i < A.getSize(); ++i) {
        // NB: we are now on the ith column

        // For numerical stability, find the largest value in this column
        double big = fabs(A[i][0]);
        int col = 0;
        for (int k = 1; k < A.getSize(); ++k) {
            if (fabs(A[i][k]) > big) {
                big = fabs(A[i][k]);
                col = k;
            }
        }

        // Given our random initialization, singular matrices are possible!
        if (big == 0.0) {
            cout << "The matrix is singular!" << endl;
            exit(-1);
        }

        // Eliminate the ith row from all subsequent rows
        //
        // NB: this will lead to all subsequent rows having a 0 in the ith
        // column
        tbb::parallel_for(
                tbb::blocked_range<int>(i + 1, A.getSize()),
                [&](tbb::blocked_range<int> r) {
                    for (int k = r.begin(); k != r.end(); ++k) {
                        double c = -A[k][col] / A[i][col];

                        for (int j = 0; j < A.getSize(); ++j)
                            if (col == j)
                                A[k][j] = 0;
                            else
                                A[k][j] += c * A[i][j];
                        B[k] += c * B[i];
                    }
                });
    }

    // Use back substitution to solve equation A * x = b
    for (int i = A.getSize() - 1; i >= 0; --i) {
        int ec;
        for(int j = 0; j < A.getSize(); ++j)
            if(A[i][j] != 0){
                ec = j;
                break;
            }
        X[ec] = B[i] / A[i][ec];
        for (int k = i - 1; k >= 0; --k) {
            B[k] -= A[k][ec] * X[ec];
            A[k][ec] = 0;
        }
    }
}

// This function makes sure that the values in X actually satisfy
// the equation A * x = b
//
// Q2:
// Why is the code of the check so complex and involves calculating the ratio?
// The answer is related with the used data types.

void check(matrix_t& A, vector_t& B, vector_t& X) {
    for (int i = 0; i < A.getSize(); ++i) {
        // compute the value of B based on X
        double ans = 0;
        for (int j = 0; j < A.getSize(); j++)
            ans += A[i][j] * X[j];

        double ratio = std::max(abs(ans / B[i]), abs(B[i] / ans));
        if (ratio != 1) {
            cout << "Verification failed for index = " << i << "." << endl;
            cout << ans << " != " << B[i] << endl;
            return;
        }
    }
    //cout << "Verification succeeded" << endl;
}

// This function prints some helpful usage information
void usage() {
    cout << "Gaussian Elimination Solver" << endl;
    cout << "  Usage: gauss [options]" << endl;
    cout << "    -r <int> : specify a seed for the random number generator (default 475)" << endl;
    cout << "    -n <int> : indicate the number of rows in the matrix (default 256)" << endl;
    cout << "    -g <int> : specify a range for values in the matrix (default 65536)" << endl;
    cout << "    -v       : toggle verbose output (default true)" << endl;
    cout << "    -p       : toggle parallel mode (default false)" << endl;
    cout << "    -c       : toggle verifying the result (default true)" << endl;
    cout << "    -h       : print this message" << endl;
}


int main(int argc, char *argv[]) {

    int  seed     = 475;   // random seed
    int  size     = 256;   // # rows in the matrix
    int  range    = 65536; // matrix elements will have values between -range and range
    bool verbose  = false; // should we print some diagnostics?
    bool docheck  = true;  // should we verify the output?
    bool parallel = false; // use parallelism?

    // Parse the command line options:
    int o;
    while ((o = getopt(argc, argv, "r:n:g:hvcp")) != -1) {
        switch (o) {
            case 'r': seed = atoi(optarg);  break;
            case 'n': size = atoi(optarg);  break;
            case 'g': range = atoi(optarg); break;
            case 'h': usage();              break;
            case 'v': verbose = !verbose;   break;
            case 'c': docheck = !docheck;   break;
            case 'p': parallel = !parallel; break;
            default:  usage();              exit(-1);
        }
    }

    // Print the configuration... this makes results of scripted experiments
    // much easier to parse
    //cout << "r,n,g,p = " << seed << ", " << size << ", " << range << ", " << parallel << endl;

    // Create our matrix and vectors, and populate them with default values
    matrix_t A(size);
    vector_t B(size);
    vector_t X(size);
    initializeFromSeed(seed, A, B, range);

    // Print initial matrix
    if (verbose) {
        cout << "Matrix (A) | B" << endl;
        print(A, B);
    }

    // Calculate solution
    auto starttime = high_resolution_clock::now();
    if (parallel)
        //cout << "Parallel version not yet implemented" << endl;
        parallelGauss1(A, B, X);
    else
        gauss(A, B, X);
    auto endtime = high_resolution_clock::now();

    // Print result
    if (verbose) {
        cout << "Result X" << endl;
        for (int i = 0; i < A.getSize(); ++i)
            cout << X[i] << " ";
        cout << endl << endl;
    }

    // Check the solution?
    if (docheck) {
        // Pseudorandom number generators are nice... We can re-create A and
        // B by re-initializing them from the same seed as before
        initializeFromSeed(seed, A, B, range);
        check(A, B, X);
    }

    // Print the execution time
    duration<double> time_span = duration_cast<duration<double>>(endtime - starttime);
    //cout << "Total execution time: " << time_span.count() << " seconds" << endl;
    cout << time_span.count() << endl;
}
