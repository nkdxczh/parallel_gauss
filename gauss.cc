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
#include <time.h>

using std::cout;
using std::endl;
using namespace std::chrono;

// A 2d (square) array of doubles
class matrix_t {

    //Q1:
    // We use an array of array rather than an explit matrix.
    // Discuss why in terms of complexity of the operations on this data structure.

    // Answer:
    // If we use matrix, we need to traverse all elements to swap two rows, whose time complexity is O(n).
    // While it takes constant time (O(1)) to swap rows by swapping two pointers of array.

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
            cout << A[j][i] << "\t";
        cout << " | " << B[i] << "\n";
    }
    cout << endl;
}

// For a system of equations A * x = b, with Matrix A and Vectors B and X,
// and assuming we only know A and b, compute x via the Gaussian Elimination
// technique
void gauss_col(matrix_t& A, vector_t& B, vector_t& X) {
    // iterate over rows
    for (int i = 0; i < A.getSize(); ++i) {
        // NB: we are now on the ith column

        // For numerical stability, find the largest value in this column
        double big = abs(A[i][i]);
        int row = i;
        for (int k = i + 1; k < A.getSize(); ++k) {
            if (abs(A[i][k]) > big) {
                big = abs(A[i][k]);
                row = k;
            }
        }
        // Given our random initialization, singular matrices are possible!
        if (big == 0.0) {
            cout << "The matrix is singular!" << endl;
            exit(-1);
        }

        // swap so max column value is in ith row
        for (int k = i; k != A.getSize(); ++k) {
            double tem = A[k][i];
            A[k][i] = A[k][row];
            A[k][row] = tem;
        }
        double tem = B[i];
        B[i] = B[row];
        B[row] = tem;

        double c[A.getSize() - i - 1];
        for(int k = i + 1; k < A.getSize(); ++k){
            c[k - i - 1] = -A[i][k] / A[i][i];
        }

        // Eliminate the ith row from all subsequent rows
        //
        // NB: this will lead to all subsequent rows having a 0 in the ith
        // column
        for (int k = i+1; k < A.getSize(); ++k) {
            for (int j = i + 1; j < A.getSize(); ++j)
                A[k][j] += c[j - i - 1] * A[k][i];
        }

        for(int k = i + 1; k < A.getSize(); ++k){
            A[i][k] = 0;
            B[k] += c[k - i -1] * B[i];
        }
    }

    // NB: A is now an upper triangular matrix

    // Use back substitution to solve equation A * x = b
    for (int i = A.getSize() - 1; i >= 0; --i) {
        X[i] = B[i] / A[i][i];
        for (int k = i - 1; k >= 0; --k)
            B[k] -= A[i][k] * X[i];
    }
}

// For a system of equations A * x = b, with Matrix A and Vectors B and X,
// and assuming we only know A and b, compute x via the Gaussian Elimination
// technique
void gauss_row(matrix_t& A, vector_t& B, vector_t& X) {
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

class Max_col{
    int col;
    int row;
    matrix_t& m;
    double big;
public:
    Max_col(int col_, matrix_t& m_):col(col_), m(m_), big(abs(m[col][col])), row(col_){ }
    Max_col(Max_col& b, tbb::split):col(b.col), m(b.m), big(abs(m[col][col])), row(col){}
    void assign(Max_col& b){big = b.big; row = b.row;}

    template<typename Tag>
    void operator()(const tbb::blocked_range<int>& r, Tag tag){
        for(int k = r.begin(); k != r.end(); ++k){
            if(!Tag::is_final_scan) {
                if (abs(m[col][k]) > big) {
                    big = abs(m[col][k]);
                    row = k;
                }
            }
        }
    }

    void operator()(const tbb::blocked_range<int>& r){
        for(int k = r.begin(); k != r.end(); ++k){
            if (abs(m[col][k]) > big) {
                big = abs(m[col][k]);
                row = k;
            }
        }
    }

    int get_row(){return row;}
    int get_big(){return big;}

    void reverse_join(Max_col& b){
        if(b.big > big){
            big = b.big;
            row = b.row;
        }
    }

    void join( Max_col& b ) {
        if(b.big > big){
            big = b.big;
            row = b.row;
        }
    }
};

void parallelGauss_col(matrix_t& A, vector_t& B, vector_t& X) {
    // iterate over rows
    for (int i = 0; i < A.getSize(); ++i) {
        // NB: we are now on the ith column
        //print(A,B);

        // For numerical stability, find the largest value in this column

        //serial
        /*double big = abs(A[i][i]);
        int row = i;
        for (int k = i + 1; k < A.getSize(); ++k) {
            if (abs(A[i][k]) > big) {
                big = abs(A[i][k]);
                row = k;
            }
        }*/

        //parallel
        Max_col body(i, A);
        //reduce
        //tbb::parallel_reduce(tbb::blocked_range<int>(i + 1, A.getSize()), body);
        //scan
        tbb::parallel_scan(tbb::blocked_range<int>(i + 1, A.getSize()), body);

        double big = body.get_big();
        int row = body.get_row();

        // Given our random initialization, singular matrices are possible!
        if (big == 0.0) {
            cout << "The matrix is singular!" << endl;
            exit(-1);
        }

        // swap so max column value is in ith row
        // parallel swap
        tbb::parallel_for(
                tbb::blocked_range<int>(i, A.getSize()),
                [&](tbb::blocked_range<int> r) {
                    for (int k = r.begin(); k != r.end(); ++k) {
                        double tem = A[k][i];
                        A[k][i] = A[k][row];
                        A[k][row] = tem;
                    }
                });
        // serial swap
        /*for (int k = i; k != A.getSize(); ++k) {
            double tem = A[k][i];
            A[k][i] = A[k][row];
            A[k][row] = tem;
        }
        double tem = B[i];
        B[i] = B[row];
        B[row] = tem;*/

        // Eliminate the ith row from all subsequent rows
        //
        // NB: this will lead to all subsequent rows having a 0 in the ith
        double c[A.getSize() - i - 1];
        // parallel compute
        /*tbb::parallel_for(
                tbb::blocked_range<int>(i + 1, A.getSize()),
                [&](tbb::blocked_range<int> r) {
                    for (int k = r.begin(); k != r.end(); ++k) {
                        c[k - i - 1] = -A[i][k] / A[i][i];
                    }
                });*/

        //serial compute
        for(int k = i + 1; k < A.getSize(); ++k){
            c[k - i - 1] = -A[i][k] / A[i][i];
        }
        // column
        tbb::parallel_for(
                tbb::blocked_range<int>(i + 1, A.getSize()),
                [&](tbb::blocked_range<int> r) {
                    for (int k = r.begin(); k != r.end(); ++k) {
                        for (int j = i + 1; j < A.getSize(); ++j)
                            A[k][j] += c[j - i - 1] * A[k][i];
                    }
                });

        // parallel compute
        /*tbb::parallel_for(
                tbb::blocked_range<int>(i + 1, A.getSize()),
                [&](tbb::blocked_range<int> r) {
                    for (int k = r.begin(); k != r.end(); ++k) {
                        A[i][k] = 0;
                        B[k] += c[k - i -1] * B[i];
                    }
                });*/

        // serial compute
        for(int k = i + 1; k < A.getSize(); ++k){
            A[i][k] = 0;
            B[k] += c[k - i -1] * B[i];
        }
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

    for (int i = A.getSize() - 1; i >= 0; --i) {
        X[i] = B[i] / A[i][i];
        /*for (int k = i - 1; k >= 0; --k)
            B[k] -= A[i][k] * X[i];*/
        tbb::parallel_for(
                tbb::blocked_range<int>(0, i),
                [&](tbb::blocked_range<int> r ) {
                    for (int k = r.begin(); k != r.end(); ++k)
                        B[k] -= A[i][k] * X[i];
                });
    }
}

class Max_row{
    int col;
    int row;
    matrix_t& m;
    double big;
public:
    Max_row(int col_, matrix_t& m_):col(col_), m(m_), big(abs(m[col][col])), row(col_){ }
    Max_row(Max_row& b, tbb::split):col(b.col), m(b.m), big(abs(m[col][col])), row(col){}
    void assign(Max_row& b){big = b.big; row = b.row;}

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

    void operator()(const tbb::blocked_range<int>& r){
        for(int k = r.begin(); k != r.end(); ++k){
            if (abs(m[k][col]) > big) {
                big = abs(m[k][col]);
                row = k;
            }
        }
    }

    int get_row(){return row;}
    int get_big(){return big;}

    void reverse_join(Max_row& b){
        if(b.big > big){
            big = b.big;
            row = b.row;
        }
    }

    void join( Max_row& b ) {
        if(b.big > big){
            big = b.big;
            row = b.row;
        }
    }
};

void parallelGauss_row(matrix_t& A, vector_t& B, vector_t& X) {
    // iterate over rows
    for (int i = 0; i < A.getSize(); ++i) {
        // NB: we are now on the ith column

        // For numerical stability, find the largest value in this column

        //serial
        double big = abs(A[i][i]);
        int row = i;
        for (int k = i + 1; k < A.getSize(); ++k) {
            if (abs(A[i][k]) > big) {
                big = abs(A[i][k]);
                row = k;
            }
        }

        //parallel
        //Max_row body(i, A);
        //reduce
        //tbb::parallel_reduce(tbb::blocked_range<int>(i + 1, A.getSize()), body);
        //scan
        //tbb::parallel_scan(tbb::blocked_range<int>(i + 1, A.getSize()), body);

        //double big = body.get_big();
        //int row = body.get_row();

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

    // Use back substitution to solve equation A * x = b
    // serial
    /*for (int i = A.getSize() - 1; i >= 0; --i) {
        X[i] = B[i] / A[i][i];
        for (int k = i - 1; k >= 0; --k)
            B[k] -= A[k][i] * X[i];
    }*/

    for (int i = A.getSize() - 1; i >= 0; --i) {
        X[i] = B[i] / A[i][i];
        tbb::parallel_for(
                tbb::blocked_range<int>(0, i),
                [&](tbb::blocked_range<int> r ) {
                    for (int k = r.begin(); k != r.end(); ++k)
                        B[k] -= A[k][i] * X[i];
                });
    }
}

// This function makes sure that the values in X actually satisfy
// the equation A * x = b
//
// Q2:
// Why is the code of the check so complex and involves calculating the ratio?
// The answer is related with the used data types.

//Answer: Because We use double to save the result, which has precision loss.
//So the ans would not equal to the original value. But the difference is so small that it could be ignored in division calculation.

void check_col(matrix_t& A, vector_t& B, vector_t& X) {
    for (int i = 0; i < A.getSize(); ++i) {
        // compute the value of B based on X
        double ans = 0;
        for (int j = 0; j < A.getSize(); j++)
            ans += A[j][i] * X[j];

        double ratio = std::max(abs(ans / B[i]), abs(B[i] / ans));
        if (ratio != 1) {
            cout << "Verification failed for index = " << i << "." << endl;
            cout << ans << " != " << B[i] << endl;
            return;
        }
    }
    //cout << "Verification succeeded" << endl;
}

void check_row(matrix_t& A, vector_t& B, vector_t& X) {
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

    int  seed     = int(time(0));   // random seed
    int  size     = 256;   // # rows in the matrix
    int  range    = 65536; // matrix elements will have values between -range and range
    bool verbose  = false; // should we print some diagnostics?
    bool docheck  = true;  // should we verify the output?
    bool parallel = false; // use parallelism?
    bool row = true;

    // Parse the command line options:
    int o;
    while ((o = getopt(argc, argv, "r:n:g:hvcpl")) != -1) {
        switch (o) {
            case 'r': seed = atoi(optarg);  break;
            case 'n': size = atoi(optarg);  break;
            case 'g': range = atoi(optarg); break;
            case 'h': usage();              break;
            case 'v': verbose = !verbose;   break;
            case 'c': docheck = !docheck;   break;
            case 'p': parallel = !parallel; break;
            case 'l': row = !row; break;
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
        if(row)
            parallelGauss_row(A, B, X);
        else
            parallelGauss_col(A, B, X);
    else
        if(row)
            gauss_row(A, B, X);
        else
            gauss_row(A, B, X);
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
        if(row)
            check_row(A, B, X);
        else
            check_col(A, B, X);
    }

    // Print the execution time
    duration<double> time_span = duration_cast<duration<double>>(endtime - starttime);
    //cout << "Total execution time: " << time_span.count() << " seconds" << endl;
    cout << time_span.count() << endl;
}
