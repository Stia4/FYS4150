#include <iostream>  // input/output enabler
#include <armadillo> // better vectors/matrices
#include <cmath>     // math functions
#include <fstream>   // read/write to files
#include <chrono>    // measure time
using namespace std; // avoid having to write std::

arma::vec u(arma::vec x){
    // function u(x) from problem set
    return 1 - (1-exp(-10))*x - exp(-10*x);
}

void write_x_y(arma::vec x, arma::vec y, string filename){
    /* Writing x and y data to text file data.txt */
    int n = x.n_rows;                              // number of elements, vec=column vector
    ofstream myfile (filename);                    // initializing and opening file
    if (myfile.is_open()){                         // if open (safety check)
        myfile << "     x          y    \n";       // header columns line
        scientific(myfile).precision(4);           // setting scientific output, 4 decimals
        
        for (int i=0; i<n; i++){                   // looping over lists
            myfile << x(i) << " " << y(i) << endl; // writing "x y" to file
        }
        myfile.close();                            // closing file
        cout << "Data saved as file " << filename << endl;
    }
    else {cout << "Unable to open file " << filename << endl;}  // print if unable to open file
}

arma::vec f(arma::vec x){
    return 100 * exp(-10*x);
}

arma::mat make_tridiagonal(int n, double signature[3]){
    arma::mat A = arma::mat(n, n).fill(0.0); // initializing matrix

    // setting first and last row manually
    A(  0,   0) = signature[1]; // b_1, A(y, x)
    A(  0,   1) = signature[2]; // c_1
    A(n-1, n-1) = signature[1]; // b_n
    A(n-1, n-2) = signature[0]; // a_n

    // setting all other values
    for (int i=1; i<n-1; i++){
        A(i, i-1) = signature[0]; // a_i+1 | (indexing diff,
        A(i, i  ) = signature[1]; // b_i+1 |  i.e i=3 -> b_4)
        A(i, i+1) = signature[2]; // c_i+1 |
    }

    return A;
}

arma::vec solve_general(arma::mat A, arma::vec g){
    /*
    Solves matrix equation Av = g, where A is a general tridiagonal matrix nxn
    with defined as sub-diagonal a, main diagonal b, and super-diagonal c.

    Example system, n=5:
        | b1 c1  0  0  0 |     | v1 |     | g1 |
        | a2 b2 c2  0  0 |     | v2 |     | g2 |
    A = |  0 a3 b3 c3  0 | v = | v3 | g = | g3 |
        |  0  0 a4 b4 c4 |     | v4 |     | g4 |
        |  0  0  0 a5 b5 |     | v5 |     | g5 |
    Where A and g are known.

    Returns solution vector v for a given A and g.
    */
    int n = A.n_rows;           // size of system, assumed A: n x n and g: n x 1
    arma::vec v = arma::vec(n); // solution vector

    arma::vec b_ = arma::vec(n); // temporary vectors for fwd. sweep values
    arma::vec g_ = arma::vec(n);

    // forwards sweep, setting first b_ and g_ manually
    b_(0) = A(0, 0); // b_1
    g_(0) = g(0);    // g_1
    for (int i=1; i<n; i=i+1){                           // i actually i+1 due to indexing
        b_(i) = A(i, i) - A(i, i-1)/b_(i-1) * A(i-1, i); // b_i - a_i/b__i-1 * c_i-1
        g_(i) = g(i) - A(i, i-1)/b_(i-1) * g_(i-1);      // g_i - a_i/b__i-1 * g__i-1
    }

    // backwards sweep, setting v_n manually
    v(n-1) = g_(n-1) / b_(n-1);
    for (int i=n-2; i>=0; i=i-1){ // starting loop at v_n-1, until v_1 (indexes n-2, 0)
        v(i) = (g_(i) - A(i+1, i)*v(i+1))/b_(i);        // (g__i - c_i * v_i+1) / b__i
    }

    return v; // v is somehow a factor n^2 too large
}

arma::vec solve_special(arma::vec g){
    /*
    Solves matrix equation Av = g, where A is a special tridiagonal matrix nxn
    defined by signature {-1, 2, -1}, giving
    sub-diagonal   a = {-1, ..., -1}
    main diagonal  b = { 2, ...,  2}
    super-diagonal c = {-1, ..., -1}

    Returns solution vector v for given g
    */
    int n = g.n_rows;

    arma::vec b_ = arma::vec(n); // temp value vectors
    arma::vec g_ = arma::vec(n);
    b_(0) = 2;    // b__1 = b_1
    g_(0) = g(0); // g__1 = g_1

    int j; // saving FLOPs by computing i-1 only once per loop
    for (int i = 1; i < n; i = i+1){ // '1' FLOP (required, not part of algorithm)
        j = i-1;                     // 1 FLOP
        b_(i) = 2 - 1/b_(j);         // 2 FLOPs
        g_(i) = g(i) + g_(j)/b_(j);  // 2 FLOPs
    }

    arma::vec v = arma::vec(n); // solution vector
    j = n-1;                    // quick repurposing to save a couple FLOPs
    v(j) = g_(j)/b_(j);         // v_n = g__n/b__n, 1 FLOP

    for (int i = n-2; i >= 0; i = i-1){ // '1' FLOP
        v(i) = (g_(i) + v(i+1))/b_(i);  // 3 FLOPs
    }

    return v / pow(n, 2);
}

int main(){
/* ======== PROBLEM 2 ======== */ /*
int n = 1001;                            // number of points, n+1 to get nice step size
arma::vec x = arma::linspace(0, 1, n);   // initializing x vector, n points in [0, 1]
arma::vec y = u(x);                      // vector with values for u(x), length n
write_x_y(x, y, "xu.txt");               // writing to file

/* ======== PROBLEM 7 ======== */ /*
double signature[3] = {-1, 2, -1};            // values on sub-, main, and super-diagonal
arma::mat A = make_tridiagonal(n, signature); // main matrix A (n x n)
arma::vec g = f(x);                           // g = f(x in [0, 1])

arma::vec v = solve_general(A, g); // solving system
write_x_y(x, v, "xv.txt"); // writing x,v to file

/* ======== PROBLEM 8 ======== */ /*
arma::vec N = arma::vec({11, 101, 1001, 10001}); // values of N to use
// storing x as row in matrix with a column for each N, padded with nan
arma::mat xi = arma::mat(N.max(), N.n_rows).fill(NAN);
arma::mat ui = arma::mat(xi); // storing u and v, same format as x
arma::mat vi = arma::mat(xi);
arma::mat Ai;                 // equation matrix A for each N
arma::vec gi;                 // RHS of matrix equation, f(x)

for (int i=0; i<N.n_rows; i++){
    xi.col(i).rows(0, N(i)-1) = arma::linspace(0, 1, N(i)); // leaving padding if N.max > N(i)
    Ai = make_tridiagonal(N(i), signature);                 // Tridiagonal matrix size nxn
    gi = f(xi.col(i).rows(0, N(i)-1));                      // RHS for n=N(i)

    ui.col(i) = u(xi.col(i));                          // analytical, can be passed whole
    vi.col(i).rows(0, N(i)-1) = solve_general(Ai, gi); // numerical, need specific rows due to derivation
}

// interpreting problem description as saving by text is no longer needed, taking shortcut:
xi.save("xi.txt"); ui.save("ui.txt"); vi.save("vi.txt");
cout << "Data saved to files" << endl;
*/
/* ======== PROBLEM 9 ======== */
// see function solve_special()

/* ======== PROBLEM 10 ======= */
arma::vec N_timeit = arma::vec({1e1, 1e2, 1e3, 1e4}); // values of N to use
int N_runs = 10; // number of times to run/average each solve

// setting up variables for use later
auto t1 = chrono::high_resolution_clock::now();
auto t2 = chrono::high_resolution_clock::now();
arma::mat duration = arma::mat(2, N_timeit.n_rows).fill(0.0); // storing averages in matrix
arma::mat A_timeit; double signature_timeit[3] = {-1.0, 2.0, -1.0};
arma::vec g_timeit;
arma::vec v_timeit;

for (int i=0; i<N_timeit.n_rows; i++){
    A_timeit = make_tridiagonal(N_timeit(i), signature_timeit); // matrix for current N
    g_timeit = f(arma::linspace(0, 1, N_timeit(i))) * pow(1/(N_timeit(i)-1), 2); // g = f(x)h^2
    for (int j=0; j<N_runs; j++){
        // TIMING GENERAL SOLUTION
        t1 = chrono::high_resolution_clock::now();
        v_timeit = solve_general(A_timeit, g_timeit);
        t2 = chrono::high_resolution_clock::now();
        // adding contribution to average for solve_general() at N(i)
        duration(0, i) += chrono::duration<double>(t2 - t1).count() / N_runs;

        // TIMING SPECIAL SOLUTION
        t1 = chrono::high_resolution_clock::now();
        v_timeit = solve_special(g_timeit);
        t2 = chrono::high_resolution_clock::now();
        duration(1, i) += chrono::duration<double>(t2 - t1).count() / N_runs;
    }
}

cout << "General Algorithm:" << duration.row(0) << endl;
cout << "Special Algorithm:" << duration.row(1) << endl;

// All went well
return 0;
}