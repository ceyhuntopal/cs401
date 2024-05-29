#include "Matrix.h"
#include "constraint_app_parallel.h"
#include <chrono>
#include <ctime>
#include <random>
#include <omp.h>
#include <unordered_set>
#include <mpi.h>

template <typename T>
void vector_printer(const std::vector<T>& vec) {
    for (const auto& v : vec)
        std::cout << v << " ";
    std::cout << std::endl;
}

std::vector<double> vector_generator(size_t size, double fbound) {
    std::random_device rd;  // Non-deterministic generator
    std::mt19937 gen(rd()); // Mersenne Twister generator seeded with rd()

    // Define the distribution range
    std::uniform_real_distribution<> distr_coeff(0.0, fbound);

    std::vector<double> vec(size);
    for (auto& v : vec) {
        v = distr_coeff(gen);
    }
    return vec;
}

primaryEquation constraint_generator(size_t size, size_t cnumber, size_t mbound, double coeff_bound = 10) {
    primaryEquation eq;

    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the distribution range
    std::uniform_int_distribution<> distr_node(0, size - 1);
    std::uniform_int_distribution<> distr_m(1, mbound);
    std::uniform_real_distribution<> distr_coeff(0.01, coeff_bound);

    std::unordered_set<size_t> usedNodes;

    for (size_t i{0}; i < cnumber; ++i) {
        size_t sIdx;
        
        // Create random slave node
        do {
            sIdx = distr_node(gen);
        } while (usedNodes.find(sIdx) != usedNodes.end());
        // Add created node index to used nodes to prevent reusage
        usedNodes.insert(sIdx);

        double sCoeff = distr_coeff(gen);
        auto s = Term(sIdx, sCoeff);

        // Create random master nodes
        std::vector<Term> m(distr_m(gen));
        for (auto& term : m) {
            size_t mIdx;
            do {
                mIdx = distr_node(gen);
            } while (usedNodes.find(mIdx) != usedNodes.end());
            double mCoeff = distr_coeff(gen);
            term = Term(mIdx, mCoeff);
        }

        // Create random gap
        double gap = distr_coeff(gen);
        // Add constraint to equation system
        eq.addConstraint(s, m, gap);
    }
    return eq;
}

void constraint_parallel(std::string matrixFolder, size_t numConstraints) {

    // Initialize process rank and global process number as size
    int rank, size, nnode;
    int eq_size;
    primaryEquation eq;
    std::vector<double> serialized_eq;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Read K matrix
        std::string matrixfile = matrixFolder + ".mtx";
        auto K = Matrix<double>(matrixFolder + "/" + matrixfile);
        nnode = K[0].size();
        if (nnode < numConstraints) {
            std::cerr << "Constraint number cannot be bigger than matrix size" << std::endl;
            std::abort();
        }
        // Construct constraint equation system
        eq = constraint_generator(nnode, numConstraints, 3); // Max. three master terms in an equation
        serialized_eq = eq.serialize();
        eq_size = serialized_eq.size();
    }

    // Broadcast node number and equation system size
    MPI_Bcast(&nnode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eq_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize the vector on other processes to receive the data
    if (rank != 0) {
        serialized_eq.resize(eq_size);
    }

    // Broadcast the serialized equation system
    MPI_Bcast(serialized_eq.data(), eq_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        eq.deserialize(serialized_eq);
    }

    // Start timer
    double start_time, stop_time;
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before starting the timer
    start_time = MPI_Wtime();

    // Initialize transformation matrix
    auto T = TransformationMatrix<double>(eq, nnode);

    // Initialize job start and end per process
    size_t constraints_per_proc = T.num_constraints() / size;
    size_t start = rank * constraints_per_proc;
    size_t end = (rank == size - 1) ? T.num_constraints() : start + constraints_per_proc;

    // Perform operation (constraint application)
    T.applyConstraints(start, end);

    // Flatten the matrix for MPI_Allreduce
    std::vector<double> flat_matrix(T.get().rows * T.get().cols);
    size_t idx = 0;
    for (const auto& row : T.get().data) {
        std::copy(row.begin(), row.end(), flat_matrix.begin() + idx);
        idx += row.size();
    }

    // Prepare a buffer for the reduced result
    std::vector<double> reduced_flat_matrix(flat_matrix.size());

    // Perform the non-blocking allreduce operation
    MPI_Request request;
    MPI_Iallreduce(flat_matrix.data(), reduced_flat_matrix.data(), flat_matrix.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);

    // Perform other computations if possible while waiting for the reduction to complete

    // Wait for the non-blocking allreduce to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    // Convert the flattened result back into matrix form
    size_t rows = T.get().rows;
    size_t cols = T.get().cols;
    Matrix<double> reduced_matrix(rows, cols);

    # pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            reduced_matrix[i][j] = reduced_flat_matrix[i * cols + j] - (i == j ? size - 1 : 0);
        }
    }

    // Print process completion info
    if (rank == 0) {
        stop_time = MPI_Wtime();
        double totalTime = stop_time - start_time;
        std::cout << "Process finished in " << totalTime << " seconds." << std::endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int numConstraints;
    sscanf(argv[2], "%d", &numConstraints);
    
    // Run parallel algorithm
    constraint_parallel(argv[1], numConstraints);

    MPI_Finalize();

    return 0;
}
