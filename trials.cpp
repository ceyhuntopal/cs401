#include "Matrix.h"
#include "constraint_app_parallel.h"
#include <chrono>
#include <ctime>
#include <random>
#include <omp.h>
#include <unordered_set>

template <typename T>
void vector_printer(const std::vector<T>& vec) {
    for (const auto& v : vec)
        std::cout << v << " ";
    std::cout << std::endl;
}

std::vector<double> vector_generator(size_t size, double fbound) {
    std::random_device rd;  // Non-deterministic generator
    std::mt19937 gen(rd()); // Mersenne Twister generator seeded with rd()

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

    std::uniform_int_distribution<> distr_node(0, size - 1);
    std::uniform_int_distribution<> distr_m(1, mbound);
    std::uniform_real_distribution<> distr_coeff(0.01, coeff_bound);

    std::unordered_set<size_t> usedNodes;

    for (size_t i{0}; i < cnumber; ++i) {
        size_t sIdx;
        
        do {
            sIdx = distr_node(gen);
        } while (usedNodes.find(sIdx) != usedNodes.end());
        usedNodes.insert(sIdx);

        double sCoeff = distr_coeff(gen);
        auto s = Term(sIdx, sCoeff);

        std::vector<Term> m(distr_m(gen));
        for (auto& term : m) {
            size_t mIdx;
            do {
                mIdx = distr_node(gen);
            } while (usedNodes.find(mIdx) != usedNodes.end());
            double mCoeff = distr_coeff(gen);
            term = Term(mIdx, mCoeff);
        }

        double gap = distr_coeff(gen);
        eq.addConstraint(s, m, gap);
    }
    return eq;
}

void constraint_parallel(primaryEquation& eq) {
    clock_t start_time, stop_time;
    double totalTime;
    start_time = clock();

    int rank, size, nnode;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    nnode = eq.numc(); // Assuming a method to get the number of nodes

    auto T = TransformationMatrix<double>(eq, nnode);

    size_t constraints_per_proc = T.num_constraints() / size;
    size_t start = rank * constraints_per_proc;
    size_t end = (rank == size - 1) ? T.num_constraints() : start + constraints_per_proc;

        T.applyConstraints(start, end);

    std::vector<double> flat_matrix(T.get().rows * T.get().cols);
    size_t idx = 0;
    for (const auto& row : T.get().data) {
        std::copy(row.begin(), row.end(), flat_matrix.begin() + idx);
        idx += row.size();
    }

    std::vector<double> reduced_flat_matrix(flat_matrix.size());

    MPI_Allreduce(flat_matrix.data(), reduced_flat_matrix.data(), flat_matrix.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    size_t rows = T.get().rows;
    size_t cols = T.get().cols;
    Matrix<double> reduced_matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            reduced_matrix[i][j] = reduced_flat_matrix[i * cols + j] - (i == j ? size - 1 : 0);
        }
    }

    if (rank == 0) {
        stop_time = clock();
        totalTime = (stop_time - start_time) / (double)CLOCKS_PER_SEC;
        std::cout << totalTime << " ";
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    primaryEquation eq;
    if (rank == 0) {
        // Generate the primaryEquation object
        std::string matrixfolder = "bcsstk21";
        std::string matrixfile = matrixfolder + ".mtx";
        auto K = Matrix<double>(matrixfolder + "/" + matrixfile);
        int nnode = K[0].size();
        eq = constraint_generator(nnode, 3200, 3);
    }

    // Serialize the primaryEquation object
    std::vector<double> serialized_eq = eq.serialize();
    int eq_size = serialized_eq.size();

    // Broadcast the size of the serialized data
    MPI_Bcast(&eq_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize the vector on other processes to receive the data
    if (rank != 0) {
        serialized_eq.resize(eq_size);
    }

    // Broadcast the serialized equation system
    MPI_Bcast(serialized_eq.data(), eq_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Deserialize the equation system on all processes
    if (rank != 0) {
        eq.deserialize(serialized_eq);
    }

    // Run the parallel algorithm
    constraint_parallel(eq);

    MPI_Finalize();
    std::cout << std::endl;

    return 0;
}
