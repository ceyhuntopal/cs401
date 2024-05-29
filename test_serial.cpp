#include "Matrix.h"
#include "constraint_app_parallel.h"
#include <chrono>
#include <ctime>
#include <random>

template <typename T>
void vector_printer(std::vector<T> vec) {
    for(int i{0}; i < vec.size(); ++i)
        std::cout << vec[i] << " ";
    std::cout << std::endl;
}

std::vector<double> vector_generator(size_t size, double fbound) {
    std::random_device rd;  // Non-deterministic generator
    std::mt19937 gen(rd()); // Mersenne Twister generator seeded with rd()
    
    // Define the distribution range
    std::uniform_real_distribution<> distr_coeff(0.0, fbound);

    std::vector<double> vec(size);
    for (size_t i{0}; i < vec.size(); ++i) {
        vec[i] = distr_coeff(gen);
    }
    return vec;
}

primaryEquation constraint_generator(size_t size, size_t cnumber, size_t mbound, double coeff_bound = 10) {
    
    primaryEquation eq;

    std::random_device rd;  // Non-deterministic generator
    std::mt19937 gen(rd()); // Mersenne Twister generator seeded with rd()
    
    // Define the distribution range
    std::uniform_int_distribution<> distr_node(0, size-1);
    std::uniform_int_distribution<> distr_m(1, mbound);
    std::uniform_real_distribution<> distr_coeff(0.01, coeff_bound);

    std::vector<size_t> usedNodes(1);
    size_t sIdx, mIdx;
    double sCoeff, mCoeff, gap;

    for (size_t i{0}; i < cnumber; ++i) {
        // Create random slave node
        while(true) {
            sIdx = distr_node(gen);
            if (std::find(usedNodes.begin(), usedNodes.end(), sIdx)==usedNodes.end())
                break;
        }
        sCoeff = distr_coeff(gen);
        auto s = Term(sIdx, sCoeff);

        // Add created node index to used nodes to prevent reusage
        usedNodes[usedNodes.size() - 1] = sIdx;
        usedNodes.resize(usedNodes.size() + 1); 
        
        // Create random master nodes
        std::vector<Term> m(distr_m(gen));
        for(size_t i{0}; i < m.size(); ++i) {
            while(true) {
                mIdx = distr_node(gen);
                if (std::find(usedNodes.begin(), usedNodes.end(), mIdx)==usedNodes.end())
                    break;
            }
            mCoeff = distr_coeff(gen);
            m[i] = Term(mIdx, mCoeff);
        }

        // Create random gap
        gap = distr_coeff(gen);

        // Add constraint to equation system
        eq.addConstraint(s, m, gap);
    }
    return eq;
}

void constraint_serial(std::string matrixFolder, size_t numConstraints) {
    clock_t start_time, stop_time;
    double totalTime;
    start_time = clock();

    std::string matrixfile = matrixFolder + ".mtx";
    auto K = Matrix<double>(matrixFolder + "/" + matrixfile);
    size_t nnode = K[0].size();

    primaryEquation eq = constraint_generator(nnode, numConstraints, 3);

    auto T = TransformationMatrix<double>(eq, nnode);

    // std::cout << "Initial Matrix:" << std::endl;
    // T.print();

    T.applyConstraints();

    stop_time = clock();
	totalTime = (stop_time - start_time) / (double)CLOCKS_PER_SEC;

    // std::cout << "Resulting matrix:" << std::endl;
    // T.print();
    
    std::cout << "Process finished in " << totalTime << " seconds." << std::endl;
}

int main(int argc, char* argv[]) {
    int numConstraints;
    sscanf(argv[2], "%d", &numConstraints);
    
    // Run parallel algorithm
    constraint_serial(argv[1], numConstraints);

    return 0;
}
