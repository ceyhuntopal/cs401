#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <stdint.h>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Sparse>

template <typename T>
class Matrix {
public:

    Matrix() = default;

    Matrix(size_t rows, size_t cols)
        : data(rows, std::vector<T>(cols, 0)), rows(rows), cols(cols) {}

    Matrix(const std::string& filename) { readMatrixMarketFile(filename); }

    static Matrix<T> identity(size_t size) {
        Matrix<T> identity(size, size);
        for (size_t i = 0; i < size; ++i) {
            identity(i, i) = 1;
        }
        return identity;
    }

    std::vector<T> row(size_t row) const {
        if (row >= rows) throw std::out_of_range("Row index out of range");
        return data[row];
    }

    std::vector<T> col(size_t col) const {
        if (col >= cols) throw std::out_of_range("Column out of range");
        std::vector<T> column(rows);
        for (size_t i = 0; i < rows; ++i) {
            column[i] = data[i][col];
        }
        return column;
    }

    void resize(size_t newRows, size_t newCols) {
        data.resize(newRows);
        for (size_t i = 0; i < newRows; ++i) {
            data[i].resize(newCols, 0);
        }
        rows = newRows;
        cols = newCols;
    }

    void setRowToZero(size_t row) {
        if (row >= rows) throw std::out_of_range("Row index out of range");
        std::fill(data[row].begin(), data[row].end(), 0);
    }

    void setColumnToZero(size_t col) {
        if (col >= cols) throw std::out_of_range("Column index out of range");
        for (size_t i = 0; i < rows; ++i) {
            data[i][col] = 0;
        }
    }

    void setZero() {
        for (size_t i = 0; i < rows; ++i) {
            std::fill(data[i].begin(), data[i].end(), 0);
        }
    }

    void addValueToRow(size_t row, T value) {
        if (row >= rows) throw std::out_of_range("Row index out of range");
        for (auto& elem : data[row]) {
            elem += value;
        }
    }

    void multiplyRow(size_t row, T value) {
        if (row >= rows) throw std::out_of_range("Row index out of range");
        for (auto& elem : data[row]) {
            elem *= value;
        }
    }

    void addValueToColumn(size_t col, T value) {
        if (col >= cols) throw std::out_of_range("Column index out of range");
        for (size_t i = 0; i < rows; ++i) {
            data[i][col] += value;
        }
    }

    void multiplyColumn(size_t col, T value) {
        if (col >= cols) throw std::out_of_range("Column index out of range");
        for (size_t i = 0; i < rows; ++i) {
            data[i][col] *= value;
        }
    }

    void addVectorToRow(size_t row, const std::vector<T>& vec) {
        if (row >= rows) throw std::out_of_range("Row index out of range");
        if (vec.size() != cols) throw std::invalid_argument("Vector size must match the number of columns in the matrix");
        for (size_t j = 0; j < cols; ++j) {
            data[row][j] += vec[j];
        }
    }

    void addVectorToColumn(size_t col, const std::vector<T>& vec) {
        if (col >= cols) throw std::out_of_range("Column index out of range");
        if (vec.size() != rows) throw std::invalid_argument("Vector size must match the number of rows in the matrix");
        for (size_t i = 0; i < rows; ++i) {
            data[i][col] += vec[i];
        }
    }

    void addRowToRow(const Matrix<T>& srcMatrix, size_t srcRow, size_t destRow) {
        if (srcRow >= srcMatrix.rows || destRow >= rows) throw std::out_of_range("Row index out of range");
        if (cols != srcMatrix.cols) throw std::invalid_argument("Column count must match for row addition");
        for (size_t j = 0; j < cols; ++j) {
            data[destRow][j] += srcMatrix(srcRow, j);
        }
    }

    void addColToCol(const Matrix<T>& srcMatrix, size_t srcCol, size_t destCol) {
        if (srcCol >= srcMatrix.cols || destCol >= cols) throw std::out_of_range("Column index out of range");
        if (rows != srcMatrix.rows) throw std::invalid_argument("Row count must match for column addition");
        for (size_t i = 0; i < rows; ++i) {
            data[i][destCol] += srcMatrix(i, srcCol);
        }
    }

    Matrix<T> submatrix(const std::vector<size_t>& rowIndices, const std::vector<size_t>& colIndices) const {
        Matrix<T> submatrix(rowIndices.size(), colIndices.size());
        for (size_t i = 0; i < rowIndices.size(); ++i) {
            for (size_t j = 0; j < colIndices.size(); ++j) {
                if (rowIndices[i] >= rows || colIndices[j] >= cols) throw std::out_of_range("Index out of range");
                submatrix(i, j) = data[rowIndices[i]][colIndices[j]];
            }
        }
        return submatrix;
    }

    Matrix<T> transpose() const {
        Matrix<T> result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }

    T& operator()(size_t row, size_t col) {
        if (row >= rows || col >= cols) throw std::out_of_range("Index out of range");
        return data[row][col];
    }

    const T& operator()(size_t row, size_t col) const {
        if (row >= rows || col >= cols) throw std::out_of_range("Index out of range");
        return data[row][col];
    }

    Matrix<T> operator*(const Matrix<T>& other) const {
        if (cols != other.rows) throw std::invalid_argument("Matrix dimensions do not match for multiplication");

        Matrix<T> result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result(i, j) += data[i][k] * other(k, j);
                }
            }
        }
        return result;
    }

    std::vector<std::vector<T>> data_() { return data; }
    
    std::vector<T>& operator [] (size_t index){
        return data[index];
    }

    size_t totalSize() const {
        return rows * cols;
    }

    void print() const {
        for (const auto& row : data) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    std::vector<std::vector<T>> data;
    size_t rows, cols;

private:

    void readMatrixMarketFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file");
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line[0] == '%') {
                continue;
            } else {
                std::istringstream iss(line);
                size_t numRows, numCols, numEntries;
                iss >> numRows >> numCols >> numEntries;

                resize(numRows, numCols);

                for (size_t i = 0; i < numEntries; ++i) {
                    size_t row, col;
                    T value;
                    file >> row >> col >> value;
                    data[row - 1][col - 1] = value;
                }
                break;
            }
        }
    }
};
#endif
