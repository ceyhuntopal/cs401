#ifndef CONSTRAINT_APP_PARALLEL_H
#define CONSTRAINT_APP_PARALLEL_H

#include <mpi.h>
#include <stdio.h>
#include <vector>
#include "Matrix.h"

class Term
{
public:
    Term() = default;
    Term(size_t index, float coeff) : index_(index), coeff_(coeff) {}
    size_t index() { return index_; }
    float coeff() { return coeff_; }
    void print()
    {
        std::cout << "(" << index_ << ", " << coeff_ << ")";
    }

    std::vector<double> serialize() const
    {
        return {static_cast<double>(index_), static_cast<double>(coeff_)};
    }

    // Deserialization function for Term
    void deserialize(const std::vector<double> &data, size_t &idx)
    {
        index_ = static_cast<size_t>(data[idx++]);
        coeff_ = static_cast<float>(data[idx++]);
    }

private:
    size_t index_;
    float coeff_;
};

class Constraint
{
public:
    Constraint() = default;

    Constraint(Term s, std::vector<Term> m, float gap)
        : secondaryTerm_(s), primaryTerms_(m), gap_(gap) {}

    Term secondaryTerm() { return secondaryTerm_; }
    std::vector<Term> primaryTerms() { return primaryTerms_; };
    float gap() { return gap_; }
    void setGap(float new_gap) { gap_ = new_gap; }

    void print()
    {
        std::cout << "s = (" << secondaryTerm_.index() << ", " << secondaryTerm_.coeff() << ")\t\t| ";
        std::cout << "m = {";
        for (size_t i{0}; i < primaryTerms_.size(); ++i)
            std::cout << "(" << primaryTerms_[i].index() << ", " << primaryTerms_[i].coeff() << ") ";
        std::cout << "}" << std::endl;
    }

    std::vector<double> serialize() const
    {
        std::vector<double> data;
        // Serialize secondaryTerm_
        auto secondary_data = secondaryTerm_.serialize();
        data.insert(data.end(), secondary_data.begin(), secondary_data.end());
        // Serialize primaryTerms_
        data.push_back(static_cast<double>(primaryTerms_.size()));
        for (const auto &term : primaryTerms_)
        {
            auto primary_data = term.serialize();
            data.insert(data.end(), primary_data.begin(), primary_data.end());
        }
        // Serialize gap_
        data.push_back(static_cast<double>(gap_));
        return data;
    }

    void deserialize(const std::vector<double> &data, size_t &idx)
    {
        // Deserialize secondaryTerm_
        secondaryTerm_.deserialize(data, idx);
        // Deserialize primaryTerms_
        size_t primary_size = static_cast<size_t>(data[idx++]);
        primaryTerms_.resize(primary_size);
        for (auto &term : primaryTerms_)
        {
            term.deserialize(data, idx);
        }
        // Deserialize gap_
        gap_ = static_cast<float>(data[idx++]);
    }

private:
    Term secondaryTerm_;
    std::vector<Term> primaryTerms_;
    float gap_;
};

class primaryEquation
{
public:
    primaryEquation() = default;

    void addConstraint(Term s, std::vector<Term> m, float gap)
    {
        auto c = Constraint(s, m, gap);
        constraints_.resize(constraints_.size() + 1);
        constraints_[constraints_.size() - 1] = c;
    }

    void pushBackInactiveTerm(size_t term)
    {
        inactiveDOFS.resize(inactiveDOFS.size() + 1);
        inactiveDOFS[inactiveDOFS.size() - 1] = term;
    }

    size_t getTermIdx(size_t term)
    {
        for (size_t idx{0}; idx < activeCols_.size(); ++idx)
        {
            if (activeCols_[idx] == term)
                return idx;
        }
        return -1;
    }

    void removeFromActiveCols(size_t term)
    {
        for (size_t colIdx{0}; colIdx < activeCols_.size(); ++colIdx)
        {
            if (term == activeCols_[colIdx])
                activeCols_.erase(activeCols_.begin() + colIdx);
        }
    }

    void setActiveCols(std::vector<size_t> activeCols)
    {
        activeCols_ = activeCols;
    }

    void print()
    {
        for (size_t i{0}; i < constraints_.size(); ++i)
            constraints_[i].print();
        std::cout << std::endl;
    }

    // Serialization function
    std::vector<double> serialize()
    {
        std::vector<double> data;
        // Serialize constraints
        for (const auto &constraint : constraints_)
        {
            std::vector<double> constraint_data = constraint.serialize();
            data.insert(data.end(), constraint_data.begin(), constraint_data.end());
        }
        // Serialize other members
        data.push_back(static_cast<double>(activeCols_.size()));
        data.insert(data.end(), activeCols_.begin(), activeCols_.end());
        return data;
    }

    // Deserialization function
    void deserialize(const std::vector<double> &data)
    {
        size_t idx = 0;
        // Deserialize constraints
        constraints_.clear();
        while (idx < data.size() - 1)
        {
            Constraint constraint;
            constraint.deserialize(data, idx);
            constraints_.push_back(constraint);
        }
        // Deserialize other members
        size_t size = static_cast<size_t>(data[idx++]);
        activeCols_.assign(data.begin() + idx, data.begin() + idx + size);
        idx += size;
    }

    std::vector<Constraint> constraints() { return constraints_; }
    std::vector<size_t> activeCols() { return activeCols_; }
    size_t numc() { return constraints_.size(); }
    // std::vector<size_t> activeRows() { return activeRows_; }
    // std::vector<size_t> inactiveDofs() { return inactiveDOFS; }
    // size_t activeColNum() { return activeCols_.size(); }

private:
    std::vector<Constraint> constraints_;
    std::vector<size_t> activeRows_;
    std::vector<size_t> activeCols_;
    std::vector<size_t> inactiveDOFS;
};

template <typename T>
class TransformationMatrix
{
public:
    TransformationMatrix() = default;

    TransformationMatrix(primaryEquation eq, size_t n)
    {
        eq_ = eq;
        std::vector<size_t> activeCols(n);
        for (size_t i{0}; i < activeCols.size(); ++i)
            activeCols[i] = i;
        eq_.setActiveCols(activeCols);
        T_ = Matrix<T>::identity(n);
        gaps_ = std::vector<T>(n);
    }

    void applyConstraints()
    {
        for (auto& constraint : eq_.constraints())
        {
            Term s = constraint.secondaryTerm();
            gaps_[s.index()] = constraint.gap() / s.coeff();
            size_t secondaryTermIdx = eq_.getTermIdx(s.index());
            T_.setColumnToZero(secondaryTermIdx);

            for (auto term : constraint.primaryTerms())
            {
                size_t primaryTermIdx = eq_.getTermIdx(term.index());
                std::vector<T> primaryRow = T_.row(primaryTermIdx);
                std::transform(primaryRow.begin(), primaryRow.end(), primaryRow.begin(),
                               [&term, &s](auto &c)
                               { return c * term.coeff() / s.coeff(); });
                T_.addVectorToRow(secondaryTermIdx, primaryRow);
            }
        }
    }

    void applyConstraint(size_t i)
    {
        auto constraint = eq_.constraints()[i];
        Term s = constraint.secondaryTerm();
        gaps_[s.index()] = constraint.gap() / s.coeff();
        size_t secondaryTermIdx = eq_.getTermIdx(s.index());
        T_.setColumnToZero(secondaryTermIdx);

        for (auto term : constraint.primaryTerms())
        {
            size_t primaryTermIdx = eq_.getTermIdx(term.index());
            std::vector<T> primaryRow = T_.row(primaryTermIdx);
            std::transform(primaryRow.begin(), primaryRow.end(), primaryRow.begin(),
                            [&term, &s](auto &c)
                            { return c * term.coeff() / s.coeff(); });
            T_.addVectorToRow(secondaryTermIdx, primaryRow);
        }
    }


    void applyConstraints(size_t start, size_t end)
    {
        for (size_t i = start; i < end; ++i)
        {
            auto constraint = eq_.constraints()[i];
            Term s = constraint.secondaryTerm();
            gaps_[s.index()] = constraint.gap() / s.coeff(); // index out of bounds
            size_t secondaryTermIdx = eq_.getTermIdx(s.index());
            T_.setColumnToZero(secondaryTermIdx);
            for (auto term : constraint.primaryTerms())
            {
                size_t primaryTermIdx = eq_.getTermIdx(term.index());
                std::vector<T> primaryRow = T_.row(primaryTermIdx);
                std::transform(primaryRow.begin(), primaryRow.end(), primaryRow.begin(),
                               [&term, &s](auto &c)
                               { return c * term.coeff() / s.coeff(); });
                T_.addVectorToRow(secondaryTermIdx, primaryRow);
            }
        }
    }

    Matrix<T> &get() { return T_; }

    std::vector<T> gaps() { return gaps_; };

    primaryEquation equationSystem() { return eq_; }

    size_t num_constraints() { return eq_.constraints().size(); }

    std::vector<T> modified_gaps() { return gaps_; }

    void print() { T_.print(); }

private:
    primaryEquation eq_;
    Matrix<T> T_;
    std::vector<T> gaps_;
};

#endif
