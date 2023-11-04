// svd.cpp
#include <Rcpp.h>
using namespace Rcpp;
#include <vector>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>


// [[Rcpp::export]]
List performSVD(const Eigen::MatrixXd& X) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    // TODO: Further operations based on U, V, and E

    return List::create(Named("U") = U, Named("V") = V);
}


// [[Rcpp::export]]
double computeR(const Eigen::MatrixXd& X, const Eigen::MatrixXd& V) {
    Eigen::MatrixXd XV = X * V;
    double result = 0;

    // Summation based on the given formula
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.rows(); ++j) {
            result += (XV.row(i) - XV.row(j)).norm();
        }
    }

    return result;
}

// [[Rcpp::export]]
Eigen::MatrixXd gradientDescent(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U, const Eigen::MatrixXd& E, const Eigen::MatrixXd& D, double lambda, int maxIterations, double learningRate) {
    int n = X.cols(); // Assuming V has the same number of columns as X
    Eigen::MatrixXd V = Eigen::MatrixXd::Random(n, n); // Random initialization

    for (int i = 0; i < maxIterations; ++i) {
        // Compute the gradient of the objective function with respect to V
        //Eigen::MatrixXd gradientU = 2 * (U * X - V * E.transpose()) * X.transpose();
        Eigen::MatrixXd gradientV = 2 * (V.transpose() * X.transpose() - U * E) * X + 2 * lambda * V.transpose() * (D * (X * V)).transpose() * (D * (X * V));

        // Update U and V using the computed gradient
        //U = U - learningRate * gradientU;
        V = V - learningRate * gradientV.transpose();
    }

    return V; // Depending on your needs, you can also return U
}




