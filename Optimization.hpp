#ifndef DENG_OPTIMIZATION_HPP
#define DENG_OPTIMIZATION_HPP

#include <iostream>
#include <complex>
#include <armadillo>

namespace Deng
{
    namespace Optimization
    {
        class Conj_Grad_Min
        {
        protected:
            unsigned int _dim_para;
            const double _epsilon;//precision of parametric space
            const unsigned int _max_iteration;//maximum allowed trials when searching for minimum
        public:
            //coordinate space
            arma::Col<double> coordinate;
            //search directions
            arma::Col<double> current_search_direction;
            arma::Col<double> previous_search_direction;
            //constructor to
            Conj_Grad_Min(unsigned int dim_para, double epsilon = 1E-3, unsigned int max_iteration = 50) : _dim_para(dim_para), _epsilon(epsilon), _max_iteration(max_iteration)
            {
                coordinate = arma::zeros<arma::Col<double> >(_dim_para);
                current_search_direction = arma::zeros<arma::Col<double> >(_dim_para);
                previous_search_direction = arma::zeros<arma::Col<double> >(_dim_para);
            };

            //if not giving a start coordinate, we start from 0
            virtual arma::Col<double> Conj_Grad_Search(arma::Col<double> start_coordinate = 0);
            //optimization in 1D
            virtual double OneD_Minimum(const arma::Col<double> start_coordinate, const arma::Col<double> search_direction_given);
            //virtual double Golden_Section(const arma::Col<double> give_coordinate, const arma::Col<double> give_search_direction);

            //calculate the target function, usually called during optimization in 1D
            //must be given as it is pure virtual
            virtual double Target_function(const arma::Col<double> & coordinate_given) = 0;
            //return the value of target function, also give the gradient information to the reference, usually called when choosing the conjugate gradient
            //must be given as it is pure virtual
            virtual double Target_function_and_negative_gradient(const arma::Col<double>& coordinate_given, arma::Col<double>& negative_gradient) = 0;


        };
    }
}

/*
//an instance/test case for the realization of optimization
class GOAT_OCT : public Deng::Optimization::Conj_Grad_Min
{
    using Conj_Grad_Min::Conj_Grad_Min;
    virtual double Target_function(const arma::Col<double> & coordinate_given) override;
    virtual double Target_function_and_negative_gradient(const arma::Col<double> & coordinate_given, arma::Col<double>& negative_gradient) override;
};
double GOAT_OCT::Target_function(const arma::Col<double> & coordinate_given)
{
    arma::Col<double> x = { 5.0, 1.0};
    arma::Mat<double> A = { {4.0, 1.0}, {1.0, 3.0} };
    x = coordinate_given - x;

    return arma::as_scalar(x.t()*A*x);
}
double GOAT_OCT::Target_function_and_negative_gradient(const arma::Col<double> & coordinate_given, arma::Col<double>& negative_gradient)
{
    arma::Col<double> x = { 5.0, 1.0};
    arma::Mat<double> A = { {4.0, 1.0}, {1.0, 3.0} };
    x = coordinate_given - x;

    negative_gradient = -A*x;

    return arma::as_scalar(x.t()*A*x);
}
*/

#endif // DENG_OPTIMIZATION_HPP
