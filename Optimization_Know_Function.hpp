#ifndef DENG_OPTIMIZATION_KNOW_FUNCTION_HPP
#define DENG_OPTIMIZATION_KNOW_FUNCTION_HPP

#include <iostream>
#include <complex>
#include <armadillo>


namespace Deng
{
	namespace Optimization
	{
		namespace Know_Function
		//this name space is for the optimizing a function when we only know the function value
		{
			template<typename real>
			class Target_function
			{
			public:
				//we only know the function value
				virtual real function_value(const arma::Col<real>& coordinate_given) = 0;
				virtual ~Target_function() = default;
			};

			template<typename real>
			class Minimization
			{
			protected:
				unsigned int _dim_para;
				real _epsilon;
				unsigned int _max_iteration;
				//must know the target function
				//since the target function could be very complex, I prefer to write it as a class, rather than just a function pointer
				Target_function<real>* f;
			public:
				//coordinate in parametric space
				//suppose to store the initial coordinate, and the final optimized parameters
				mutable arma::Col<real> coordinate;
				//omit the default constructor
				//i.e. Must know the parameters before declare this object!!!
				Minimization(unsigned int dim_para, real epsilon, unsigned int max_iteration) : _dim_para(dim_para), _epsilon(epsilon), _max_iteration(max_iteration)
				{
					coordinate = arma::zeros<arma::Col<real> >(_dim_para);
					f = nullptr;
				};
				//This function will be covered in namespace Know_Gradient
				//THIS MIGHT BE DANGEROUS!!!!!
				void Assign_Target_Function(Target_function<real>* pt_f) { f = pt_f; };

				virtual ~Minimization()
				{
					delete f;
				};
			};

		}
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

#endif // DENG_OPTIMIZATION_KNOW_FUNCTION_HPP
