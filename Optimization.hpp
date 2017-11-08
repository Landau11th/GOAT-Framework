#ifndef DENG_OPTIMIZATION_HPP
#define DENG_OPTIMIZATION_HPP

#include <iostream>
#include <complex>
#include <armadillo>
#include <cassert>


#include <ctime>

namespace Deng
{
	namespace Optimization
	{
		template<typename real>
		class Target_function
		{
		public:
			//we only know the function value
			//every pure virtual function must be defined
			//if certain function is unknown, one could define it as final and use assert in the body
			//example is given below. search "//Target_function example"
			virtual real function_value(const arma::Col<real>& coordinate_given) = 0;
			virtual arma::Col<real> negative_gradient(const arma::Col<real>& coordinate_given, real &function_value) = 0;
			virtual arma::Mat<real> Hessian(const arma::Col<real>& coordinate_given, real &function_value, arma::Col<real> &negative_gradient) = 0;
			virtual void higher_order(const arma::Col<real>& coordinate_given, const real order) = 0;
			virtual ~Target_function() = default;
		};

		//this class is to optimize a function when we only know the function value
		template<typename real>
		class Min_Know_Function
		{
		protected:
			unsigned int _dim_para;
			real _epsilon;
			unsigned int _max_iteration;
			//must know the target function
			//since the target function could be very complex, I prefer to write it as a class, rather than just a function pointer
			Target_function<real> *f;
		public:
			//coordinate in parametric space
			//suppose to store the initial coordinate, and the final optimized parameters
			mutable arma::Col<real> coordinate;
			//omit the default constructor
			//i.e. Must know the parameters before declare this object!!!
			Min_Know_Function(unsigned int dim_para, real epsilon, unsigned int max_iteration) : _dim_para(dim_para), _epsilon(epsilon), _max_iteration(max_iteration)
			{
				coordinate = arma::zeros<arma::Col<real> >(_dim_para);
				f = nullptr;
			};
			//f is Deng::Optimization::Target_function<real>
			virtual void Assign_Target_Function(Target_function<real> *pt_f)
			{
				f = pt_f;
			};

			virtual ~Min_Know_Function() = default;
		};


		//this class is to optimize a function when we know the gradient
		template<typename real>
		class Min_Know_Gradient : public Min_Know_Function<real>
		{
		public:
			//search directions
			mutable arma::Col<real> search_direction;

			//Target_function<real>* f;
			////to cover the defination of Assign_Target_Function in Deng::Optimization::Know_Function::Minimization<real>
			////mainly because we want f points to Deng::Optimization::Know_Gradient::Target_function
			////THIS MIGHT BE DANGEROUS!!!!!
			//void Assign_Target_Function(Target_function<real>* pt_f) { f = pt_f; };

			//using Min_Know_Function<real>::Min_Know_Function;
			//omit the default constructor.
			//Must know the parameters before declare this object
			Min_Know_Gradient(unsigned int dim_para, real epsilon, unsigned int max_iteration) : Min_Know_Function<real>(dim_para, epsilon, max_iteration)
			{
				search_direction = arma::zeros<arma::Col<real> >(dim_para);
			};
			virtual ~Min_Know_Gradient() = default;
		};

		//find minimum with conjugate gradient method
		template<typename real>
		class Min_Conj_Grad : public Min_Know_Gradient<real>
		{
		public:
			//previous search directions
			//it is needed in conj grad for a better search direction
			mutable arma::Col<real> previous_search_direction;
			//constructor
			//using Min_Know_Gradient<real>::Min_Know_Gradient;
			Min_Conj_Grad(unsigned int dim_para, real epsilon, unsigned int max_iteration) : Min_Know_Gradient<real>(dim_para, epsilon, max_iteration)
			{
				previous_search_direction = arma::zeros<arma::Col<real> >(dim_para);
			};

			//Here I implicitly assume the format of the 1D optimization function!!!!!!
			real(*Opt_1D)(const arma::Col<real> start_coordinate, const arma::Col<real> search_direction_given, Target_function<real> *f, const unsigned int max_iteration, const real epsilon);

			//if not giving a start coordinate, we start from 0
			virtual arma::Col<real> Conj_Grad_Search(arma::Col<real> start_coordinate = 0) const;
			//optimization in 1D
			//virtual real OneD_Minimum(const arma::Col<real> start_coordinate, const arma::Col<real> search_direction_given) const;
			//virtual double Golden_Section(const arma::Col<double> give_coordinate, const arma::Col<double> give_search_direction);

			virtual ~Min_Conj_Grad() = default;
		};

		//a frequently used oned search
		template<typename real>
		real OneD_Golden_Search(const arma::Col<real> start_coordinate, const arma::Col<real> search_direction_given, Target_function<real> *f, const unsigned int max_iteration, const real epsilon);



	}
}
//Target_function example
/*
typedef double real;
class test_Target_function : public Deng::Optimization::Target_function<real>
{
	virtual real function_value(const arma::Col<real>& coordinate_given) override;
	virtual arma::Col<real> negative_gradient(const arma::Col<real>& coordinate_given, real &function_value) override;
	virtual arma::Mat<real> Hessian(const arma::Col<real>& coordinate_given, real &function_value, arma::Col<real> &negative_gradient)
	{
		assert(false && "Hessian is not known!");
		return 0;
	};
	virtual void higher_order(const arma::Col<real>& coordinate_given, const real order)
	{
		assert(false && "Higher order derivatives are not known!");
	};
};
real test_Target_function::function_value(const arma::Col<real>& coordinate_given)
{
	arma::Col<double> x = { 5.0, 1.0};
	arma::Mat<double> A = { {4.0, 1.0}, {1.0, 3.0} };
	x = coordinate_given - x;

	return arma::as_scalar(x.t()*A*x);
}
arma::Col<real> test_Target_function::negative_gradient(const arma::Col<real>& coordinate_given, real &function_value)
{
	arma::Col<double> x = { 5.0, 1.0};
	arma::Mat<double> A = { {4.0, 1.0}, {1.0, 3.0} };
	x = coordinate_given - x;

	return -A*x;
}
*/

#endif // DENG_OPTIMIZATION_HPP
