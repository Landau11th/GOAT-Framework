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
			virtual real function_value(const arma::Col<real>& coordinate_given) const = 0;
			virtual arma::Col<real> negative_gradient(const arma::Col<real>& coordinate_given, real &function_value) const = 0;
			virtual arma::Mat<real> Hessian(const arma::Col<real>& coordinate_given, real &function_value, arma::Col<real> &negative_gradient) const = 0;
			virtual void higher_order(const arma::Col<real>& coordinate_given, const real order) const = 0;
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

			void Revise_epsilon(real epsilon) { _epsilon = epsilon; }
			real Value_of_epsilon() const { return _epsilon; }

			virtual ~Min_Know_Function() = default;
		};

		//this class is to optimize a function when we know the gradient
		template<typename real>
		class Min_Know_Gradient : public Min_Know_Function<real>
		{
		protected:
			real _epsilon_gradient;
		public:
			//search directions
			mutable arma::Col<real> search_direction;
			//may need a variable to record current minimum
			//real current_min = nan;

			//using Min_Know_Function<real>::Min_Know_Function;
			//omit the default constructor.
			//Must know the parameters before declare this object
			Min_Know_Gradient(unsigned int dim_para, real epsilon, unsigned int max_iteration, real epsilon_gradient) 
				: Min_Know_Function<real>(dim_para, epsilon, max_iteration), _epsilon_gradient(epsilon_gradient)
			{
				search_direction = arma::zeros<arma::Col<real> >(dim_para);
			};
			virtual ~Min_Know_Gradient() = default;
		};


		template<typename real>
		class Min_Know_Hessian : public Min_Know_Function<real>
		{
		protected:
			real _epsilon_gradient;
		public:
			//may need a variable to record current minimum
			//real current_min = nan;

			//using Min_Know_Function<real>::Min_Know_Function;
			//omit the default constructor.
			//Must know the parameters before declare this object
			Min_Know_Hessian(unsigned int dim_para, real epsilon, unsigned int max_iteration, real epsilon_gradient)
				: Min_Know_Function<real>(dim_para, epsilon, max_iteration), _epsilon_gradient(epsilon_gradient)
			{
			};
			virtual ~Min_Know_Hessian() = default;
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
			//value of _epsilon and _epsilon_gradient implicitly assume the scale of the space
			Min_Conj_Grad(unsigned int dim_para, real epsilon, unsigned int max_iteration, real epsilon_gradient) 
				: Min_Know_Gradient<real>(dim_para, epsilon, max_iteration, epsilon_gradient)
			{
				previous_search_direction = arma::zeros<arma::Col<real> >(dim_para);
			};

			//Here I implicitly assume the format of the 1D optimization function!!!!!!
			real(*Opt_1D)(const arma::Col<real> start_coordinate, const arma::Col<real> search_direction_given, const Target_function<real>* const f, const unsigned int max_iteration, const real epsilon);

			//if not giving a start coordinate, we start from 0
			virtual arma::Col<real> Conj_Grad_Search(arma::Col<real> start_coordinate = arma::zeros<arma::Col<real> >(dim_para)) const;

			virtual ~Min_Conj_Grad() = default;
		};


		//a frequently used oned search
		template<typename real>
		real OneD_Golden_Search(const arma::Col<real> start_coordinate, const arma::Col<real> search_direction_given,
			const Target_function<real>* const f, const unsigned int max_iteration, const real epsilon
		);
		//template<typename real>
		//real Search_Appropriate_Shape(const arma::Col<real> start_coordinate, const arma::Col<real> search_direction_given,
		//	const Target_function<real>* const f, const unsigned int max_iteration, const real epsilon,
		//	real &middle, real &right, real &f_middle, real &f_right
		//);
		template<typename real>
		real My_1D_foward_method(const arma::Col<real> start_coordinate, const arma::Col<real> search_direction_given,
			const Target_function<real>* const f, const unsigned int max_iteration, const real epsilon
		);


		//find the root of a function
		//does not perform well in my tests
		template<typename real>
		class Newton_Find_Root : public Min_Know_Gradient<real>
		{
		protected:
			real target_value;
		public:
			//previous search directions
			//it is needed in conj grad for a better search direction
			mutable arma::Col<real> previous_search_direction;
			//constructor
			using Min_Know_Gradient<real>::Min_Know_Gradient;
			//find root with first order derivative
			arma::Col<real> Newton_Find_Root<real>::Newton_1st_order(arma::Col<real> start_coordinate = arma::zeros<arma::Col<real> >(dim_para, arma::fill::zeros)) const;
			virtual void Assign_Target_Function_Value(real target)
			{
				target_value = target;
			}
			real Value_of_target_value() const { return target_value; }
			virtual ~Newton_Find_Root() = default;
		};


		//find minimum with Newton method
		template<typename real>
		class Newton_Find_Min : public Min_Know_Hessian<real>
		{
		protected:
			real target_value;
		public:
			//constructor
			using Min_Know_Hessian<real>::Min_Know_Hessian;
			arma::Col<real> Newton_2nd_order(arma::Col<real> start_coordinate) const;
			real Value_of_target_value() const { return target_value; }
			virtual ~Newton_Find_Min() = default;
		};


		//find minimum with Newton method
		template<typename real>
		class Quasi_Newton : public Min_Know_Gradient<real>
		{
		protected:
			mutable arma::Mat<real> H_k;
			mutable arma::Mat<real> H_kp1;
			mutable arma::Mat<real> x_k;
			mutable arma::Mat<real> x_kp1;
			mutable arma::Mat<real> y_k;
			mutable arma::Mat<real> delta_x_k;
			mutable arma::Mat<real> neg_grad_k;
			mutable arma::Mat<real> neg_grad_kp1;
			arma::Mat<real> identity;
		public:
			//constructor
			using Min_Know_Gradient<real>::Min_Know_Gradient;

			Quasi_Newton(unsigned int dim_para, real epsilon, unsigned int max_iteration, real epsilon_gradient)
				: Min_Know_Gradient<real>(dim_para, epsilon, max_iteration, epsilon_gradient)
			{
				identity.eye(_dim_para, _dim_para);
			};

			arma::Col<real> BFGS(arma::Col<real> start_coordinate) const
			{
				assert(start_coordinate.size() == _dim_para && "coordinate dimention mismatch in BFGS!");
				x_k = start_coordinate;
				real f_value = 0;
				neg_grad_k = f->negative_gradient(x_k, f_value);
				H_k = identity;

				real temp = 0.0;

				real gradient_norm = 0.0;
				real coordinate_norm = 0.0;

				real alpha_k = 0.125;

				do
				{
					delta_x_k = alpha_k*(H_k*neg_grad_k);
					x_kp1 = x_k + delta_x_k;
					neg_grad_kp1 = f->negative_gradient(x_kp1, f_value);
					y_k = neg_grad_k - neg_grad_kp1;

					temp = arma::as_scalar(y_k.t()*delta_x_k);
					H_kp1 = (identity - delta_x_k*y_k.t() / temp)*H_k*(identity - y_k*delta_x_k.t() / temp) + delta_x_k*delta_x_k.t() / temp;
					
					gradient_norm = sqrt(arma::as_scalar(neg_grad_kp1.t()*neg_grad_kp1));
					coordinate_norm = sqrt(arma::as_scalar(x_kp1.t()*x_kp1));

					if (coordinate_norm > 1.0*_dim_para && gradient_norm > _epsilon_gradient)
					{
						std::cout << "too far away from origin, start with random position near origin\n";
						x_k.randu();
						x_k = sqrt(_dim_para)*2.0*(x_k - 0.5);
						neg_grad_k = f->negative_gradient(x_k, f_value);
						H_k = identity;
						continue;
					}
					else if (f_value != f_value)
					{
						std::cerr << "NaN in Runge Kutta. Might be too far away from origin, or dt is not small enough\n";
						std::cout << "NaN in Runge Kutta. Might be too far away from origin, or dt is not small enough\n";
						x_k.randu();
						x_k = sqrt(_dim_para)*2.0*(x_k - 0.5);
						neg_grad_k = f->negative_gradient(x_k, f_value);
						H_k = identity;
						gradient_norm = 10.0*_epsilon_gradient;
						continue;
					}

					std::cout << "new function value: " << f_value << " with gradient norm " << gradient_norm << std::endl;
					std::cout << x_kp1.t() << std::endl;


					H_k = H_kp1;
					neg_grad_k = neg_grad_kp1;
					x_k = x_kp1;
				} while (gradient_norm > _epsilon_gradient);

				return x_kp1;
			}

			virtual ~Quasi_Newton() = default;
		};

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
