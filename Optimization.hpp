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


		//find minimum with conjugate gradient method
		template<typename real>
		class Newton_1st_order : public Min_Know_Gradient<real>
		{
		protected:
			real target_value;
		public:
			//previous search directions
			//it is needed in conj grad for a better search direction
			mutable arma::Col<real> previous_search_direction;
			//constructor
			//using Min_Know_Gradient<real>::Min_Know_Gradient;
			//value of _epsilon and _epsilon_gradient implicitly assume the scale of the space
			Newton_1st_order(unsigned int dim_para, real epsilon, unsigned int max_iteration, real epsilon_gradient)
				: Min_Know_Gradient<real>(dim_para, epsilon, max_iteration, epsilon_gradient)
			{
				previous_search_direction = arma::zeros<arma::Col<real> >(dim_para);
			};

			virtual void Assign_Target_Function_Newton(Target_function<real> *pt_f, real target)
			{
				f = pt_f;
				target_value = target;
			}
			//if not giving a start coordinate, we start from 0
			virtual arma::Col<real> Newton(arma::Col<real> start_coordinate = arma::zeros<arma::Col<real> >(dim_para, arma::fill::zeros)) const
			{
				std::cout << "Start search for " << target_value << "\n";
				arma::Col<real> coordinate = start_coordinate;
				real lambda = 0.0;
				real function_value = 0;
				arma::Col<real> search_direction = coordinate;
				search_direction.zeros();
				real gradient_norm = 1.0;
				
				bool if_start_random_search = false;

				do 
				{
					unsigned int i = 0;
					if_start_random_search = false;
					
					do 
					{
						lambda = (function_value - target_value) / (gradient_norm*gradient_norm);
						coordinate += lambda*search_direction;

						search_direction = f->negative_gradient(coordinate, function_value);
						gradient_norm = sqrt(arma::as_scalar(search_direction.t()*search_direction));
						
						std::cout << i++ << " th search reach " << function_value << std::endl;
						std::cout << coordinate.t() << std::endl;
						if (i > _max_iteration)
						{
							if_start_random_search = true;
							coordinate.randn();
							std::cout << "Newton method hit max allowed iterations, start over" << std::endl;
							break;
						}
					
					} while (abs(function_value - target_value) > _epsilon && gradient_norm > _epsilon_gradient);

				} while (if_start_random_search);

				return coordinate;
			}

			real Value_of_target_value() const { return target_value; }

			virtual ~Newton_1st_order() = default;
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
		)
		{
			const real step_size = 1.0 / sqrt(as_scalar(search_direction_given.t()*search_direction_given));
			real current_size = 10*step_size;
			real lambda = 0;
			
			real f_current = f->function_value(start_coordinate);
			real f_next = f_current + 1.0;

			unsigned int count_iter = 0;
			while (f_current <= f_next)
			{
				lambda = current_size;
				f_next = f->function_value(start_coordinate + (lambda)*search_direction_given);
				current_size = current_size*0.5;

				if (count_iter >= max_iteration)
				{
					std::cout << "My_1D_search_method hit max iteration limit in the first part!" << std::endl;
					break;
				}
			}

			
			real f_forward = f_current + 100* epsilon;
			real f_backward = f_current + 100 * epsilon;
			count_iter = 0;
			while ((abs(f_forward - f_current) + abs(f_backward - f_current)) >= epsilon)
			{
				f_forward = f->function_value(start_coordinate + (lambda + current_size)*search_direction_given);
				f_backward = f->function_value(start_coordinate + (lambda - current_size)*search_direction_given);

				if (f_forward < f_current)
				{
					f_current = f_forward;
					lambda += current_size;
				}
				else if (f_backward < f_current)
				{
					f_current = f_backward;
					lambda -= current_size;
				}

				current_size *= 0.5;
				count_iter++;

				if (count_iter >= max_iteration)
				{
					std::cout << "My_1D_search_method hit max iteration limit in the second part!" << std::endl;
					break;
				}
			}
			
			return lambda;
		}


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
