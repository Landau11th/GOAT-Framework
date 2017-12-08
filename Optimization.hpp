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
			virtual arma::Col<real> Conj_Grad_Search(arma::Col<real> start_coordinate) const;

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
		template<typename real>
        real OneD_Golden_Search_Recur(const arma::Col<real> start_coordinate, const arma::Col<real> search_direction_given,
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
			arma::Col<real> Newton_1st_order(arma::Col<real> start_coordinate) const;
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
			real _randomness;
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
				identity.eye(this->_dim_para, this->_dim_para);
			};

			void Set_Randomness(real rand) { _randomness = rand; };

			arma::Col<real> BFGS(arma::Col<real> start_coordinate) const;

			virtual ~Quasi_Newton() = default;
		};


		template<typename real>
		class Annealing : public Min_Know_Function<real>
		{
		protected:
			//did not use multimap since it's unlikely to have two values that are exactly the sme
			mutable std::map<real, arma::Col<real>> _min_func_values;
			unsigned int _num_mins;
			real _randomness;
			real _T_start;
		public:
			Annealing(unsigned int dim_para, real epsilon, unsigned int max_iteration, real T_start, unsigned int num_mins, real rand)
				: Min_Know_Function(dim_para, epsilon, max_iteration),_T_start(T_start), _num_mins(num_mins), _randomness(rand)
			{
				Init_Mins();
				//initialize the random seed
				arma::arma_rng::set_seed(time(nullptr));
			}

			//clear the map, and fill it with some values bigger than 0
			void Init_Mins()
			{
				_min_func_values.clear();

				arma::Col<real> zerozero(this->_dim_para, arma::fill::zeros);

				for (unsigned int i = 0; i < _num_mins; ++i)
				{
					_min_func_values.emplace(i, zerozero);
				}
			}

			arma::Col<real> New_Coord(const arma::Col<real> & coord) const
			{
				auto coord_new = coord;

				coord_new.randu();
				coord_new = 2 * coord_new - 1.0;
				coord_new = _randomness*coord_new;

				return coord_new + coord;
			};

			std::map<real, arma::Col<real>> Mins_from_Search() const { return _min_func_values; };

			void Search_for_Min(const arma::Col<real> start_coord) const
			{
				real f_value = 0;
				real f_value_temp;
				arma::Col<real> rand_shift(this->_dim_para);
				arma::Col<real> coord = start_coord;
				arma::Col<real> coord_temp = start_coord;
				arma::Mat<real> scalar_rand(1, 1);

				real T;
				
				for (unsigned int i = 0; i < this->_max_iteration; ++i)
				{
					coord_temp = New_Coord(coord);
					
					f_value_temp = this->f->function_value(coord_temp);
					
					std::cout << f_value_temp << std::endl;

					//add new pair to map
					_min_func_values.emplace(f_value_temp, coord_temp);
					//erase the pair with the largest function value
					_min_func_values.erase(--_min_func_values.end());

					//temperature decrease linearly
					T = _T_start * (1.0 - (1.0*i)/ this->_max_iteration);
					////or exponentially
					//T = T_max * std::exp(-100.0*i) / this->_max_iteration);

					if (arma::as_scalar(scalar_rand.randu()) < std::exp(-(f_value - f_value_temp) / T))
					{
						coord = coord_temp;
						f_value = f_value_temp;
					}
				}

				for (auto iter = _min_func_values.begin(); iter != _min_func_values.end(); ++iter)
				{
					std::cout << (*iter).first << "\n";
					std::cout << (*iter).second.t() << "\n";
				}
			};

			//output all read arguments to output_strm
			void Output_to_file(std::ostream & output_strm) const
			{
				//std::cout << __func__ << std::endl;
				
				for (auto iter = _min_func_values.begin(); iter != _min_func_values.end(); ++iter)
				{
					output_strm << (*iter).first << "\n";
					//output_strm << (*iter).second.t() << "\n";
					(*iter).second.t().raw_print(output_strm);
				}
				output_strm << std::endl;
			}
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
