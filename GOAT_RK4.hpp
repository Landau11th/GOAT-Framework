#ifndef DENG_GOAT_RK4_HPP
#define DENG_GOAT_RK4_HPP

#include <iostream>
#include <complex>
#include <armadillo>

#include"Deng_vector.hpp"
#include"Optimization.hpp"

namespace Deng
{
    namespace GOAT
    {
        template<typename Field, typename Parameter>
		class RK4;



		//the abstract base class for (almost) all Hamiltonians using GOAT and RK
        template <typename Field, typename Parameter>
        class Hamiltonian
		//defined the most basic Hamiltonian, with only dimension, time parameters and the parameter space dimension
        {
        protected:
            //for the derived class to use
            const unsigned int _dim_para;//dimension of parametric space
            const Parameter _dt;
            const Parameter _tau;//total time
            const unsigned int _N_t;//number of time steps
            const unsigned int _N;//dimension of state space
			//control parameters
			//have to use double as the parameters are usually real
			mutable arma::Col<Parameter> parameters;
        public:
			//interface to change the parameter from outside
            void Update_parameters(arma::Col<Parameter> new_para)
            {
				//dimension check
				if (new_para.size() == _dim_para)
				{
					parameters = new_para;
				}
				else
				{
					std::cerr << "wrong dimension for parameter input in " << __func__ << "\n";
				}
            };
            
			//constructor
			//must input necessary parameters
            Hamiltonian(unsigned int N, unsigned int N_t, Parameter tau, unsigned int dim_para = 0);

			//calculate the derivative of U and partial U in a block vector form
            //virtual 
			Deng::Col_vector<arma::Mat<Field>> Derivative (const Deng::Col_vector<arma::Mat<Field>>& position, const unsigned int t_index, const bool half_time) const;
            
			//called by member function Derivative; return H and partial_H in a block vector form
			//return the derivative of U and partial U partial aplha, scaled by imaginary unit i and hbar
			//where alpha stands for the parameters
			//the implementation depends on specific modeling
            //pure-virtual function, must be implemented in instances!!!!
            virtual Deng::Col_vector<arma::Mat<Field>> Dynamics(const Parameter t) const = 0;

			//calculate the derivative of U only
			//virtual 
			arma::Mat<Field> Derivative_U(const arma::Mat<Field>& position, const unsigned int t_index, const bool half_time) const;

			//invoked by member function Derivative; return iH only (scaled by hbar)
			//the implementation depends on specific modeling
			//pure-virtual function, must be implemented in instances!!!!
			virtual arma::Mat<Field> Dynamics_U(const Parameter t) const = 0;

			//empty virtual destructor
            virtual ~Hamiltonian() = default;

            //for RK4 and conjugate gradient to use the parameters
            friend class RK4<Field, Parameter>;
            //friend class GOAT_Target_1st_order<Field, Parameter>;
        };



		//RK4 specialized for GOAT
        template <typename Field, typename Parameter>
        class RK4//Runge-Kutta 4th order, applied to GOAT method only
				 //as the time evolution of U is sparse and has fiexd format under GOAT
        {
        public:
            //state (more precisely, unitary matrix and its partial derivatives) to be evolved
            Deng::Col_vector<arma::Mat<Field>> current_state;
            Deng::Col_vector<arma::Mat<Field>> next_state;

            //initialize initial states and identity
			//to fit the dimension of H
            //virtual 
			void Prep_for_H(const Deng::GOAT::Hamiltonian<Field, Parameter> &H);

			//time evolution for U and partial U
            //virtual 
			void Evolve_one_step(const Deng::GOAT::Hamiltonian<Field, Parameter> &H, const unsigned int t_index);
            //virtual 
			void Evolve_to_final(const Deng::GOAT::Hamiltonian<Field, Parameter> &H);

			//evolve U only
			//virtual 
			void Prep_for_H_U(const Deng::GOAT::Hamiltonian<Field, Parameter> &H);
			//virtual 
			void Evolve_one_step_U(const Deng::GOAT::Hamiltonian<Field, Parameter> &H, const unsigned int t_index);
			//virtual 
			void Evolve_to_final_U(const Deng::GOAT::Hamiltonian<Field, Parameter> &H);

			////one may change all the above member function to virtual
			//for new revised method inheriting from RK
			//together with the following virtual destructor
			//virtual ~RK4() = default;
        };



		//given the target U and U under certain parameters
		//calculate the target function by taking the trace, which will be minimized later
		//inherit the wrap Deng::Optimization::Target_function
		template<typename Field, typename Parameter>
		class GOAT_Target_1st_order : public Deng::Optimization::Target_function<Parameter>
		{
		protected:
			//target unitary matrix
			arma::Mat<Field> unitary_goal;
			//give the Hamiltonian and corresponding time-evolution method through a dynamic binding
			Deng::GOAT::RK4<Field, Parameter> *RK_pt;
			Deng::GOAT::Hamiltonian<Field, Parameter> *H_and_partial_H_pt;
		public:
			//give the parameter (coordinate_given/phase space coordinate)
			//calculate the target function value
			virtual Parameter function_value(const arma::Col<Parameter>& coordinate_given) const override;
			//claculate the negative gradient, together with function value as reference
			virtual arma::Col<Parameter> negative_gradient(const arma::Col<Parameter>& coordinate_given, Parameter &function_value) const override;
			
			///////////////////////////////////////////////////////////////
			//Hessians and higher orders should not be called
			virtual arma::Mat<Parameter> Hessian(const arma::Col<Parameter>& coordinate_given, Parameter &function_value, arma::Col<Parameter> &negative_gradient) const override
			{
				assert(false && "Hessian is not known for class GOAT_Target_1st_order!\n");
				return 0;
			};
			virtual void higher_order(const arma::Col<Parameter>& coordinate_given, const Parameter order) const override
			{
				assert(false && "Higher order derivatives are not known for class GOAT_Target_1st_order!\n");
			};
			/////////////////////////////////////////////////////////////////

			//constructor
			GOAT_Target_1st_order(Deng::GOAT::RK4<Field, Parameter> *input_RK_pt, Deng::GOAT::Hamiltonian<Field, Parameter> *input_H_and_partial_H_pt)
			{
				RK_pt = input_RK_pt;
				H_and_partial_H_pt = input_H_and_partial_H_pt;
			}
			//interface for setting target unitary matrix
			virtual void Set_Controlled_Unitary_Matrix(const arma::Mat<Field> &matrix_desired)
			{
				unitary_goal = matrix_desired;
			};
			//release the pointer
			virtual ~GOAT_Target_1st_order()
			{
				RK_pt = nullptr;
				H_and_partial_H_pt = nullptr;
			};
		};



		//given the target U and U under certain parameters
		//calculate the target function but neglect the phase information
		//i.e. focus only the transition probability
		template<typename Field, typename Parameter>
		class GOAT_Target_1st_order_no_phase : public Deng::GOAT::GOAT_Target_1st_order<Field, Parameter>
		{
		protected:
			//to know the transition probability one also need to know the initial states
			arma::Mat<Field> initial_states;
		public:
			using GOAT_Target_1st_order<Field, Parameter>::GOAT_Target_1st_order;
			//override virtual function
			//roughly like calculating the transition probabitlity and sum over
			virtual Parameter function_value(const arma::Col<Parameter>& coordinate_given) const override;
			//new negative gradient
			virtual arma::Col<Parameter> negative_gradient(const arma::Col<Parameter>& coordinate_given, Parameter &function_value) const override;

			virtual void Set_Initial_States(const arma::Mat<Field> &initial_states_desired)
			{
				initial_states = initial_states_desired;
			};
		};



		//calculate Hessian approximately using finite difference
		//slow and not accurate. Avoid using it
		template<typename Field, typename Parameter>
		class GOAT_Target_2nd_order_no_phase : public Deng::GOAT::GOAT_Target_1st_order_no_phase<Field, Parameter>
		{
		public:
			using GOAT_Target_1st_order_no_phase<Field, Parameter>::GOAT_Target_1st_order_no_phase;
			//calculated Hessian approximately
			virtual arma::Mat<Parameter> Hessian(const arma::Col<Parameter>& coordinate_given, Parameter &function_value, arma::Col<Parameter> &negative_gradient) const override
			{
				negative_gradient = this->negative_gradient(coordinate_given, function_value);

				arma::Col<Parameter> finite_diff = coordinate_given;
				const unsigned int dim_coordinate = coordinate_given.size();
				arma::Mat<Parameter> hess(dim_coordinate, dim_coordinate, arma::fill::zeros);

				for (unsigned int i = 0; i < dim_coordinate; ++i)
				{
					finite_diff.zeros();
					finite_diff(i) = 0.0078125;

					hess.col(i) = this->negative_gradient(coordinate_given + finite_diff, function_value) - negative_gradient;
				}
				hess = -64.0*(hess+hess.t());
				//hess = -128.0*hess;
				return hess;
			};
		};

    }
}

#endif // DENG_GOAT_MATRIX_RK4_HPP
