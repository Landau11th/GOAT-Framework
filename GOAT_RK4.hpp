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

		//template<typename Field, typename Parameter>
		//class GOAT_Target_1st_order;
		
		//the abstract base class for (almost) all Hamiltonians using GOAT
        template <typename Field, typename Parameter>
        class Hamiltonian//defined the most basic Hamiltonian, with only dimension, time parameters and the parameter space dimension
        {
        protected:
            //for the derived class to use
            const unsigned int _dim_para;//dimension of parametric space
            const Parameter _dt;
            const Parameter _tau;//total time
            const unsigned int _N_t;//# of time steps
            const unsigned int _N;//dimension of state space
			//control parameters
			//have to use double as the parameters are usually real
			mutable arma::Col<Parameter> parameters;
        public:
            virtual void Update_parameters(arma::Col<Parameter> new_para) const
            {
                parameters = new_para;
            };
            //input necessary parameters
            Hamiltonian(unsigned int N, unsigned int N_t, Parameter tau, unsigned int dim_para = 0);
            
			//calculate the derivative of U and partial U in a block vector form
            virtual Deng::Col_vector<arma::Mat<Field>> Derivative (Deng::Col_vector<arma::Mat<Field> > position, unsigned int t_index, bool half_time) const;
            //invoked by member function Derivative; return H and partial_H in a block vector form
            //pure-virtual function, must be implemented in instances!!!!
            virtual Deng::Col_vector<arma::Mat<Field>> Dynamics(Parameter t) const = 0;

			//calculate the derivative of U only
			virtual arma::Mat<Field> Derivative_U(arma::Mat<Field> position, unsigned int t_index, bool half_time) const;
			//invoked by member function Derivative; return H only
			//pure-virtual function, must be implemented in instances!!!!
			virtual arma::Mat<Field> Dynamics_U(Parameter t) const = 0;
            
			
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
            //state to be evolved
            Deng::Col_vector<arma::Mat<Field>> current_state;
            Deng::Col_vector<arma::Mat<Field>> next_state;

            //initialize initial states and identity
            virtual void Prep_for_H(const Deng::GOAT::Hamiltonian<Field, Parameter> &H);
			//time evolution for U and partial U
            virtual void Evolve_one_step(const Deng::GOAT::Hamiltonian<Field, Parameter> &H, const unsigned int t_index);
            virtual void Evolve_to_final(const Deng::GOAT::Hamiltonian<Field, Parameter> &H);

			//evolve U only
			virtual void Prep_for_H_U(const Deng::GOAT::Hamiltonian<Field, Parameter> &H);
			virtual void Evolve_one_step_U(const Deng::GOAT::Hamiltonian<Field, Parameter> &H, const unsigned int t_index);
			virtual void Evolve_to_final_U(const Deng::GOAT::Hamiltonian<Field, Parameter> &H);

        };



		template<typename Field, typename Parameter>
		class GOAT_Target_1st_order : public Deng::Optimization::Target_function<Parameter>
		{
		public:
			virtual Parameter function_value(const arma::Col<Parameter>& coordinate_given) override;
			virtual arma::Col<Parameter> negative_gradient(const arma::Col<Parameter>& coordinate_given, Parameter &function_value) override;
			virtual arma::Mat<Parameter> Hessian(const arma::Col<Parameter>& coordinate_given, Parameter &function_value, arma::Col<Parameter> &negative_gradient)
			{
				assert(false && "Hessian is not known for class GOAT_Target_1st_order!\n");
				return 0;
			};
			virtual void higher_order(const arma::Col<Parameter>& coordinate_given, const Parameter order)
			{
				assert(false && "Higher order derivatives are not known for class GOAT_Target_1st_order!\n");
			};
			Deng::GOAT::RK4<Field, Parameter> *RK_pt;
			//Deng::GOAT::Hamiltonian<std::complex<real> > *H_only_pt;
			Deng::GOAT::Hamiltonian<Field, Parameter> *H_and_partial_H_pt;

			//GOAT_Target(Deng::GOAT::RK4<std::complex<real> > *input_RK_pt, Deng::GOAT::Hamiltonian<std::complex<real> > *input_H_only_pt, Deng::GOAT::Hamiltonian<std::complex<real> > *input_H_and_partial_H_pt)
			GOAT_Target_1st_order(Deng::GOAT::RK4<Field, Parameter> *input_RK_pt, Deng::GOAT::Hamiltonian<Field, Parameter> *input_H_and_partial_H_pt)
			{
				RK_pt = input_RK_pt;
				H_and_partial_H_pt = input_H_and_partial_H_pt;
			}
			virtual void Set_Controlled_Unitary_Matrix(const arma::Mat<Field> &matrix_desired)
			{
				unitary_goal = matrix_desired;
			};
			arma::Mat<Field> unitary_goal;

		};
		
    }
}



#endif // DENG_GOAT_MATRIX_RK4_HPP
