#ifndef DENG_GOAT_RK4_HPP
#define DENG_GOAT_RK4_HPP

#include <iostream>
#include <complex>
#include <armadillo>

#include"Deng_vector.hpp"

namespace Deng
{
    namespace GOAT
    {
        template <typename Field>
        class RK4;

        //the abstract base class for (almost) all Hamiltonians using GOAT
        template <typename Field>
        class Hamiltonian//defined the most basic Hamiltonian, with only dimension, time parameters and the parameter space dimension
        {
        protected:
            //for the derived class to use
            int _dim_para;//dimension of parametric space
            double _dt;
            double _tau;//total time
            int _N_t;//# of time steps
            int _N;//dimension of state space
        public:
            //control parameters
            //have to use double as the parameters are usually real
            mutable arma::Col<double> parameters;
            void Update_parameters(arma::Col<double> new_para)
            {
                parameters = new_para;
            };
            //input necessary parameters
            Hamiltonian(int N, int N_t, double tau, int dim_para = 0);
            //return the protected data
            //int dimension_of_parametric_space() const {return _dim_para;};
            //int total_steps_in_time() const {return _N_t;};
            //int dimension_of_state_space() const {return _N;};
            //double total_time() const {return _dt;};
            //calculate the derivative of U in a block vector form
            virtual Deng::Col_vector<arma::Mat<Field> > Derivative (Deng::Col_vector<arma::Mat<Field> > position, int t_index, bool half_time) const;
            //invoked by member function Derivative; return H and partial_H in a block vector form
            //pure-virtual function, must be implemented in instances!!!!
            virtual Deng::Col_vector<arma::Mat<Field> > Dynamics(double t) const = 0;
            //empty virtual destructor
            virtual ~Hamiltonian() = default;
            //for RK4 and conjugate gradient to use the parameters
            friend class RK4<Field>;
            friend class Conjugate_Gradient_Method;
        };

        template <typename Field>
        class RK4//Runge-Kutta 4th order, applied to GOAT method only, as the time evolution of U is sparse under GOAT
        {
        private:
            //arma::Mat<Field> _eyedentity;
        public:
            //state to be evolved
            Deng::Col_vector<arma::Mat<Field> > current_state;
            Deng::Col_vector<arma::Mat<Field> > next_state;
            //Deng::Col_vector<arma::Mat<Field> > initial_state;
            //Deng::Col_vector<arma::Mat<Field> > final_state;

            //constructor is not necessary here, as RK4 could be defined as friend of Hamiltonian
            //therefore could directly use the private variable of Hamiltonian
            //RK4(int dim_para, double tau, int N_t);


            //initialize initial states and identity
            virtual void Prep_for_H(const Deng::GOAT::Hamiltonian<Field> &H);
            //we use the keyword const here to prevent changing H
            virtual void Evolve_one_step(const Deng::GOAT::Hamiltonian<Field> &H, const int t_index);
            virtual void Evolve_to_final(const Deng::GOAT::Hamiltonian<Field> &H);

        };

    }
}



#endif // DENG_GOAT_MATRIX_RK4_HPP
