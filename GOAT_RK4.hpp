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
            //Deng::Col_vector<Field> parameters;
            arma::Col<Field> parameters;
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
            virtual Deng::Col_vector<arma::Mat<Field> > dynamics(double t) const = 0;
            //empty virtual destructor
            virtual ~Hamiltonian() {};
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

            //we use the keyword const here to prevent changing H
            virtual void Evolve_one_step(const Deng::GOAT::Hamiltonian<Field> &H, int t_index);
            virtual void Evolve_to_final(const Deng::GOAT::Hamiltonian<Field> &H);
            //initialize initial states and identity
            virtual void Prep_for_H(const Deng::GOAT::Hamiltonian<Field> &H);
        };


        /*
        class Conj_Grad_Min
        {
        protected:
            unsigned int _dim_para;
            const double _epsilon;//precision of parametric space
            const unsigned int _max_iteration;//maximum allowed trials when searching for minimum
        public:
            //coordinate space
            arma::Col<double> coordinate;
            //search direction
            arma::Col<double> search_direction;
            //constructor to
            Conj_Grad_Min(unsigned int dim_para, double epsilon = 1E-3, unsigned int max_iteration = 100) : _dim_para(dim_para), _epsilon(epsilon), _max_iteration(max_iteration) {};



            //calculate the target function.
            //usually called during optimization in 1D
            virtual double Target_function(const arma::Col<double> & give_coordinate) = 0;
            //return the value of target function, also give the gradient information to the reference
            //usually called when choosing the conjugate gradient
            virtual double Target_function_and_gradient(const arma::Col<double>& give_coordinate, arma::Col<double> &gradient) = 0;
            //optimization in 1D
            virtual double OneD_Minimum(const arma::Col<double> give_coordinate, const arma::Col<double> give_search_direction);
            //virtual double Golden_Section(const arma::Col<double> give_coordinate, const arma::Col<double> give_search_direction);

        };
        */
    }
}



#endif // DENG_GOAT_MATRIX_RK4_HPP
