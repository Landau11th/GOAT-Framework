#include <iostream>
#include <complex>
#include <armadillo>
//#include <typeinfo>
#include "GOAT_RK4.hpp"

using namespace Deng::GOAT;

template class Deng::GOAT::Hamiltonian<float>;
template class Deng::GOAT::Hamiltonian<double>;
template class Deng::GOAT::Hamiltonian<std::complex<float> >;
template class Deng::GOAT::Hamiltonian<std::complex<double> >;

template <typename Field>
Hamiltonian<Field>::Hamiltonian(int N, int N_t, double tau, int dim_para)
{
    _dim_para = dim_para;
    _tau = tau;
    _N_t = N_t;
    _dt = tau/N_t;
    _N = N;

    parameters.set_size(_dim_para);
    for(int i = 0; i < _dim_para; ++i)
    {
        parameters[i] = 0;
    }
}
template <typename Field>
Deng::Col_vector<arma::Mat<Field> > Hamiltonian<Field>::Derivative(Deng::Col_vector<arma::Mat<Field> > U, int t_index, bool half_time) const
{
    double shift;
    if(half_time)
        shift = 0.5;
    else
        shift = 0;//0 or 1???

    double t = t_index*_dt + shift*_dt;

    Deng::Col_vector<arma::Mat<Field> > k(_dim_para + 1);

    Deng::Col_vector<arma::Mat<Field> > H_and_partial_H(_dim_para + 1);
    H_and_partial_H = Dynamics(t);
    //std::cout << "here?" << std::endl;
    k[0] = H_and_partial_H[0]*U[0];

    for(int i = 1; i <= _dim_para; ++i)
    {
        k[i] = H_and_partial_H[i]*U[0] + H_and_partial_H[0]*U[i];
    }

    return k;
}






template class Deng::GOAT::RK4<float>;
template class Deng::GOAT::RK4<double>;
template class Deng::GOAT::RK4<std::complex<float> >;
template class Deng::GOAT::RK4<std::complex<double> >;

/*
template <typename Field>
RK4<Field>::RK4(int dim_para, double tau, int N_t)
{
    _dim_para = dim_para;
    _tau = tau;
    _N_t = N_t;
    _dt = _tau/_N_t;
    _current_state.Constructor(_dim_para);
    _next_state.Constructor(_dim_para);
    initial_state.Constructor(_dim_para);
    final_state.Constructor(_dim_para);
}
*/
template <typename Field>
void RK4<Field>::Evolve_one_step(const Deng::GOAT::Hamiltonian<Field> &H, const int t_index)
{
    //Deng::Col_vector<arma::Mat<Field> > temp_state(H._dim_para + 1);
    //std::cout << "here0" << std::endl;
    Deng::Col_vector<arma::Mat<Field> > k1 = H.Derivative(current_state               , t_index    , false);

    //temp_state = current_state + 0.5*H._dt*k1;
    Deng::Col_vector<arma::Mat<Field> > k2 = H.Derivative(current_state + 0.5*H._dt*k1, t_index    , true );

    //temp_state = current_state + 0.5*H._dt*k2;
    Deng::Col_vector<arma::Mat<Field> > k3 = H.Derivative(current_state + 0.5*H._dt*k2, t_index    , true );

    //temp_state = current_state +     H._dt*k3;
    Deng::Col_vector<arma::Mat<Field> > k4 = H.Derivative(current_state +     H._dt*k3, t_index + 1, false);

    next_state = current_state + (H._dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);

    /*
    //temp_state = current_state + k1*(H._dt*0.5);
    //Deng::Col_vector<arma::Mat<Field> > k2 = H.Derivative(temp_state    , t_index  , true);

    //temp_state = current_state + k2*(H._dt*0.5);
    //Deng::Col_vector<arma::Mat<Field> > k3 = H.Derivative(temp_state    , t_index  , true );

    //temp_state = current_state + k3*(H._dt    );
    //Deng::Col_vector<arma::Mat<Field> > k4 = H.Derivative(temp_state    , t_index+1, false);

    //next_state = current_state + (k1 + k2*2.0 + k3*2.0 + k4)*(H._dt/6.0);
    */

    //std::cout << "at" << t_index << std::endl;

}
template <typename Field>
void RK4<Field>::Evolve_to_final(const Deng::GOAT::Hamiltonian<Field> &H)
{
    for(int i = 0; i < H._N_t; ++i)
    {
        Evolve_one_step(H, i);
        current_state = next_state;
    }
}
template <typename Field>
void RK4<Field>::Prep_for_H(const Deng::GOAT::Hamiltonian<Field> &H)
{
    //make sure the dimension fits
    current_state.set_size(H._dim_para + 1);
    next_state.set_size(H._dim_para + 1);
    current_state[0].eye(H._N, H._N);
    next_state[0].eye(H._N, H._N);
    for(int i = 1; i <= H._dim_para; ++i)
    {
        current_state[i].zeros(H._N, H._N);
        next_state[i].zeros(H._N, H._N);
    }
    //_eyedentity.eye(H._N, H._N);
}



