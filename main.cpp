#include<iostream>
#include<fstream>
#include<string>
#include<ctime>
#include<complex>

#define ARMA_DONT_USE_WRAPPER
//#define ARMA_USE_WRAPPER
//#define ARMA_BLAS_CAPITALS
//#define ARMA_BLAS_UNDERSCORE
//#define ARMA_BLAS_LONG
//#define ARMA_BLAS_LONG_LONG


#include <armadillo>

#include "Deng_vector.hpp"
#include "GOAT_RK4.hpp"
#include "Optimization.hpp"
//using namespace arma;


/*
void output_runtime_message(std::string specific_job, clock_t global_timer, std::ofstream output_file)
{
    clock_t runtime_timer = clock() - global_timer;
    std::cout << "Finished " << specific_job << " at " << ((float)runtime_timer)/CLOCKS_PER_SEC << "s" << std::endl;
    output_file << "Finished " << specific_job << " at " << ((float)runtime_timer)/CLOCKS_PER_SEC << "s" << std::endl;
}
*/

/*
const double tau = 10.0;
const int N_t = 10;
const int N = 512;
const int dim_para = 1;

Deng::Col_vector<arma::Mat<std::complex<double> > > Time_Evolution(double time)
{
    Deng::Col_vector<arma::Mat<std::complex<double> > > iH(dim_para + 1);

    for(int i = 0; i <= dim_para; ++i)
    {
        iH[i].zeros(N, N);
    }

    return iH;
}
*/
//mathematical constants
#ifndef DENG_PI_DEFINED
#define DENG_PI_DEFINED
const double Pi = 3.14159265358979324;
const double Pi_sqrt = sqrt(Pi);
const double Pi_fourth = sqrt(Pi_sqrt);
const std::complex<double> imag_i(0, 1.0);
#endif


class QHO_1D : public Deng::GOAT::Hamiltonian<std::complex<double> >
{
    friend class Deng::GOAT::RK4<std::complex<double> >;
public:
    virtual Deng::Col_vector<arma::Mat<std::complex<double> > > dynamics(double t) const override;
    arma::Mat<std::complex<double> > T_x_base;
    arma::Mat<std::complex<double> > V_x_base;
    arma::Mat<std::complex<double> > control_field;
    //inherit constructor
    using Deng::GOAT::Hamiltonian<std::complex<double> >::Hamiltonian;
    QHO_1D(int N, int N_t, double tau, double L, double omega_0, double omega_tau, int dim_para = 0, double hbar = 1.0);
    void Initialize_Harmonic_Oscillator();
protected:
    double _L;
    double _hbar;
    double _dx;
    double _x_min;
    double _dp;
    double _omega_0;
    double _omega_tau;
};
//either constructor should be in this way, or Hamiltonian() is explicitly defined
QHO_1D::QHO_1D(int N, int N_t, double tau, double L, double omega_0, double omega_tau, int dim_para, double hbar) :
    Deng::GOAT::Hamiltonian<std::complex<double> >(N, N_t, tau, dim_para), _L(L), _omega_0(omega_0), _omega_tau(omega_tau), _hbar(hbar)
{
    T_x_base.set_size(_N, _N);
    T_x_base.zeros();
    V_x_base.set_size(_N, _N);
    V_x_base.zeros();
    control_field.set_size(_N, _N);
    control_field.zeros();

    _dx = _L/_N;
    _x_min = _dx/2 - _L/2;
    _dp = 2*Pi/_L*_hbar;

    const double Miller_x_min = _x_min - _dx;
    const double Miller_x_max = -Miller_x_min;
    const double Miller_L = Miller_x_max - Miller_x_min;
    const double Miller_prefactor = sqrt(2.0/Miller_L);
    const int Miller_N = N + 1;

    //Miller DVR, a novel ......
    for(int i = 0; i < N; ++i)
    {
        V_x_base(i, i) = pow(_x_min + i*_dx, 2.0);//V(x) = x^2, which is not the final hamiltonian we use
        control_field(i, i) = pow(_x_min + i*_dx, 2.0);
        int k = i + 1;
        //Miller DVR A6
        T_x_base(i, i) = pow(1.0 * Pi / Miller_L, 2.0)/(4.0)*((2.0*Miller_N*Miller_N + 1.0)/3.0 - 1.0/pow(sin((Pi * k)/Miller_N), 2.0));
        for(int iprime = 0; iprime < i; ++iprime)
        {
            int kprime = iprime + 1;
            if( ((k - kprime)%2) == 1)
            {
                T_x_base(i, iprime) = -pow(1.0 * Pi / Miller_L, 2.0)/(4.0)*((1.0/pow(sin(Pi*(k - kprime)/2.0/Miller_N), 2.0)) - (1.0/pow(sin(Pi*(k + kprime)/2.0/Miller_N), 2.0)));
            }
            else
            {
                T_x_base(i, iprime) = pow(1.0 * Pi / Miller_L, 2.0)/(4.0)*((1.0/pow(sin(Pi*(k - kprime)/2.0/Miller_N), 2.0)) - (1.0/pow(sin(Pi*(k + kprime)/2.0/Miller_N), 2.0)));
            }
            T_x_base(iprime, i) = T_x_base(i, iprime);
        }
    }

    //test for my T. Need modification
    /*
    const double L = 10.0;
    const int N = 128;
    const double dx = L/N;
    const double dp = 2*Pi/L;
    arma::Mat<std::complex<double> > kinetic_energy(N, N, arma::fill::zeros);
    arma::Col<std::complex<double> > x_func(N);
    for(int i = 0; i < N; ++i)
    {
        x_func(i) = dx/2 - L/2 + i*dx;
        if(i<=N/2)
            kinetic_energy(i, i) = i*dp;
        else
            //kinetic_energy(i, i) = i*dp;
            kinetic_energy(i, i) = (i - N)*dp;
    }
    //kinetic energy is assumed to be invariant in most cases
    kinetic_energy = kinetic_energy*kinetic_energy/2.0;//mass is assumed to be 1.0
    kinetic_energy = arma::ifft2(kinetic_energy);
    arma::Col<std::complex<double> > func_to_be_diff = x_func%x_func%x_func;

    arma::Mat<double> output(N, 2);
    output.col(0) = arma::real(-kinetic_energy*func_to_be_diff);
    output.col(1) = arma::real(output.col(0)/(x_func%x_func*3.0));

    std::cout << output;
    //std::cout << arma::real(kinetic_energy);
    */
}
Deng::Col_vector<arma::Mat<std::complex<double> > > QHO_1D::dynamics(double t) const
{


    Deng::Col_vector<arma::Mat<std::complex<double> > > iH_and_partial_H(_dim_para + 1);
    iH_and_partial_H[0] =imag_i*(T_x_base + 0.5*pow(_omega_0 + (_omega_tau - _omega_0)*t/_tau, 2.0)*V_x_base);
    for(int i = 0; i < _dim_para; ++i)
    {
        if(i%2 == 0)
        {
            iH_and_partial_H[0] = imag_i*(iH_and_partial_H[0] + cos(i*Pi*t/_tau)*parameters[i]*control_field);
            iH_and_partial_H[i+1] = imag_i*(cos(i*Pi*t/_tau)*control_field);
        }
        else
        {
            iH_and_partial_H[0] = imag_i*(iH_and_partial_H[0] + sin((i+1)*Pi*t/_tau)*parameters[i]*control_field);
            iH_and_partial_H[i+1] = imag_i*(sin((i+1)*Pi*t/_tau)*control_field);
        }
    }

    return iH_and_partial_H;
}
/*
void QHO_1D::Initialize_Harmonic_Oscillator()
{
    kinetic_energy.set_size(_N, _N);
    kinetic_energy.zeros();
    potential_energy.set_size(_N, _N);
    potential_energy.zeros();
    control_field.set_size(_N, _N);
    control_field.zeros();

    for(int i = 0; i < _N; ++i)
    {
        potential_energy(i, i) = _x_min + i*_dx;

        if(i<=_N/2)
            kinetic_energy(i, i) = i*_dp;
        else
            kinetic_energy(i, i) = (i - _N)*_dp;
    }
}
*/

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



int main()
{
    clock_t runtime_timer;
    std::ofstream runtime_statistics;
    runtime_statistics.open("runtime_statistics.dat");

    runtime_timer = clock();

    //output_runtime_message("lalala", runtime_timer, runtime_statistics);

    const int N = 1024;
    const int N_t = 20;
    const double tau = 3.0;
    const double L = 16.0;
    const double omega_0 = 1.0;
    const double omega_tau = 2.0;
    const int dim_para = 5;
    const double hbar = 1.0;

    GOAT_OCT Opt_Ctrl(2);
    std::cout << Opt_Ctrl.Conj_Grad_Search({9.0, 11.0}) << std::endl;




    /*
    QHO_1D H(N, N_t, tau, L, omega_0, omega_tau, dim_para, hbar);
    Deng::GOAT::RK4<std::complex<double> > RungeKutta;
    //std::cout << "here?" << std::endl;
    RungeKutta.Prep_for_H(H);
    RungeKutta.Evolve_to_final(H);


    runtime_timer = clock() - runtime_timer;
    std::cout << "costs " << ((float)runtime_timer)/CLOCKS_PER_SEC << "s" << std::endl;
    runtime_statistics << "costs " << ((float)runtime_timer)/CLOCKS_PER_SEC << "s" << std::endl;
    */






    /*
    //test diagonalization
    arma::Mat<std::complex<double> > H_x = H.T_x_base + H.V_x_base;
    vec eigval2;
    cx_mat eigvec2;

    eig_sym(eigval2, eigvec2, H_x);
    std::cout << eigval2.subvec(0,20);
    */

    //std::cout << arma::real(H.T_x_base);
    /*
    arma::Col<std::complex<double> > x(N), x_func(N);
    arma::Mat<std::complex<double> > kinetic_energy(N, N, arma::fill::zeros);
    const double dx = L/N;
    const double dp = 2*Pi/L;
    for(int i = 0; i < N; ++i)
    {
        x(i) = dx/2 - L/2 + i*dx;
        if(i<=N/2)
            kinetic_energy(i, i) = i*dp;
        else
            //kinetic_energy(i, i) = i*dp;
            kinetic_energy(i, i) = (i - N)*dp;
    }
    kinetic_energy = kinetic_energy*kinetic_energy/2.0;//mass is assumed to be 1.0
    kinetic_energy = arma::ifft2(kinetic_energy);
    //std::cout << x;
    x_func = x%x%x;
    arma::Mat<double> output(N, 2);
    output.col(0) = arma::real(H.T_x_base*x_func);
    output.col(1) = arma::real(kinetic_energy*x_func)-output.col(0);
    //std::cout << arma::real(-H.T_x_base*x_func);
    //std::cout << arma::real(x_func);
    std::cout << output;
    std::cout << arma::real(kinetic_energy);
    //std::cout << kinetic_energy;
    */


    /*
    const int dim_para = 5;
    const int dim = 4;

    Deng::Col_vector<arma::mat> a(dim_para), b(dim_para), c(dim_para);

    for(int i = 0; i < dim_para; ++i)
    {
        a[i] = eye<mat>(dim, dim)*i;
        b[i] = eye<mat>(dim, dim)*(i + 1)*(i + 1);
    }
    c = a + b;

    for(int i = 0; i < dim_para; ++i)
        std::cout << b[i];
    */




    //Deng::GOAT::Hamiltonian<std::complex<double>>(N, N_t, tau, dim_para);

/*
    clock_t runtime_timer;
    std::ofstream runtime_statistics;
    runtime_statistics.open("runtime_statistics.dat");

    const int matrix_size = 4000;


    runtime_timer = clock();

    mat A = randu<mat>(matrix_size, matrix_size);

    runtime_timer = clock() - runtime_timer;
    std::cout << "Generating random matrix of size " << matrix_size << " costs " << ((float)runtime_timer)/CLOCKS_PER_SEC << "s" << std::endl;
    runtime_statistics << "Generating random matrix of size " << matrix_size << " costs " << ((float)runtime_timer)/CLOCKS_PER_SEC << "s" << std::endl;
*/


/*
    runtime_timer = clock();

    mat B = A.t() * A;  // generate a symmetric matrix

    runtime_timer = clock() - runtime_timer;
    std::cout << "Multiplying random matrix of size " << matrix_size << " costs " << ((float)runtime_timer)/CLOCKS_PER_SEC << "s" << std::endl;
    runtime_statistics << "Multiplying random matrix of size " << matrix_size << " costs " << ((float)runtime_timer)/CLOCKS_PER_SEC << "s" << std::endl;
*/


/*
    runtime_timer = clock();

    vec eigval;
    mat eigvec;
    eig_sym(eigval, eigvec, B);

    runtime_timer = clock() - runtime_timer;
    std::cout << "Diagonalizing random matrix of size " << matrix_size << " costs " << ((float)runtime_timer)/CLOCKS_PER_SEC << "s" << std::endl;
    runtime_statistics << "Diagonalizing random matrix of size " << matrix_size << " costs " << ((float)runtime_timer)/CLOCKS_PER_SEC << "s" << std::endl;
*/
    return 0;

}
