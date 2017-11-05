#include <iostream>
#include <cmath>
#include <ctime>


#include <armadillo>
#include "Optimization.hpp"
#include "GOAT_RK4.hpp"
#include "Deng_vector.hpp"


typedef float real;
typedef std::complex<float> elementtype;


//mathematical constants
#ifndef DENG_PI_DEFINED
#define DENG_PI_DEFINED
const double Pi = 3.14159265358979324;
const double Pi_sqrt = sqrt(Pi);
const double Pi_fourth = sqrt(Pi_sqrt);
const std::complex<double> imag_i(0, 1.0);
#endif


//Berry's transition-less driving
class Single_spin_half : public Deng::GOAT::Hamiltonian<elementtype, real>
{
    friend class Deng::GOAT::RK4<elementtype, real>;
public:
    virtual Deng::Col_vector<arma::Mat<elementtype>> Dynamics(real t) const override;
    //Pauli matrix, magnetic field and its partial derivative wrt time
    Deng::Col_vector<arma::Mat<elementtype>> S;
    Deng::Col_vector<real> B(real t) const;
    Deng::Col_vector<real> partialB(real t) const;
    //control magnetic field
    Deng::Col_vector<real> control_field(real t) const;
    //inherit constructor
    using Deng::GOAT::Hamiltonian<elementtype, real>::Hamiltonian;
    Single_spin_half(int N_t, real tau, int dim_para = 0, real hbar = 1.0);
    const real B_x_max = 1.0;
    const real B_y_max = 1.0;
    const real B_z_max = 2.0;
    const real omega;
protected:
    double _hbar;
};
Single_spin_half::Single_spin_half(int N_t, real tau, int dim_para, real hbar) :
    Deng::GOAT::Hamiltonian<elementtype, real> (2, N_t, tau, dim_para), omega(2.0*Pi/tau), _hbar(hbar)
    //2 for the dimension of spin half system
{
    S.set_size(3);
    S[0].zeros(2, 2);
    S[0](0, 1) = 1;
    S[0](1, 0) = 1;
    S[1].zeros(2, 2);
    S[1](0, 1) = -imag_i;
    S[1](1, 0) = imag_i;
    S[2].zeros(2, 2);
    S[2](0, 0) = 1;
    S[2](1, 1) = -1;
    S = 0.5*_hbar*S;//Eq 3.2
}
Deng::Col_vector<real> Single_spin_half::B (real t) const
{
    Deng::Col_vector<real> B_field(3);

    //rotating B field
    B_field[0] = cos(omega*t)*B_x_max;
    B_field[1] = sin(omega*t)*B_y_max;
    B_field[2] = B_z_max;

    return B_field;
}
Deng::Col_vector<real> Single_spin_half::control_field(real t) const
{
//    //Berry transitionless field
//    Deng::Col_vector<double> dB(3);
//    Deng::Col_vector<double> B_field(3);
//    B_field = B(t);
//    //derivative of B field
//    dB[0] = -omega*sin(omega*t)*B_x_max;
//    dB[1] =  omega*cos(omega*t)*B_y_max;
//    dB[2] = 0;
//
//    Deng::Col_vector<double> Ctrl(3);
//    //Eq. 3.8
//    Ctrl[0] = B_field[1]*dB[2] - B_field[2]*dB[1];
//    Ctrl[1] = B_field[2]*dB[0] - B_field[0]*dB[2];
//    Ctrl[2] = B_field[0]*dB[1] - B_field[1]*dB[0];
//    //Eq. 3.9
//    Ctrl = 1/(B_field^B_field)*Ctrl;
//    //Ctrl = 0.5 * Ctrl;

    Deng::Col_vector<real> ctrl(3);
    ctrl[0] = 0.0;
    ctrl[1] = 0.0;
    ctrl[2] = parameters[0];

    for(int i = 1; i < _dim_para; ++i)
    {
        int mode = (i+3)/4;
		if (((i - 1) % 4) <= 1)
		{
			ctrl[(i - 1) % 2] += parameters[i] * sin(mode * 2 * Pi*t / _tau);
		}
		else
		{
			ctrl[(i - 1) % 2] += parameters[i] * cos(mode * 2 * Pi*t / _tau);
		}

    }

    return ctrl;
}
Deng::Col_vector<arma::Mat<elementtype>> Single_spin_half::Dynamics(real t) const
{
    Deng::Col_vector<arma::Mat<elementtype>> iH_and_partial_H(_dim_para + 1);

    iH_and_partial_H[0] = (-imag_i/_hbar)*((B(t) + control_field(t))^S);

	for (unsigned int i = 1; i <= _dim_para; ++i)
	{
		//could be generalize?
		//double original_para = parameters[i];
		parameters[i-1] += 0.01;
		Deng::Col_vector<real> partial_control = control_field(t);
		parameters[i-1] -= 0.01;
		partial_control = (1/0.01)*(partial_control - control_field(t));

		iH_and_partial_H[i] = (-imag_i / _hbar)*(partial_control^S);
	}

    return iH_and_partial_H;
}






int main(int argc, char** argv)
{
    const int N_t = 2000;
    const double tau = 3.0;
    const int dim_para = std::stoi(argv[1]);
    //const double hbar = 1.0;

    const double epsilon = 0.01;
    const int max_iteration = 20;

    Single_spin_half H_only(N_t, tau, 0);
    Single_spin_half H_and_partial_H(N_t, tau, dim_para);
    Deng::GOAT::RK4<elementtype, real> RungeKutta;

    //GOAT_Target target(&RungeKutta, &H_only, &H_and_partial_H);
	Deng::GOAT::GOAT_Target_1st_order<elementtype, real> target(&RungeKutta, &H_and_partial_H);


    arma::Col<real> eigval_0;
    arma::Mat<elementtype> eigvec_0;
    arma::Mat<elementtype> H_0 = H_and_partial_H.B(0)^ H_and_partial_H.S;
    arma::eig_sym(eigval_0  , eigvec_0  , H_0  );

    arma::Col<real> eigval_tau;
    arma::Mat<elementtype> eigvec_tau;
    arma::Mat<elementtype>H_tau = H_and_partial_H.B(tau)^ H_and_partial_H.S;
    arma::eig_sym(eigval_tau, eigvec_tau, H_tau);

    arma::Mat<elementtype> unitary_goal = eigvec_0;
    unitary_goal.zeros();

	std::cout << eigvec_0 << eigvec_tau << std::endl;

    unitary_goal += eigvec_tau.col(0)*eigvec_0.col(0).t()*exp(-2.59742*imag_i);
    unitary_goal += eigvec_tau.col(1)*eigvec_0.col(1).t()*exp(-0.544176*imag_i);

    target.Set_Controlled_Unitary_Matrix(unitary_goal);
	std::cout << unitary_goal << std::endl;

    //double aa;
    //arma::Col<double> position(dim_para, arma::fill::zeros);
    //std::cout << target.negative_gradient(position, aa) << std::endl;

    Deng::Optimization::Min_Conj_Grad<real> Conj_Grad(dim_para, epsilon, max_iteration);


    Conj_Grad.Assign_Target_Function(&target);
    Conj_Grad.Opt_1D = Deng::Optimization::OneD_Golden_Search<real>;
	arma::arma_rng::set_seed(time(nullptr));
	arma::Col<real> start(dim_para, arma::fill::randu);
	start = start - 1;
    start = Conj_Grad.Conj_Grad_Search(start);





//    arma::Col<double> berry_control(dim_para, arma::fill::zeros);
//    berry_control[0] =  2.0*Pi/tau/5.0;
//    berry_control[3] = -4.0*Pi/tau/5.0;
//    berry_control[2] = -4.0*Pi/tau/5.0;
//    Conj_Grad.Conj_Grad_Search(berry_control);
//    target.function_value(berry_control);
//    std::complex<double> phase0, phase1;
//    phase0 = arma::as_scalar(eigvec_tau.col(0).t()*RungeKutta.next_state[0]*eigvec_0.col(0));
//    phase1 = arma::as_scalar(eigvec_tau.col(1).t()*RungeKutta.next_state[0]*eigvec_0.col(1));
//    std::cout << std::conj(phase0)*phase0 << " with arg " << std::arg(phase0) <<std::endl;
//    std::cout << std::conj(phase1)*phase1 << " with arg " << std::arg(phase1) <<std::endl;




    return 0;
}
