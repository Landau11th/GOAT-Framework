#include <iostream>
#include <cmath>
#include <ctime>


#include <armadillo>
#include "Optimization.hpp"
#include "GOAT_RK4.hpp"
#include "Deng_vector.hpp"

//define matrix element type
typedef float real;
typedef std::complex<real> elementtype;

//mathematical constants
#ifndef DENG_PI_DEFINED
#define DENG_PI_DEFINED
const real Pi = 3.14159265358979324;
const real Pi_sqrt = sqrt(Pi);
const real Pi_fourth = sqrt(Pi_sqrt);
const std::complex<real> imag_i(0, 1.0);
#endif

//transeverse Ising model
class Transverse_Ising : public Deng::GOAT::Hamiltonian<elementtype, real>
{
	friend class Deng::GOAT::RK4<elementtype, real>;
public:
	//give the dynamics 
	virtual Deng::Col_vector<arma::Mat<elementtype>> Dynamics(real t) const override;
	virtual arma::Mat<elementtype> Dynamics_U(real t) const override;

	//Pauli matrix, magnetic field and its partial derivative wrt time
	Deng::Col_vector<arma::Mat<elementtype>> S;
	//external magnetic field and its partial derivative wrt time
	Deng::Col_vector<real> B(real t) const;
	Deng::Col_vector<real> dB(real t) const;
	//control magnetic field
	Deng::Col_vector<real> control_field(real t) const;

	//inherit constructor
	using Deng::GOAT::Hamiltonian<elementtype, real>::Hamiltonian;
	Transverse_Ising(int N_t, real tau, int dim_para = 0, real hbar = 1.0);

	//give the bare Hamiltonian including the external B field, w/o the control field
	arma::Mat<elementtype> H_bare(real t) const;
	//give the control Hamiltonian
	arma::Mat<elementtype> H_control(real t) const;
	//pressumed constants of the model
	const real B_x_max = 1.0;
	const real B_y_max = 1.0;
	const real B_z_max = 2.0;
	const real omega;
protected:
	real _hbar;
};
Transverse_Ising::Transverse_Ising(int N_t, real tau, int dim_para, real hbar) :
	Deng::GOAT::Hamiltonian<elementtype, real>(2, N_t, tau, dim_para), omega(2.0*Pi / tau), _hbar(hbar)
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
Deng::Col_vector<real> Transverse_Ising::B(real t) const
{
	Deng::Col_vector<real> B_field(3);

	//rotating B field
	B_field[0] = cos(omega*t)*B_x_max;
	B_field[1] = sin(omega*t)*B_y_max;
	B_field[2] = B_z_max;

	return B_field;
}
Deng::Col_vector<real> Transverse_Ising::dB(real t) const
{
	Deng::Col_vector<real> B_field(3);

	//rotating B field
	B_field[0] = -omega*sin(omega*t)*B_x_max;
	B_field[1] = omega*cos(omega*t)*B_y_max;
	B_field[2] = 0;

	return B_field;
}
Deng::Col_vector<real> Transverse_Ising::control_field(real t) const
{
	Deng::Col_vector<real> ctrl(3);
	ctrl[0] = 0.0;
	ctrl[1] = 0.0;
	ctrl[2] = parameters[0];

	for (int i = 1; i < _dim_para; ++i)
	{
		int mode = (i + 3) / 4;
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
Deng::Col_vector<arma::Mat<elementtype> > Transverse_Ising::Dynamics(real t) const
{
	Deng::Col_vector<arma::Mat<elementtype> > iH_and_partial_H(_dim_para + 1);

	iH_and_partial_H[0] = (-imag_i / _hbar)*((B(t) + control_field(t)) ^ S);

	for (unsigned int i = 1; i <= _dim_para; ++i)
	{
		//could be generalize?
		//double original_para = parameters[i];
		parameters[i - 1] += 0.01;
		Deng::Col_vector<real> partial_control = control_field(t);
		parameters[i - 1] -= 0.01;
		partial_control = (1 / 0.01)*(partial_control - control_field(t));

		iH_and_partial_H[i] = (-imag_i / _hbar)*(partial_control^S);
	}

	return iH_and_partial_H;
}
arma::Mat<elementtype> Transverse_Ising::H_control(real t) const
{
	return 0;
}

