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

	//pressumed constants of the model
	const real B_x_max = 1.5;
	const real B_y_max = 0.0;
	const real B_z_max = 1.0;
	const real omega;
	const real _hbar;
	//Pauli matrix, magnetic field and its partial derivative wrt time
	Deng::Col_vector<arma::Mat<elementtype>> S;
	arma::Mat<elementtype> S_identity;

	//intermediate matrix
	//total S alone 3 directions
	Deng::Col_vector<arma::Mat<elementtype>> S_total;
	//nereast interactions, usually does not change with time
	arma::Mat<elementtype> interaction;

	//external magnetic field
	Deng::Col_vector<real> B(real t) const;
	//calculate control magnetic field based on the parameters
	Deng::Col_vector<real> control_field(real t) const;

	//inherit constructor
	//using Deng::GOAT::Hamiltonian<elementtype, real>::Hamiltonian;
	Transverse_Ising(unsigned int num_spinor, unsigned int N_t, real tau, unsigned int dim_para = 0, real hbar = 1.0);

	//give the bare Hamiltonian including the external B field, w/o the control field
	arma::Mat<elementtype> H_0(real t) const;
	//give the control Hamiltonian
	arma::Mat<elementtype> H_control(real t) const;

};
Transverse_Ising::Transverse_Ising(unsigned int num_spinor, unsigned int N_t, real tau, unsigned int dim_para, real hbar) :
	Deng::GOAT::Hamiltonian<elementtype, real>((1<<num_spinor), N_t, tau, dim_para), omega(2.0*Pi / tau), _hbar(hbar)
	//2 for the dimension of spin half system
{	
	//set up Pauli matrices
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
	S_identity.zeros(2, 2);
	S_identity(0, 0) = 1;
	S_identity(1, 1) = 1;
	//std::cout << S_identity;
	//S = 0.5*_hbar*S;//Eq 3.2
	

	//initialize the interaction part
	interaction.set_size(_N, _N);
	interaction.zeros();
	//calculate the interaction part
	for (unsigned int i = 0; i < (num_spinor - 1); ++i)
	{
		auto temp = i == 0 ? S[2] : S_identity;
		for (unsigned int j = 1; j < num_spinor; ++j)
		{
			temp = arma::kron(temp, (j == i) || (j == i + 1) ? S[2] : S_identity);
		}
		interaction += temp;
	}

	//total S alone 3 directions
	S_total.set_size(3);
	for (unsigned int i = 0; i < 3; ++i)
	{
		S_total[i].set_size(_N, _N);
		S_total[i].zeros();
		//calculate the total S alone 3 directions
		for (unsigned int j = 0; j < num_spinor; ++j)
		{
			auto temp_B = j == 0 ? S[i] : S_identity;
			for (unsigned int k = 1; k < num_spinor; ++k)
			{
				temp_B = arma::kron(temp_B, j == k ? S[i] : S_identity);
			}
			S_total[i] += temp_B;
		}
	}
}
Deng::Col_vector<real> Transverse_Ising::B(real t) const
{
	Deng::Col_vector<real> B_field(3);

	//rotating B field
	//B_field[0] = cos(0.25*omega*t)*B_x_max;
	//B_field[1] = B_y_max;
	//B_field[2] = sin(0.25*omega*t)*B_z_max;
	B_field[0] = 0.0;// cos(0.25*omega*t)*B_x_max;
	B_field[1] = 0.2;// B_y_max;
	B_field[2] = B_z_max*t/_tau;//sin(0.25*omega*t)*B_z_max;

	return B_field;
}
//calculate control magnetic field based on the parameters
Deng::Col_vector<real> Transverse_Ising::control_field(real t) const
{
	Deng::Col_vector<real> ctrl(3);
	ctrl[0] = 0.0;
	ctrl[1] = 0.0;
	ctrl[2] = 0.0;

	for (unsigned int i = 1; i < _dim_para; ++i)
	{
		unsigned int mode = (i + 3) / 4;
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
arma::Mat<elementtype> Transverse_Ising::H_0(real t) const
{
	//bracket is necessary as operator ^ is of low priority
	return interaction + (B(t) ^ S_total);
}
arma::Mat<elementtype> Transverse_Ising::H_control(real t) const
{
	return control_field(t)^S_total;
}
Deng::Col_vector<arma::Mat<elementtype> > Transverse_Ising::Dynamics(real t) const
{
	Deng::Col_vector<arma::Mat<elementtype> > iH_and_partial_H(_dim_para + 1);

	iH_and_partial_H[0] = Dynamics_U(t);

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
arma::Mat<elementtype> Transverse_Ising::Dynamics_U(real t) const
{
	return H_0(t) + H_control(t);
}

