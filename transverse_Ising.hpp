#ifndef DENG_ISING_HPP
#define DENG_ISING_HPP

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




//#define ISING_CONTROL_INDIVIDUAL

//transeverse Ising model
class Transverse_Ising : public Deng::GOAT::Hamiltonian<elementtype, real>
{
	friend class Deng::GOAT::RK4<elementtype, real>;
public:
	//give the dynamics 
	virtual Deng::Col_vector<arma::Mat<elementtype>> Dynamics(real t) const override;
	virtual arma::Mat<elementtype> Dynamics_U(real t) const override;

	//pressumed constants of the model
	const real B_x_max = 0.6;
	const real B_y_max = 0.0;
	const real B_z_max = 1.725;
	const real omega;
	const real _hbar;
	//Pauli matrix, magnetic field and its partial derivative wrt time
	Deng::Col_vector<arma::Mat<elementtype>> S;
	arma::Mat<elementtype> S_identity;

	//intermediate matrix
	//total S alone 3 directions
	Deng::Col_vector<arma::Mat<elementtype>> S_total;
#ifdef ISING_CONTROL_INDIVIDUAL
	//S of each spin
	Deng::Col_vector<arma::Mat<elementtype>>* S_each;
#endif
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

	virtual ~Transverse_Ising()
	{
#ifdef ISING_CONTROL_INDIVIDUAL
		delete[] S_each;
#endif
	}

};



#endif // !DENG_ISING_HPP





