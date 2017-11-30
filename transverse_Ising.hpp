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
//typedef double Parameter;
//typedef std::complex<Parameter> Field;

//mathematical constants
#ifndef DENG_PI_DEFINED
#define DENG_PI_DEFINED
const double Pi = 3.14159265358979324;
const double Pi_sqrt = sqrt(Pi);
const double Pi_fourth = sqrt(Pi_sqrt);
const std::complex<double> imag_i(0, 1.0);
#endif

//transeverse Ising model, with a unified control B field
template<typename Field, typename Parameter>
class Transverse_Ising : public Deng::GOAT::Hamiltonian<Field, Parameter>
{
	//friend class Deng::GOAT::RK4<Field, Parameter>;
protected:
	//pressumed constants of the model
	Parameter _B_x_max = 0.65;
	Parameter _B_y_max = 0.0;
	Parameter _B_z_max = 1.1;
    Parameter _J = 1.0;
	Parameter _omega;
    Parameter _hbar;
	const Field minus_i_over_hbar;
	unsigned int _num_spin;
	//an important const
	//specifies how the parameters are turned to control field
	//on each direction (or even of each spin)
	unsigned int _dim_para_each_direction;
protected:
	//important intermediate variables
	//will be used a lot
	//Pauli matrix, magnetic field and its partial derivative wrt time
	Deng::Col_vector<arma::Mat<Field>> S;
	arma::Mat<Field> S_identity;
	//total S alone 3 directions
	Deng::Col_vector<arma::Mat<Field>> S_total;
	//nereast interactions, usually does not change with time
	arma::Mat<Field> interaction;
public:
	//constructor
	Transverse_Ising(const unsigned int num_spin, const unsigned int N_t, const Parameter tau,
	const unsigned int dim_para, const unsigned int dim_para_each_direction, const Parameter hbar = 1.0);

	//external magnetic field
	//if one want to give only initial and final B
	//just let B be constant and jump to final B at t = tau
	Deng::Col_vector<Parameter> B(Parameter t) const;

	//give the bare Hamiltonian including the external B field, w/o the control field
	arma::Mat<Field> H_0(Parameter t) const;

	//calculate control magnetic field based on the parameters
	Deng::Col_vector<Parameter> control_field(Parameter t) const;
	//calculate magnitude of control magnetic field
	//along certain direction (of certain spin)
	//the index of direction (as well as spin) is given by para_idx_begin
	//REMARK: this virtual function is for sinusoidal control B field. 
	//        One should override it when treating Gaussian impulse or other type of control
	virtual Parameter control_field_component(const Parameter t, const unsigned int para_idx_begin) const;
	//similar with the above, has additional argument para_idx_derivative (relative to para_idx_begin) 
	//to indicate which parameter we are doing derivative against
	//hence further evaluate H_alpha
	//REMARK1: here we assume the parameter will not affect B field along other direction (or of other spin)
	//REMARK2: this virtual function is for sinusoidal control B field. 
	//		   One should override it when treating Gaussian impulse or other type of control
	virtual Parameter control_field_component_derivative(const Parameter t, const unsigned int para_idx_begin, const unsigned int para_idx_derivative) const;

	//give the control Hamiltonian
	virtual arma::Mat<Field> H_control(Parameter t) const;

	//give the dynamics
	virtual Deng::Col_vector<arma::Mat<Field>> Dynamics(const Parameter t) const override;
	virtual arma::Mat<Field> Dynamics_U(const Parameter t) const override;

	virtual ~Transverse_Ising() = default;
};




//transeverse Ising model, with local control fields
//i.e. apply different field to each spin
template<typename Field, typename Parameter>
class Transverse_Ising_Local_Control : public Transverse_Ising<Field, Parameter>
{
	//friend class Deng::GOAT::RK4<Field, Parameter>;
public:
	//S of each spin
	Deng::Col_vector<arma::Mat<Field>>* S_each;

	Transverse_Ising_Local_Control(const unsigned int num_spin, const unsigned int N_t, const Parameter tau,
		const unsigned int dim_para, const unsigned int dim_para_each_direction, const Parameter hbar = 1.0);

	//calculate local control field
	virtual Deng::Col_vector<Parameter> local_control_field(Parameter t, unsigned int ith_spin) const;
	//calculate derivative of local control field
	virtual Deng::Col_vector<Parameter> local_control_field_derivative(Parameter t, unsigned int ith_spin, const unsigned int para_idx_derivative) const;
	//override the control Hamiltonian
	virtual arma::Mat<Field> H_control(Parameter t) const override;

	//give the dynamics, as U_alpha will be different
	//however, U and H_0 are still the same, therefore no need to override Dynamics_U
	virtual Deng::Col_vector<arma::Mat<Field>> Dynamics(const Parameter t) const override;

	//virtual destructor
	virtual ~Transverse_Ising_Local_Control() { delete[] S_each; };
};






#endif // !DENG_ISING_HPP

