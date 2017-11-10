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
	friend class Deng::GOAT::RK4<Field, Parameter>;
protected:
	//pressumed constants of the model
	Parameter _B_x_max = 0.65;
	Parameter _B_y_max = 0.0;
	Parameter _B_z_max = 1.1;
	Parameter _omega;
	Parameter _hbar;
	Parameter _J = 1.0;
	unsigned int _num_spin;
	unsigned int _dim_para_each_direction;
public:
	//Pauli matrix, magnetic field and its partial derivative wrt time
	Deng::Col_vector<arma::Mat<Field>> S;
	arma::Mat<Field> S_identity;
	//total S alone 3 directions
	Deng::Col_vector<arma::Mat<Field>> S_total;
	//nereast interactions, usually does not change with time
	arma::Mat<Field> interaction;

	//member functions
	Transverse_Ising(unsigned int num_spin, unsigned int N_t, Parameter tau, unsigned int dim_para, unsigned int dim_para_each_direction, Parameter hbar = 1.0);

	//external magnetic field
	Deng::Col_vector<Parameter> B(Parameter t) const;
	//give the bare Hamiltonian including the external B field, w/o the control field
	arma::Mat<Field> H_0(Parameter t) const;

	//calculate control magnetic field based on the parameters
	Deng::Col_vector<Parameter> control_field(Parameter t) const;
	Parameter control_field_component(Parameter t, unsigned int para_idx_begin) const;
	//give the control Hamiltonian
	virtual arma::Mat<Field> H_control(Parameter t) const;

	//give the dynamics 
	virtual Deng::Col_vector<arma::Mat<Field>> Dynamics(Parameter t) const override;
	virtual arma::Mat<Field> Dynamics_U(Parameter t) const override;

	virtual ~Transverse_Ising() = default;
};




//transeverse Ising model, with local control fields
template<typename Field, typename Parameter>
class Transverse_Ising_Local_Control : public Transverse_Ising<Field, Parameter>
{
	friend class Deng::GOAT::RK4<Field, Parameter>;
public:
	//S of each spin
	Deng::Col_vector<arma::Mat<Field>>* S_each;

	Transverse_Ising_Local_Control(unsigned int num_spin, unsigned int N_t, Parameter tau,
		unsigned int dim_para, unsigned int dim_para_each_direction, Parameter hbar = 1.0);

	//give the control Hamiltonian
	Deng::Col_vector<Parameter> local_control_field(Parameter t, unsigned int ith_spin) const;
	virtual arma::Mat<Field> H_control(Parameter t) const override;

	//give the dynamics
	virtual Deng::Col_vector<arma::Mat<Field>> Dynamics(Parameter t) const override;

	virtual ~Transverse_Ising_Local_Control()
	{
		delete[] S_each;
	}
};



#endif // !DENG_ISING_HPP

