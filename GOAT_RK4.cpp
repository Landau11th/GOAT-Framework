//#include <typeinfo>
#include "GOAT_RK4.hpp"

using namespace Deng::GOAT;


template class Deng::GOAT::Hamiltonian<float , float >;
template class Deng::GOAT::Hamiltonian<double, double>;
template class Deng::GOAT::Hamiltonian<std::complex<float> , float >;
template class Deng::GOAT::Hamiltonian<std::complex<double>, double>;

template <typename Field, typename Parameter>
Hamiltonian<Field, Parameter>::Hamiltonian(unsigned int N, unsigned int N_t, Parameter tau, unsigned int dim_para)
	: _N(N), _N_t(N_t), _tau(tau), _dim_para(dim_para), _dt(tau/N_t)
{
	//set the parameters to 0
	parameters.zeros(_dim_para);
}
//time derivative of U(t) and partial U partial alpha
//could be leveraged by Runge Kutta or other similar methods
//i.e. when block matrix/vector could be used to optimize computation
template <typename Field, typename Parameter>
Deng::Col_vector<arma::Mat<Field>> Hamiltonian<Field, Parameter>::Derivative(const Deng::Col_vector<arma::Mat<Field>>& U, const unsigned int t_index, const bool half_time) const
{
    Parameter shift = half_time ? 0.5 : 0.0;//caution: 0 or 1???
	//need to test the data type
    Parameter t = (t_index + shift)*_dt;

	//+1 to include both time derivatives of U(t) and partial U partial alpha
    Deng::Col_vector<arma::Mat<Field>> k(_dim_para + 1);

	//dynamics of a specific model
	//which should be implemented in derived class
    Deng::Col_vector<arma::Mat<Field>> iH_and_partial_H = Dynamics(t);

	//for the U, it's just normal time evolution
    k[0] = iH_and_partial_H[0]*U[0];

    for(unsigned int i = 1; i <= _dim_para; ++i)
    {
		//std::cout << iH_and_partial_H[i];

		//central equation of GOAT
		//time derivative of partial U partial alpha
		k[i] = iH_and_partial_H[i]*U[0] + iH_and_partial_H[0]*U[i];
    }

	////for debug use when encounter NaN or Inf
	//{
	//	int count_nan = 0;
	//	for (int i = 0; i < k.dimension(); ++i)
	//	{
	//		if (k[i].has_inf())
	//		{
	//			std::cout << i << " th component of k has inf at " << t_index << " th step" << std::endl;
	//			++count_nan;
	//		}
	//	}
	//	assert(count_nan == 0 && "Derivative of Hamiltonian generates NaN\n");
	//}

    return k;
}
//called when we only need to evolve U
//i.e. we only need to know the value of the target fucntion
//rather than the value and derivatives
template <typename Field, typename Parameter>
arma::Mat<Field> Hamiltonian<Field, Parameter>::Derivative_U(const arma::Mat<Field>& position, const unsigned int t_index, const bool half_time) const
{
	Parameter shift = half_time ? 0.5 : 0.0;//caution: 0 or 1???
	//need to test the data type
	Parameter t = (t_index + shift)*_dt;

	arma::Mat<Field> iH = Dynamics_U(t);

	return iH*position;
}




//realization of Runge Kutta 4th order method
//specialized for GOAT
//could be used for general matrix evolution after some adaption
template class Deng::GOAT::RK4<float , float >;
template class Deng::GOAT::RK4<double, double>;
template class Deng::GOAT::RK4<std::complex<float> , float >;
template class Deng::GOAT::RK4<std::complex<double>, double>;

template <typename Field, typename Parameter>
void RK4<Field, Parameter>::Prep_for_H(const Deng::GOAT::Hamiltonian<Field, Parameter> &H)
{
    //make sure the dimension fits
    current_state.set_size(H._dim_para + 1);
    next_state.set_size(H._dim_para + 1);

	//initial state of U is identity
    current_state[0].eye(H._N, H._N);
    next_state[0].eye(H._N, H._N);

    for(unsigned int i = 1; i <= H._dim_para; ++i)
    {
        //initial states of partial U partial alpha is 0
		current_state[i].zeros(H._N, H._N);
        next_state[i].zeros(H._N, H._N);
    }
}
//core equation of Runge Kutta
template <typename Field, typename Parameter>
void RK4<Field, Parameter>::Evolve_one_step(const Deng::GOAT::Hamiltonian<Field, Parameter> &H, const unsigned int t_index)
{
	//one should avoid using auto for armadillo
	//especially for return time of the build-in arma function
	auto k1 = H.Derivative(current_state               , t_index    , false);

	auto k2 = H.Derivative(current_state + 0.5*H._dt*k1, t_index    , true );

	auto k3 = H.Derivative(current_state + 0.5*H._dt*k2, t_index    , true );

	auto k4 = H.Derivative(current_state +     H._dt*k3, t_index + 1, false);

	//armadillo will throw runtime error if dimention does not fit here
    next_state = current_state + (H._dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
}
template <typename Field, typename Parameter>
void RK4<Field, Parameter>::Evolve_to_final(const Deng::GOAT::Hamiltonian<Field, Parameter> &H)
{
	for(unsigned int i = 0; i < H._N_t; ++i)
    {
        Evolve_one_step(H, i);
        current_state = next_state;
    }
}
//evolve U only
//for performance when calculating function value only
template <typename Field, typename Parameter>
void RK4<Field, Parameter>::Prep_for_H_U(const Deng::GOAT::Hamiltonian<Field, Parameter> &H)
{
	Prep_for_H(H);
}
template <typename Field, typename Parameter>
void RK4<Field, Parameter>::Evolve_one_step_U(const Deng::GOAT::Hamiltonian<Field, Parameter> &H, const unsigned int t_index)
{
	auto k1 = H.Derivative_U(current_state[0], t_index, false);

	auto k2 = H.Derivative_U(current_state[0] + 0.5*H._dt*k1, t_index, true);

	auto k3 = H.Derivative_U(current_state[0] + 0.5*H._dt*k2, t_index, true);

	auto k4 = H.Derivative_U(current_state[0] + H._dt*k3, t_index + 1, false);
	//armadillo will throw runtime error if dimention does not fit here
	next_state[0] = current_state[0] + (H._dt / 6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
}
template <typename Field, typename Parameter>
void RK4<Field, Parameter>::Evolve_to_final_U(const Deng::GOAT::Hamiltonian<Field, Parameter> &H)
{
	for (unsigned int i = 0; i < H._N_t; ++i)
	{
		Evolve_one_step(H, i);
		current_state[0] = next_state[0];
	}
}



//realization of Deng::GOAT::GOAT_Target_1st_order
//target function gives the overlap of target U and U under certain control
//the phase must be exact
template class Deng::GOAT::GOAT_Target_1st_order<std::complex<float>, float >;
template class Deng::GOAT::GOAT_Target_1st_order<std::complex<double>, double>;

//give the parameter (coordinate_given)
//calculate the function value
//desired final states are with specific phase
template<typename Field, typename Parameter>
Parameter GOAT_Target_1st_order<Field, Parameter>::function_value(const arma::Col<Parameter>& coordinate_given) const
{
	//prepare for time evolution
	this->H_and_partial_H_pt->Update_parameters(coordinate_given);
	this->RK_pt->Prep_for_H_U(*H_and_partial_H_pt);
	this->RK_pt->Evolve_to_final_U(*H_and_partial_H_pt);

	//Parameter value;
	//negative_gradient(coordinate_given, value);
	////std::cout << value << std::endl;
	Field trace_of_unitary = arma::trace(unitary_goal.t()*RK_pt->next_state[0]);
	//negative sign here since we what to minimize the function
	return -trace_of_unitary.real();
}
//give the parameter (coordinate_given)
//calculate the negative gradient (of the function which should be minimize)
template<typename Field, typename Parameter>
arma::Col<Parameter> GOAT_Target_1st_order<Field, Parameter>::negative_gradient(const arma::Col<Parameter>& coordinate_given, Parameter &function_value) const
{
	//prepare for time evolution
	H_and_partial_H_pt->Update_parameters(coordinate_given);
	RK_pt->Prep_for_H(*H_and_partial_H_pt);
	RK_pt->Evolve_to_final(*H_and_partial_H_pt);

	//give function value
	Field trace_of_unitary = arma::trace(unitary_goal.t()*RK_pt->next_state[0]);
	Field g_phase_factor = trace_of_unitary;
	g_phase_factor = std::conj(g_phase_factor) / std::abs(g_phase_factor);
	function_value = -trace_of_unitary.real();

	//calculate gradient
	//first determine the dimension and set to 0
	arma::Col<Parameter> gradient = coordinate_given;
	gradient.zeros();
	for (unsigned int i = 0; i < gradient.n_elem; ++i)
	{
		//derivative of the trace turns out to be simple as target U is constant
		trace_of_unitary = -arma::trace(unitary_goal.t()*RK_pt->next_state[i + 1]);
		gradient[i] = trace_of_unitary.real();
	}
	return -gradient;
}



//realization of Deng::GOAT::GOAT_Target_1st_order
//target function gives the overlap of target U up to a phase factor and U under certain control
//the phase could be arbitrary
template class Deng::GOAT::GOAT_Target_1st_order_no_phase<std::complex<float>, float >;
template class Deng::GOAT::GOAT_Target_1st_order_no_phase<std::complex<double>, double>;

//give the parameter (coordinate_given)
//calculate the function value
//desired final states WITHOUT specific phase
template<typename Field, typename Parameter>
Parameter GOAT_Target_1st_order_no_phase<Field, Parameter>::function_value(const arma::Col<Parameter>& coordinate_given) const
{
	//prepare for time evolution
	//this-> is necessary
	//the compiler does not look in dependent base classes when looking up nondependent names
	this->H_and_partial_H_pt->Update_parameters(coordinate_given);
	this->RK_pt->Prep_for_H_U(*(this->H_and_partial_H_pt));
	this->RK_pt->Evolve_to_final_U(*(this->H_and_partial_H_pt));

	//take out the diagonal elements
	//with the initial states as the basis
	arma::Col<Field> UU_diag = arma::diagvec(initial_states.t() * (this->unitary_goal.t())*(this->RK_pt->next_state[0]) * initial_states);

	//for debug
	//std::cout << arma::as_scalar(UU_diag.t()*UU_diag) << std::endl;
	
	//take modulos and sum over
	//similarly, negative sign for minimizing purpose
	return -arma::as_scalar(UU_diag.t()*UU_diag).real();
}
template<typename Field, typename Parameter>
arma::Col<Parameter> GOAT_Target_1st_order_no_phase<Field, Parameter>::negative_gradient(const arma::Col<Parameter>& coordinate_given, Parameter &function_value) const
{
	//prepare for time evolution
	//this-> is necessary
	//the compiler does not look in dependent base classes when looking up nondependent names
	this->H_and_partial_H_pt->Update_parameters(coordinate_given);
	this->RK_pt->Prep_for_H(*(this->H_and_partial_H_pt));
	this->RK_pt->Evolve_to_final(*(this->H_and_partial_H_pt));

	//give function value
	//DO NOT USE auto here
	//complex temporary obj are used here which will cause error
	arma::Col<Field> UU_diag = arma::diagvec(initial_states.t() * (this->unitary_goal.t())*(this->RK_pt->next_state[0]) * initial_states);

	Field trace_of_UUUU = arma::as_scalar(UU_diag.t()*UU_diag).real();
	function_value = -trace_of_UUUU.real();

	////check for nan. Debug use
	//if (function_value != function_value)
	//{
	//	std::cout << UU_diag << std::endl;
	//	std::cout << initial_states << std::endl;
	//	std::cout << unitary_goal << std::endl;
	//	std::cout << RK_pt->next_state[0] << std::endl;
	//	assert(false && "GOAT_RK4 generates NaN!\n");
	//}

	//calculate gradient
	//first determine the dimension and set to 0
	arma::Col<Parameter> gradient = coordinate_given;
	gradient.zeros();

	for (unsigned int i = 0; i < gradient.n_elem; ++i)
	{
		arma::Col<Field> UpartialU_diag = -arma::diagvec(initial_states.t() * (this->unitary_goal.t())*(this->RK_pt->next_state[i+1]) * initial_states);
		gradient[i] = 2.0*arma::as_scalar(UU_diag.t()*UpartialU_diag).real();
	}

	return -gradient;
}

