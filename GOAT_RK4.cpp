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
    parameters.set_size(_dim_para);
    for(unsigned int i = 0; i < _dim_para; ++i)
    {
        parameters[i] = 0;
    }
}
template <typename Field, typename Parameter>
Deng::Col_vector<arma::Mat<Field>> Hamiltonian<Field, Parameter>::Derivative(Deng::Col_vector<arma::Mat<Field>> U, unsigned int t_index, bool half_time) const
{
    Parameter shift = half_time ? 0.5 : 0.0;
    // if(half_time)
        // shift = 0.5;
    // else
        // shift = 0;//0 or 1???
	
	//need to test the data type
    Parameter t = (t_index + shift)*_dt;
	
	//+1 for the additional U with original time evolution
    Deng::Col_vector<arma::Mat<Field>> k(_dim_para + 1);

    Deng::Col_vector<arma::Mat<Field>> iH_and_partial_H = Dynamics(t);
    
    k[0] = iH_and_partial_H[0]*U[0];

    for(unsigned int i = 1; i <= _dim_para; ++i)
    {
		//std::cout << iH_and_partial_H[i];
		k[i] = iH_and_partial_H[i]*U[0] + iH_and_partial_H[0]*U[i];
    }

    return k;
}
template <typename Field, typename Parameter>
arma::Mat<Field> Hamiltonian<Field, Parameter>::Derivative_U(arma::Mat<Field> position, unsigned int t_index, bool half_time) const
{
	Parameter shift = half_time ? 0.5 : 0.0;
	// if(half_time)
	// shift = 0.5;
	// else
	// shift = 0;//0 or 1???

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
    current_state[0].eye(H._N, H._N);
    next_state[0].eye(H._N, H._N);
    for(unsigned int i = 1; i <= H._dim_para; ++i)
    {
        current_state[i].zeros(H._N, H._N);
        next_state[i].zeros(H._N, H._N);
    }
}
template <typename Field, typename Parameter>
void RK4<Field, Parameter>::Evolve_one_step(const Deng::GOAT::Hamiltonian<Field, Parameter> &H, const unsigned int t_index)
{
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
		Evolve_one_step_U(H, i);
		current_state[0] = next_state[0];
	}
}



//realization of Deng::GOAT::GOAT_Target_1st_order
//target function gives the overlap of target U and U under certain control
//the phase must be exact
template class Deng::GOAT::GOAT_Target_1st_order<std::complex<float>, float >;
template class Deng::GOAT::GOAT_Target_1st_order<std::complex<double>, double>;

template<typename Field, typename Parameter>
Parameter GOAT_Target_1st_order<Field, Parameter>::function_value(const arma::Col<Parameter>& coordinate_given)
{
	H_and_partial_H_pt->Update_parameters(coordinate_given);
	RK_pt->Prep_for_H_U(*H_and_partial_H_pt);
	RK_pt->Evolve_to_final_U(*H_and_partial_H_pt);
	
	//Parameter value;
	//negative_gradient(coordinate_given, value);
	////std::cout << value << std::endl;
	Field trace_of_unitary = arma::trace(unitary_goal.t()*RK_pt->next_state[0]);
	return -trace_of_unitary.real();

}
template<typename Field, typename Parameter>
arma::Col<Parameter> GOAT_Target_1st_order<Field, Parameter>::negative_gradient(const arma::Col<Parameter>& coordinate_given, Parameter &function_value)
{
	H_and_partial_H_pt->Update_parameters(coordinate_given);
	RK_pt->Prep_for_H(*H_and_partial_H_pt);
	RK_pt->Evolve_to_final(*H_and_partial_H_pt);
	//give function value
	Field trace_of_unitary = arma::trace(unitary_goal.t()*RK_pt->next_state[0]);
	Field g_phase_factor = trace_of_unitary;
	g_phase_factor = std::conj(g_phase_factor) / std::abs(g_phase_factor);
	function_value = -trace_of_unitary.real();

	//calc gradient
	arma::Col<Parameter> gradient = coordinate_given;
	gradient.zeros();
	//std::cout << RK_pt->next_state[0];
	//std::cout << -arma::trace(unitary_goal.t()*RK_pt->next_state[0]) << std::endl;

	for (unsigned int i = 0; i < gradient.n_elem; ++i)
	{
		trace_of_unitary = -arma::trace(unitary_goal.t()*RK_pt->next_state[i + 1]);
		//trace_of_unitary = g_phase_factor*trace_of_unitary/(double)gradient.n_elem;
		gradient[i] = trace_of_unitary.real();
	}

	return -gradient;
}


//realization of Deng::GOAT::GOAT_Target_1st_order
//target function gives the overlap of target U up to a phase factor and U under certain control
//the phase could be arbitrary
template class Deng::GOAT::GOAT_Target_1st_order_no_phase<std::complex<float>, float >;
template class Deng::GOAT::GOAT_Target_1st_order_no_phase<std::complex<double>, double>;

template<typename Field, typename Parameter>
Parameter GOAT_Target_1st_order_no_phase<Field, Parameter>::function_value(const arma::Col<Parameter>& coordinate_given)
{
	H_and_partial_H_pt->Update_parameters(coordinate_given);
	RK_pt->Prep_for_H_U(*H_and_partial_H_pt);
	RK_pt->Evolve_to_final_U(*H_and_partial_H_pt);

	//Parameter value;
	//negative_gradient(coordinate_given, value);
	////std::cout << value << std::endl;
	arma::trace(unitary_goal.t()*RK_pt->next_state[0]);
	auto UU_diag = arma::diagvec(unitary_goal.t()*RK_pt->next_state[0]);
	
	//std::cout << arma::as_scalar(UU_diag.t()*UU_diag) << std::endl;
	return -arma::as_scalar(UU_diag.t()*UU_diag).real();

}
template<typename Field, typename Parameter>
arma::Col<Parameter> GOAT_Target_1st_order_no_phase<Field, Parameter>::negative_gradient(const arma::Col<Parameter>& coordinate_given, Parameter &function_value)
{
	H_and_partial_H_pt->Update_parameters(coordinate_given);
	RK_pt->Prep_for_H(*H_and_partial_H_pt);
	RK_pt->Evolve_to_final(*H_and_partial_H_pt);
	
	//give function value
	//DO NOT USE auto here. It causes error. Reason is unknown
	arma::Col<Field> UU_diag = arma::diagvec(unitary_goal.t()*RK_pt->next_state[0]);
	Field trace_of_UUUU = arma::as_scalar(UU_diag.t()*UU_diag).real();
	function_value = -trace_of_UUUU.real();
	
	//calc gradient
	arma::Col<Parameter> gradient = coordinate_given;
	gradient.zeros();

	for (unsigned int i = 0; i < gradient.n_elem; ++i)
	{
		arma::Col<Field> UpartialU_diag = -arma::diagvec(unitary_goal.t()*RK_pt->next_state[i+1]);
		gradient[i] = 2.0*arma::as_scalar(UU_diag.t()*UpartialU_diag).real();
	}

	return -gradient;
}