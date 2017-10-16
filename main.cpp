#include <iostream>

#include "Optimization.hpp"
#include "GOAT_RK4.hpp"
#include "Deng_vector.hpp"


//mathematical constants
#ifndef DENG_PI_DEFINED
#define DENG_PI_DEFINED
const double Pi = 3.14159265358979324;
const double Pi_sqrt = sqrt(Pi);
const double Pi_fourth = sqrt(Pi_sqrt);
const std::complex<double> imag_i(0, 1.0);
#endif


//Berry's transition-less driving
class Single_spin_half : public Deng::GOAT::Hamiltonian<std::complex<double> >
{
    friend class Deng::GOAT::RK4<std::complex<double> >;
public:
    virtual Deng::Col_vector<arma::Mat<std::complex<double> > > Dynamics(double t) const override;
    //Pauli matrix, magnetic field and its partial derivative wrt time
    Deng::Col_vector<arma::Mat<std::complex<double> > > S;
    Deng::Col_vector<double> B(double t) const;
    Deng::Col_vector<double> partialB(double t) const;
    //control magnetic field
    Deng::Col_vector<double> control_field(double t) const;
    //inherit constructor
    using Deng::GOAT::Hamiltonian<std::complex<double> >::Hamiltonian;
    Single_spin_half(int N_t, double tau, int dim_para = 0, double hbar = 1.0);
    const double B_x_max = 1.0;
    const double B_y_max = 1.0;
    const double B_z_max = 2.0;
    const double omega;
    void Update_control_field(arma::Col<double> parameters);
protected:
    double _hbar;
};
Single_spin_half::Single_spin_half(int N_t, double tau, int dim_para, double hbar) :
    Deng::GOAT::Hamiltonian<std::complex<double> > (2, N_t, tau, dim_para), omega(2.0*Pi/tau), _hbar(hbar)
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
Deng::Col_vector<double> Single_spin_half::B (double t) const
{
    Deng::Col_vector<double> B_field(3);

    //rotating B field
    B_field[0] = cos(omega*t)*B_x_max;
    B_field[1] = sin(omega*t)*B_y_max;
    B_field[2] = B_z_max;

    return B_field;
}
Deng::Col_vector<double> Single_spin_half::control_field(double t) const
{

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

    Deng::Col_vector<double> ctrl(3);

    ctrl[2] = parameters[0];

    for(int i = 1; i < _dim_para; ++i)
    {
        int mode = (i+1)/2;
        ctrl[i%2] = parameters[i] * sin(mode*2*Pi*t/_tau);
    }

    return ctrl;
}
Deng::Col_vector<arma::Mat<std::complex<double> > > Single_spin_half::Dynamics(double t) const
{
    Deng::Col_vector<arma::Mat<std::complex<double> > > iH_and_partial_H(_dim_para + 1);

    iH_and_partial_H[0] = (-imag_i/_hbar)*((B(t) + control_field(t))^S);

    return iH_and_partial_H;
}



typedef double real;
class GOAT_Target : public Deng::Optimization::Target_function<real>
{
public:
    virtual real function_value(const arma::Col<real>& coordinate_given) override;
    virtual arma::Col<real> negative_gradient(const arma::Col<real>& coordinate_given, real &function_value) override;
    virtual arma::Mat<real> Hessian(const arma::Col<real>& coordinate_given, real &function_value, arma::Col<real> &negative_gradient)
    {
        assert(false && "Hessian is not known!");
        return 0;
    };
    virtual void higher_order(const arma::Col<real>& coordinate_given, const real order)
    {
        assert(false && "Higher order derivatives are not known!");
    };
    Deng::GOAT::RK4<std::complex<real> > *RK_pt;
    Deng::GOAT::Hamiltonian<std::complex<real> > *H_only_pt;
    Deng::GOAT::Hamiltonian<std::complex<real> > *H_and_partial_H_pt;

    GOAT_Target(Deng::GOAT::RK4<std::complex<real> > *input_RK_pt, Deng::GOAT::Hamiltonian<std::complex<real> > *input_H_only_pt, Deng::GOAT::Hamiltonian<std::complex<real> > *input_H_and_partial_H_pt)
    {
        RK_pt = input_RK_pt;
        H_only_pt = input_H_only_pt;
        H_and_partial_H_pt = input_H_and_partial_H_pt;
    }
    virtual void Set_Controlled_Unitary_Matrix(const arma::Mat<std::complex<double> > &matrix_desired)
    {
        unitary_goal = matrix_desired;
    };
    arma::Mat<std::complex<double> > unitary_goal;

};
real GOAT_Target::function_value(const arma::Col<real>& coordinate_given)
{
    H_only_pt->parameters = coordinate_given;

    RK_pt->Prep_for_H(*H_only_pt);
    RK_pt->Evolve_to_final(*H_only_pt);

    std::complex<double> trace_of_unitary = arma::trace(unitary_goal.t()*RK_pt->next_state[0]);

    //we want to know the maximum of the trace
    return -trace_of_unitary.real();

}
arma::Col<real> GOAT_Target::negative_gradient(const arma::Col<real>& coordinate_given, real &function_value)
{
    H_and_partial_H_pt->parameters = coordinate_given;
    std::cout << "grad_error?" << std::endl;
    RK_pt->Prep_for_H(*H_and_partial_H_pt);
    RK_pt->Evolve_to_final(*H_and_partial_H_pt);
    std::cout << "grad_error2?" << std::endl;
    //give function value
    std::complex<double> trace_of_unitary = arma::trace(unitary_goal.t()*RK_pt->next_state[0]);
    std::complex<double> g_phase_factor = trace_of_unitary;
    g_phase_factor = std::conj(g_phase_factor)/sqrt(pow(g_phase_factor.real(),2.0) + pow(g_phase_factor.imag(),2.0));
    function_value = -trace_of_unitary.real();

    //calc gradient
    arma::Col<real> gradient = coordinate_given;
    gradient.zeros();

    for(unsigned int i = 0; i < gradient.n_elem; ++i)
    {
        trace_of_unitary = arma::trace(unitary_goal.t()*RK_pt->next_state[i+1]);
        trace_of_unitary = g_phase_factor*trace_of_unitary/(double)gradient.n_elem;

        gradient[i] = trace_of_unitary.real();
    }

    return -gradient;
}




int main()
{
    const int N_t = 4000;
    const double tau = 3.0;
    const int dim_para = 5;
    //const double hbar = 1.0;

    const double epsilon = 0.001;
    const int max_iteration = 50;

    Single_spin_half H_only(N_t, tau, 0);
    Single_spin_half H_and_partial_H(N_t, tau, dim_para);
    Deng::GOAT::RK4<std::complex<double> > RungeKutta;

    GOAT_Target target(&RungeKutta, &H_only, &H_and_partial_H);



    arma::Col<double> eigval_0;
    arma::Mat<std::complex<double> > eigvec_0;
    arma::Mat<std::complex<double> > H_0 = H_only.B(0)^H_only.S;
    arma::eig_sym(eigval_0  , eigvec_0  , H_0  );

    arma::Col<double> eigval_tau;
    arma::Mat<std::complex<double> > eigvec_tau;
    arma::Mat<std::complex<double> >H_tau = H_only.B(tau)^H_only.S;
    arma::eig_sym(eigval_tau, eigvec_tau, H_tau);

    arma::Mat<std::complex<double> > unitary_goal = eigvec_0;
    unitary_goal.zeros();

    unitary_goal += eigvec_tau.col(0)*eigvec_0.col(0).t();
    unitary_goal += eigvec_tau.col(1)*eigvec_0.col(1).t();
    std::cout << "here2?" << std::endl;
    target.Set_Controlled_Unitary_Matrix(unitary_goal);

    double aa;
    arma::Col<double> position(dim_para, arma::fill::zeros);
    std::cout << target.negative_gradient(position, aa) << std::endl;


    assert(false);

    Deng::Optimization::Min_Conj_Grad<double> Conj_Grad(dim_para, epsilon, max_iteration);


    Conj_Grad.Assign_Target_Function(&target);
    Conj_Grad.Opt_1D = Deng::Optimization::OneD_Golden_Search<double>;

    //std::cout << "here?" << std::endl;
    std::cout << Conj_Grad.Conj_Grad_Search();

    //std::cout << "End\n";

    return 0;
}
