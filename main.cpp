#include <iostream>
#include <cmath>
#include <ctime>


#include <armadillo>
#include "Optimization.hpp"
#include "GOAT_RK4.hpp"
#include "Deng_vector.hpp"


#include "transverse_Ising.hpp"


void Verify_level_crossing()
{
	//    const int N_t = 2000;
	//    const double tau = 3.0;
	//    const int dim_para = std::stoi(argv[1]);
	const double hbar = 1.0;
	//
	//    const double epsilon = 0.01;
	//    const int max_iteration = 20;
	//
	//    Single_spin_half H_only(N_t, tau, 0);
	//    Single_spin_half H_and_partial_H(N_t, tau, dim_para);
	//    Deng::GOAT::RK4<std::complex<double> > RungeKutta;
	//
	//    //GOAT_Target target(&RungeKutta, &H_only, &H_and_partial_H);
	//	GOAT_Target target(&RungeKutta, &H_and_partial_H);

	Deng::Col_vector<arma::Mat<elementtype>> S(3);
	S[0].zeros(2, 2);
	S[0](0, 1) = 1;
	S[0](1, 0) = 1;
	S[1].zeros(2, 2);
	S[1](0, 1) = -imag_i;
	S[1](1, 0) = imag_i;
	S[2].zeros(2, 2);
	S[2](0, 0) = 1;
	S[2](1, 1) = -1;
	arma::Mat<elementtype> S_identity;
	S_identity.zeros(2, 2);
	S_identity(0, 0) = 1;
	S_identity(1, 1) = 1;
	//S = 0.5*hbar*S;

	const unsigned int N_t = 1000;
	const double tau = 3.0;

	const unsigned int num_spinor = 6;
	const unsigned int dim_hamil = 1 << num_spinor;

	//output to 
	std::ofstream outputfile;
	outputfile.open("eigenvalues.dat");

	Deng::Col_vector<real> B(3);
	B[0] = 1.5;
	B[1] = 0.2;
	B[2] = 1.0;

	Deng::Col_vector<real> B_now = B;

	Transverse_Ising H(num_spinor, N_t, tau, 9);

	for (unsigned int t_i = 0; t_i <= N_t; ++t_i)
	{
		arma::Col<real> eigen_energy(dim_hamil);
		auto current_H = H.H_0((tau*(double)t_i) / N_t);
		auto eigen_vector = current_H;

		eigen_vector.zeros();
		arma::eig_sym(eigen_energy, eigen_vector, current_H, "std");

		if ((!current_H.has_nan()) && eigen_vector.has_nan())
		{
			std::cout << arma::accu(arma::abs(current_H - current_H.t())) << "    ";
			std::cout << t_i << "th step eigen vectors has nan or inf!" << std::endl;
			std::cout << eigen_vector.col(58).has_nan() << eigen_vector.col(59).has_nan() << eigen_vector.col(60).has_nan() << std::endl;
		}

		//if (t_i < 100)
		//{
		//	std::cout << "with prev: " << arma::as_scalar(eigen_vector.col(59).t()*eigen_vector.col(58)) << "  ";
		//	std::cout << "with next: " << arma::as_scalar(eigen_vector.col(59).t()*eigen_vector.col(60)) << "  ";
		//	std::cout << "with self: " << arma::as_scalar(eigen_vector.col(59).t()*eigen_vector.col(58)) << "  ";
		//	std::cout << std::endl;
		//}

		outputfile << eigen_energy.t();
	}

	//for (unsigned int t_i = 0; t_i <= N_t; ++t_i)
	//{
	//	arma::Mat<elementtype> interaction(dim_hamil, dim_hamil, arma::fill::zeros);
	//	auto external_field = interaction;

	//	//B_now = ((2 * (double)t_i) / N_t - 1.0)*B;
	//	B_now[0] = B[0] * cos(0.5*Pi*(double)t_i / N_t);
	//	B_now[1] = 0.0;
	//	B_now[2] = B[2] * sin(0.5*Pi*(double)t_i / N_t);


	//	for (unsigned int i = 0; i < (num_spinor - 1); ++i)
	//	{
	//		auto temp = i == 0 ? S[2] : S_identity;
	//		for (unsigned int j = 1; j < num_spinor; ++j)
	//		{
	//			temp = arma::kron(temp, (j == i) || (j == i + 1) ? S[2] : S_identity);
	//		}
	//		interaction += temp;
	//	}

	//	//auto temp = S[2];
	//	//for (unsigned int j = 1; j < num_spinor; ++j)
	//	//{
	//	//	temp = arma::kron(temp, j == (num_spinor - 1) ? S[2] : S_identity);
	//	//}
	//	//interaction += temp;

	//	for (unsigned int i = 0; i < num_spinor; ++i)
	//	{
	//		auto temp_B = i == 0 ? B_now^S : S_identity;
	//		for (unsigned int j = 1; j < num_spinor; ++j)
	//		{
	//			temp_B = arma::kron(temp_B, i == j ? B_now^S : S_identity);
	//		}
	//		external_field += temp_B;
	//	}

	//	arma::Col<real> eigen_energy(dim_hamil);
	//	auto eigen_vector = interaction;
	//	arma::eig_sym(eigen_energy, eigen_vector, interaction + external_field);

	//	//std::cout  << eigen_energy.t() << std::endl;
	//	outputfile << eigen_energy.t();// << std::endl;

	//}

	//std::cout << interaction << std::endl;
	//std::cout << arma::kron(S[0], S[1]) << std::endl;

	outputfile.close();
}

int main(int argc, char** argv)
{
	//Verify_level_crossing();
	const unsigned int num_spinor = std::stoi(argv[1]);
	const unsigned int dim_hamil = 1 << num_spinor;
	const unsigned int dim_para = std::stoi(argv[2]);

	std::cout << num_spinor << ", " << dim_para << std::endl;
	
	const unsigned int N_t = 1000;
	const double tau = 1.0;

	Transverse_Ising H(num_spinor, N_t, tau, dim_para);
	Deng::GOAT::RK4<elementtype, real> RK;
	Deng::GOAT::GOAT_Target_1st_order<elementtype, real> target(&RK, &H);
	//Deng::Optimization::

	arma::Mat<elementtype> unitary_goal(dim_hamil, dim_hamil, arma::fill::zeros);
	{
		arma::Col<real> eigval_0;
		auto H_0 = H.H_0(0);
		auto eigvec_0 = H_0;
		arma::eig_sym(eigval_0, eigvec_0, H_0);

		arma::Col<real> eigval_tau;
		auto H_tau = H.H_0(tau);
		auto eigvec_tau = H_tau;
		arma::eig_sym(eigval_tau, eigvec_tau, H_tau);

		for (unsigned int i = 0; i < dim_hamil; ++i)
			unitary_goal += eigvec_tau.col(i)*eigvec_0.col(i).t();
	}

	target.Set_Controlled_Unitary_Matrix(unitary_goal);

	Deng::Optimization::Min_Conj_Grad<real> Conj_Grad(dim_para, 1E-3, 100);

	Conj_Grad.Assign_Target_Function(&target);
	Conj_Grad.Opt_1D = Deng::Optimization::OneD_Golden_Search<real>;
	arma::arma_rng::set_seed(time(nullptr));
	//arma::Col<real> start(dim_para, arma::fill::randu);
	//start = start - 0.5;
	//std::cout << start.t() << std::endl;
	arma::Col<real> start(dim_para, arma::fill::zeros);
	start = Conj_Grad.Conj_Grad_Search(start);


//    arma::Col<double> eigval_0;
//    arma::Mat<std::complex<double> > eigvec_0;
//    arma::Mat<std::complex<double> > H_0 = H_and_partial_H.B(0)^ H_and_partial_H.S;
//    arma::eig_sym(eigval_0  , eigvec_0  , H_0  );
//
//    arma::Col<double> eigval_tau;
//    arma::Mat<std::complex<double> > eigvec_tau;
//    arma::Mat<std::complex<double> >H_tau = H_and_partial_H.B(tau)^ H_and_partial_H.S;
//    arma::eig_sym(eigval_tau, eigvec_tau, H_tau);
//
//    arma::Mat<std::complex<double> > unitary_goal = eigvec_0;
//    unitary_goal.zeros();
//
//	std::cout << eigvec_0 << eigvec_tau << std::endl;
//
//    unitary_goal += eigvec_tau.col(0)*eigvec_0.col(0).t()*exp(-2.59742*imag_i);
//    unitary_goal += eigvec_tau.col(1)*eigvec_0.col(1).t()*exp(-0.544176*imag_i);
//
//    target.Set_Controlled_Unitary_Matrix(unitary_goal);
//	std::cout << unitary_goal << std::endl;

    //double aa;
    //arma::Col<double> position(dim_para, arma::fill::zeros);
    //std::cout << target.negative_gradient(position, aa) << std::endl;

//    Deng::Optimization::Min_Conj_Grad<double> Conj_Grad(dim_para, epsilon, max_iteration);
//
//    Conj_Grad.Assign_Target_Function(&target);
//    Conj_Grad.Opt_1D = Deng::Optimization::OneD_Golden_Search<double>;
//	arma::arma_rng::set_seed(time(nullptr));
//	arma::Col<double> start(dim_para, arma::fill::randu);
//	start = start - 1;
//    start = Conj_Grad.Conj_Grad_Search(start);

    return 0;
}
