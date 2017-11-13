#include <iostream>
#include <cmath>
#include <ctime>


#include <armadillo>
#include "Optimization.hpp"
#include "GOAT_RK4.hpp"
#include "Deng_vector.hpp"

#define ISING_CONTROL_INDIVIDUAL
#include "transverse_Ising.hpp"

void Verify_level_crossing();

typedef double real;
typedef std::complex<real> elementtype;

int main(int argc, char** argv)
{
	//Verify_level_crossing();
	const unsigned int num_spinor = std::stoi(argv[1]);
	const unsigned int dim_hamil = 1 << num_spinor;
	const unsigned int dim_para_each_direction = std::stoi(argv[2]);
	//const unsigned int dim_para = (2 * num_spinor + 1)*dim_para_each_direction;
	const unsigned int dim_para = (2 * num_spinor +1)*dim_para_each_direction;
	const unsigned int rand = std::stoi(argv[3]);

	std::cout << "Number of spin: " << num_spinor << std::endl;
	std::cout << "Dim of Paramateric space: " << dim_para << "  with " << dim_para_each_direction <<" paras for each direction" << std::endl;
	std::cout << "start with " << rand << " randomness" << std::endl << std::endl << std::endl;
	
	const unsigned int N_t = 2048;
	const real tau = 3.0;
	const real epsilon = 1.0/1024;
	const real epsilon_gradient = dim_hamil* epsilon;
	const unsigned int conj_grad_max_iter = 100;


	//Transverse_Ising<elementtype, real> H(num_spinor, N_t, tau, dim_para, dim_para/3);
	Transverse_Ising_Local_Control<elementtype, real> H(num_spinor, N_t, tau, dim_para, dim_para / (2 * num_spinor + 1));
	Deng::GOAT::RK4<elementtype, real> RK;

	//RK.Prep_for_H_U(H);
	//RK.Evolve_to_final_U(H);
	//std::cout << RK.next_state[0];
	//const std::complex<real> ttt( 1 + num_spinor*H.B_z_max - num_spinor, 0.0);
	//std::cout << std::exp(-ttt*imag_i);

	//set up target function
	//Deng::GOAT::GOAT_Target_1st_order<elementtype, real> target(&RK, &H);
	Deng::GOAT::GOAT_Target_1st_order_no_phase<elementtype, real> target(&RK, &H);

	//Deng::GOAT::GOAT_Target_2nd_order_no_phase<elementtype, real> target_finite_diff(&RK, &H);


	//determine the target unitary matrix
	arma::Mat<elementtype> unitary_goal(dim_hamil, dim_hamil, arma::fill::zeros);
	{
		arma::Col<real> eigval_0;
		arma::Mat<elementtype> H_0 = H.H_0(0);
		arma::Mat<elementtype> eigvec_0 = H_0;
		arma::eig_sym(eigval_0, eigvec_0, H_0, "std");
		
		//std::cout << H_0 << std::endl;
		//std::cout << eigval_0 << std::endl;
		//std::cout << eigvec_0 << std::endl;

		arma::Col<real> eigval_tau;
		arma::Mat<elementtype> H_tau = H.H_0(tau);
		arma::Mat<elementtype> eigvec_tau = H_tau;
		arma::eig_sym(eigval_tau, eigvec_tau, H_tau, "std");

		//std::cout << H_tau << std::endl;
		//std::cout << eigval_tau << std::endl;
		//std::cout << eigvec_tau << std::endl;

		for (unsigned int i = 0; i < dim_hamil; ++i)
		{
			unitary_goal += eigvec_tau.col(i)*eigvec_0.col(i).t();			
		}
		target.Set_Initial_States(eigvec_0);
		//target_finite_diff.Set_Initial_States(eigvec_0);
	}
	target.Set_Controlled_Unitary_Matrix(unitary_goal);
	//target_finite_diff.Set_Controlled_Unitary_Matrix(unitary_goal);

	//set up Conjugate gradient method for searching minimum
	//Deng::Optimization::Min_Conj_Grad<real> Conj_Grad(dim_para, epsilon, conj_grad_max_iter, epsilon_gradient);
	//Deng::Optimization::Newton_Find_Root<real> NT_search(dim_para, 0.25*dim_hamil, conj_grad_max_iter, epsilon_gradient);
	Deng::Optimization::Quasi_Newton<real> Quasi_NT(dim_para, epsilon, conj_grad_max_iter, epsilon_gradient);
	//appoint target function
	//Conj_Grad.Assign_Target_Function(&target);
	//NT_search.Assign_Target_Function(&target);
	//NT_search.Assign_Target_Function_Value(-(real)(dim_hamil));
	Quasi_NT.Assign_Target_Function(&target);
	//appoint the 1D search method
	//Conj_Grad.Opt_1D = Deng::Optimization::OneD_Golden_Search<real>;
	//Conj_Grad.Opt_1D = Deng::Optimization::My_1D_foward_method<real>;
	//generate initial coordinate to start
	arma::arma_rng::set_seed(time(nullptr));
	arma::Col<real> start(dim_para, arma::fill::randn);
	//how wide the initial coordinate we choose
	start = rand*start;

	Quasi_NT.BFGS(start);

	//Deng::Optimization::Newton_Find_Min<real> NT_search_hess(dim_para, 0.25*dim_hamil, conj_grad_max_iter, epsilon_gradient);
	//NT_search_hess.Assign_Target_Function(&target_finite_diff);
	//real a;
	//auto b = start;
	////std::cout << target_finite_diff.Hessian(start, a, b);
	//NT_search_hess.Newton_2nd_order(start);

	//std::cout << H.H_0(1.0) - H.H_0(0);
	//std::cout << H.S_total[0] << H.S_total[2];
	//std::cout << H.interaction;

	

	//H.Update_parameters(start);
	//RK.Prep_for_H_U(H);
	//RK.Evolve_to_final_U(H);
	//auto A = RK.next_state[0];
	//std::cout << A.t()*A << std::endl;

	//start = NT_search.Newton_1st_order(start);

	//real new_epsilon = 10 * epsilon;

	//do 
	//{
	//	Conj_Grad.Revise_epsilon(new_epsilon);
	//	NT_search.Revise_epsilon(new_epsilon);
	//	
	//	start = Conj_Grad.Conj_Grad_Search(start);
	//	start = NT_search.Newton_1st_order(start);

	//} while (target.function_value(start) - NT_search.Value_of_target_value() > new_epsilon);
	
	

	
	//bool is_stationary = true;

	//while (is_stationary)
	//{
	//	//redundant calculation...
	//	auto current_min_coordinate = Conj_Grad.Conj_Grad_Search(start);
	//	auto current_min_previous_search_direction = Conj_Grad.previous_search_direction;
	//	auto current_min = target.function_value(current_min_coordinate);
	//	
	//	
	//	is_stationary = false;
	//	//move the coordinate around
	//	//to see if it is only a stationary point
	//	start = current_min_coordinate;
	//	for (int i = 0; i < 2*dim_hamil; ++i)
	//	{
	//		//randomly pick a direction
	//		arma::Col<real> shift(dim_para, arma::fill::randu);
	//		auto temp_search_direction = Conj_Grad.previous_search_direction;
	//		//real scale = arma::as_scalar(temp_search_direction.t()*temp_search_direction);
	//		//shift = sqrt(scale)*shift;
	//		//shift = shift - arma::as_scalar(temp_search_direction.t()*shift) / scale * temp_search_direction;
	//		//real lambda = Deng::Optimization::OneD_Golden_Search<real>(start, temp_search_direction, &target, 200, epsilon);
	//		real lambda = Deng::Optimization::My_1D_foward_method<real>(start, temp_search_direction, &target, 200, epsilon);
	//		if (lambda<0)
	//			lambda = Deng::Optimization::OneD_Golden_Search<real>(start, -temp_search_direction, &target, 200, epsilon);

	//		real temp_func_value = target.function_value(current_min_coordinate + lambda*shift);

	//		if (temp_func_value < current_min)
	//		{
	//			//this minimum is only a stationary point
	//			is_stationary = true;
	//			start = current_min_coordinate + lambda*shift;
	//			std::cout << "\n\nNot a (local) minimum\n\n";

	//			break;
	//		}
	//	}

	//	if (!is_stationary)
	//	{
	//		std::cout << "\nReach (local) minimum" << std::endl;
	//	}	
	//}


    return 0;
}





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

	Transverse_Ising<elementtype, real> H(num_spinor, N_t, tau, 9, 2);

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