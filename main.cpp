#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <ctime>

#include <armadillo>
#include "Deng_vector.hpp"
#include "Optimization.hpp"
#include "GOAT_RK4.hpp"
#include "transverse_Ising.hpp"

#include "miscellaneous.hpp"


//used to print out energy levels without control field
//outputfile names ""eigenvalues.dat"
//should use python to do further plot
void Verify_level_crossing(const std::string);

typedef double real;
typedef std::complex<real> elementtype;

int main(int argc, char** argv)
{
	if (argc > 2)
	{
		Verify_level_crossing(argv[1]);

		return 0;
	}


	//read in arguments from file
	Deng::Misc::ReadArguments::FromFile myargs(argv[1]);
	//assign the values
	const unsigned int num_spinor = myargs("num_spinor");
	const unsigned int dim_hamil = 1 << num_spinor;
	const unsigned int dim_para_each_direction = myargs("dim_para_each_direction");
	//const unsigned int dim_para = (2 * num_spinor + 1)*dim_para_each_direction;
	const unsigned int dim_para = 3 * num_spinor * dim_para_each_direction;
	const double rand = myargs("rand");

	std::cout << "Number of spin: " << num_spinor << std::endl;
	std::cout << "Dim of Paramateric space: " << dim_para << "  with " << dim_para_each_direction << " paras for each direction" << std::endl;
	std::cout << "start with " << rand << " randomness" << std::endl << std::endl << std::endl;

	const unsigned int N_t = myargs("N_t");
	const double tau = myargs("tau");
	const double epsilon = myargs("epsilon");
	const double epsilon_gradient = sqrt(dim_hamil)*epsilon;
	const unsigned int conj_grad_max_iter = myargs("conj_grad_max_iter");


	//Transverse_Ising<elementtype, real> H(num_spinor, N_t, tau, dim_para, dim_para/3);
	//Transverse_Ising_Local_Control<elementtype, real> H(num_spinor, N_t, tau, dim_para, dim_para_each_direction);
	//Transverse_Ising_Impulse_Local<elementtype, real> H(num_spinor, N_t, tau, dim_para, dim_para_each_direction);
	LMG<elementtype, real> H(num_spinor, N_t, tau, dim_para, dim_para_each_direction);

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
	Quasi_NT.Set_Randomness(rand);

	//appoint target function
	//Conj_Grad.Assign_Target_Function(&target);
	//NT_search.Assign_Target_Function(&target);
	//NT_search.Assign_Target_Function_Value(-(real)(dim_hamil));
	Quasi_NT.Assign_Target_Function(&target);
	//appoint the 1D search method
	//Conj_Grad.Opt_1D = Deng::Optimization::OneD_Golden_Search<real>;
	//Conj_Grad.Opt_1D = Deng::Optimization::My_1D_foward_method<real>;


	//arma::Col<real> start_error = { 3.8956e+02, 2.5888e+02,	1.7711e+02,	1.1133e+02,	1.3407e+02,	3.1740e+02,	1.8245e+02,
	//								1.7379e+02,	3.8170e+02,	1.2061e+02,	2.3854e+02,	2.9739e+01,	2.0196e+02,	3.5960e+02 };

	//target.function_value(start_error);

	//output to file
	std::ofstream outputfile;
	std::string filename;
	//set file name
	filename = std::to_string(num_spinor) + "spins_" + std::to_string(dim_para) + "paras_t_dep_";
	filename = filename + Deng::Misc::TimeStamp() + ".dat";
	//set output & format
	outputfile.open(filename, std::ios_base::app);
	outputfile.precision(10);
	outputfile.setf(std::ios::fixed);
	//indicate system specs at the beginning
	myargs(outputfile);
	outputfile << "Number of spin: " << num_spinor << std::endl;
	outputfile << "Dim of Paramateric space: " << dim_para << "  with " << dim_para_each_direction << " paras for each direction" << std::endl;
	outputfile << "Take initial position with " << rand << " randomness" << std::endl << std::endl;
	outputfile << "Programme starts at " << Deng::Misc::TimeStamp() << std::endl << std::endl << std::endl;

	arma::Col<real> current_min_coordinate(dim_para);
	real current_min;
	//generate initial coordinate to start
	arma::arma_rng::set_seed(time(nullptr));
	arma::Col<real> start(dim_para, arma::fill::randu);
	//how wide the initial coordinate we choose
	start = rand*start;

	//start = { -6.6846 ,   0.7947 ,-1.8900, -6.1079 ,
	//	-1.1350, -1.0042 ,-1.2696  ,  1.4903   ,
	//	6.5364 ,   0.0558, -0.4509 ,   7.2567,
	//	-10.6137, -0.9592, -3.4418 ,   2.8441,
	//	-1.3137  ,  0.0816, -3.4504, -8.6243 ,
	//	-5.3629 ,-1.8439 ,-0.9611 ,   3.9956   ,
	//	1.7500   , 0.9611  ,  3.9424 ,-0.9289 };

	//start search
	bool is_stationary = false;
	bool is_global_min = true;
	do {
		is_global_min = true;
		/*
		do {
			//reached a position
			current_min_coordinate = Quasi_NT.BFGS(start);
			current_min = target.function_value(current_min_coordinate);

			if (current_min < -((real)dim_hamil - (1.0*epsilon)))
			{
				std::cout << "Good enough!\n";
				is_global_min = true;
				is_stationary = false;
			}
			else
			{
				//once reach the 0 negative gradient
				//randomly pick several directions to check if it's only a stationary point
				is_stationary = false;

				//set up random direction
				arma::Col<real> random_search_direction = current_min_coordinate;

				//number of trials should increases with dimention of parametric space
				for (unsigned int i = 0; i < dim_para; ++i)
				{
					//randomly pick a direction
					random_search_direction.randn();

					//go through any 1D search
					real lambda = Deng::Optimization::OneD_Golden_Search<real>(current_min_coordinate, random_search_direction, &target, 200, epsilon);
					if (lambda < 0)
					{
						random_search_direction = -random_search_direction;
						lambda = Deng::Optimization::OneD_Golden_Search<real>(current_min_coordinate, random_search_direction, &target, 200, epsilon);
					}

					//new function value, which should be <= the old one
					real temp_func_value = target.function_value(current_min_coordinate + lambda*random_search_direction);

					if (temp_func_value <= (current_min - epsilon / 256.0))
					{
						//this minimum is only a stationary point
						is_stationary = true;
						start = current_min_coordinate + lambda*random_search_direction;
						std::cout << "\n\nNot a (local) minimum\n\n";
						break;
					}
				}
				if (!is_stationary)
				{
					std::cout << "\nReach (local) minimum" << std::endl;
					outputfile << "Reach(local) minimum of " << current_min << "\n";
					//outputfile << current_min_coordinate.t() << "\n";
					current_min_coordinate.t().raw_print(outputfile);
				}
			}

		} while (is_stationary);
		*/

		current_min_coordinate = Quasi_NT.BFGS(start);
		current_min = target.function_value(current_min_coordinate);

		std::cout << "\nReach stationary point " << std::endl;
		outputfile << "Reach stationary point " << current_min << "\n";
		//outputfile << current_min_coordinate.t() << "\n";
		current_min_coordinate.t().raw_print(outputfile);
		
		auto temp = current_min;
		do
		{
			if (current_min > -((real)dim_hamil*3.0 / 4.0))
			{
				std::cout << "Local min is not close to idea global min, start over with random initial position\n";
				start.randu();
				start = sqrt(dim_para)*2.0*(start - 0.5)*rand;
				is_global_min = false;
			}
			//since we know the possible global minimum
			else if (current_min > -((real)dim_hamil - (1.0*epsilon)))
			{
				std::cout << "Local min is close to idea global min, start over with a small shift\n";
				start.randu();
				start = current_min_coordinate + (1 / 16.0)*start;
				is_global_min = false;
			}
			else
			{
				std::cout << "Good enough!\n";
				is_global_min = true;
			}

			temp = target.function_value(start);
		} while ((temp > -1.0) || (temp > -((double)dim_hamil) / 8.0) || temp!=temp);

	} while (!is_global_min);


	outputfile << "Programme ends at" << Deng::Misc::TimeStamp() << std::endl;
    outputfile.close();
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

    return 0;
}



//used to print out energy levels without control field
//outputfile names ""eigenvalues.dat"
//should use python to do further plot
void Verify_level_crossing(const std::string inputfile)
{
	
	//read in arguments from file
	Deng::Misc::ReadArguments::FromFile myargs(inputfile);
	//assign the values
	const unsigned int num_spinor = myargs("num_spinor");
	const unsigned int dim_hamil = 1 << num_spinor;
	const unsigned int dim_para_each_direction = myargs("dim_para_each_direction");
	//const unsigned int dim_para = (2 * num_spinor + 1)*dim_para_each_direction;
	const unsigned int dim_para = 3 * num_spinor * dim_para_each_direction;
	const double rand = myargs("rand");

	std::cout << "Number of spin: " << num_spinor << std::endl;
	std::cout << "Dim of Paramateric space: " << dim_para << "  with " << dim_para_each_direction << " paras for each direction" << std::endl;
	std::cout << "start with " << rand << " randomness" << std::endl << std::endl << std::endl;

	const unsigned int N_t = myargs("N_t");
	const double tau = myargs("tau");
	const double epsilon = myargs("epsilon");
	const double epsilon_gradient = sqrt(dim_hamil)*epsilon;
	const unsigned int conj_grad_max_iter = myargs("conj_grad_max_iter");
	
	
	
	//output to
	std::ofstream outputfile;
	std::string filename;
	//set file name
	filename = "eigenvalues";
	filename = filename + ".dat";
	//filename = filename + Deng::Misc::TimeStamp() + ".dat";
	//set output & format
	outputfile.open(filename);
	outputfile.precision(7);
	outputfile.setf(std::ios::fixed);


	//Transverse_Ising<elementtype, real> H(num_spinor, N_t, tau, dim_para, dim_para/3);
	//Transverse_Ising_Local_Control<elementtype, real> H(num_spinor, N_t, tau, dim_para, dim_para_each_direction);
	//Transverse_Ising_Impulse_Local<elementtype, real> H(num_spinor, N_t, tau, dim_para, dim_para_each_direction);
	LMG<elementtype, real> H(num_spinor, N_t, tau, dim_para, dim_para_each_direction);

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
		eigen_energy.t().raw_print(outputfile);
	}

	outputfile.close();
}
