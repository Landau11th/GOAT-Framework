#ifndef DENG_OPTIMIZATION_KNOW_GRADIENT_HPP
#define DENG_OPTIMIZATION_KNOW_GRADIENT_HPP

#include <iostream>
#include <complex>
#include <armadillo>
#include"Optimization_Know_Function.hpp"

namespace Deng
{
	namespace Optimization
	{
		namespace Know_Gradient
		{
			template<typename real>
			class Minimization : public Deng::Optimization::Know_Function::Minimization<real>
			{
			public:
				//search directions
				mutable arma::Col<real> search_direction;

				//Target_function<real>* f;
				////to cover the defination of Assign_Target_Function in Deng::Optimization::Know_Function::Minimization<real>
				////mainly because we want f points to Deng::Optimization::Know_Gradient::Target_function
				////THIS MIGHT BE DANGEROUS!!!!!
				//void Assign_Target_Function(Target_function<real>* pt_f) { f = pt_f; };

				Deng::Optimization::Know_Function::Minimization<real>::Minimization;
				//omit the default constructor.
				//Must know the parameters before declare this object
				Minimization(unsigned int dim_para, real epsilon, unsigned int max_iteration) : Deng::Optimization::Know_Function::Minimization<real>(dim_para, epsilon, max_iteration)
				{
					search_direction = arma::zeros<arma::Col<real> >(_dim_para);
				};

			};

			template<typename real>
			class Conj_Grad_Min : public Minimization<real>
			{
			public:
				//previous search directions
				//it is needed in conj grad for a better search direction
				mutable arma::Col<real> previous_search_direction;
				//constructor
				Minimization<real>::Minimization;
				Conj_Grad_Min(unsigned int dim_para, real epsilon, unsigned int max_iteration) : Minimization<real>(dim_para, epsilon, max_iteration)
				{
					previous_search_direction = arma::zeros<arma::Col<real> >(_dim_para);
				};

				//if not giving a start coordinate, we start from 0
				virtual arma::Col<real> Conj_Grad_Search(arma::Col<real> start_coordinate = 0) const;
				//optimization in 1D
				virtual real OneD_Minimum(const arma::Col<real> start_coordinate, const arma::Col<real> search_direction_given) const;
				//virtual double Golden_Section(const arma::Col<double> give_coordinate, const arma::Col<double> give_search_direction);
			};

		}
	}
}


#endif // DENG_OPTIMIZATION_KNOW_GRADIENT_HPP