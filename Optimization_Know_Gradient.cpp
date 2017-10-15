#include "Optimization_Know_Gradient.hpp"

using namespace Deng::Optimization::Know_Gradient;

//specilizing the abstract class
template class Minimization<float>;
template class Minimization<double>;

template class Conj_Grad_Min<float>;
template class Conj_Grad_Min<double>;

template <typename real>
arma::Col<real> Conj_Grad_Min<real>::Conj_Grad_Search(arma::Col<real> start_coordinate) const
{
	//refer to Prof Wang Jian-sheng's lecture notes Numerical Recipe
	//std::cout << "here?";
	real lambda, gamma;
	coordinate = start_coordinate;

	real function_value;
	search_direction = f->negative_gradient(coordinate, function_value);

	unsigned int count_iteration = 0;
	for (count_iteration = 0; count_iteration < _max_iteration; ++count_iteration)
	{
		std::cout << "starting " << count_iteration << " search" << std::endl;
		//lambda must be bigger than 0, or the function is wrong
		lambda = OneD_Minimum(coordinate, search_direction);
		if (lambda < 0)
		{
			std::cout << "1D search gives negative lambda at iteration " << count_iteration << " of Conjugate Gradient function" << std::endl;
			break;
		}
		else
		{
			coordinate = lambda*search_direction + coordinate;
			//
			if (sqrt(arma::as_scalar(search_direction.t()*search_direction))<_epsilon)
			{
				break;
			}
			else
			{
				previous_search_direction = search_direction;
				search_direction = f->negative_gradient(coordinate, function_value);
				gamma = arma::as_scalar(search_direction.t()*search_direction) / arma::as_scalar(previous_search_direction.t()*previous_search_direction);
				search_direction = search_direction + gamma*previous_search_direction;
			}
		}
	}
	if (count_iteration >= _max_iteration)
	{
		std::cout << "Conjugate Gradient function hits maximum allowed iterations" << std::endl;
	}

	return coordinate;
}


//arma::Col<double> Conj_Grad_Min::Conj_Grad_Search(arma::Col<double> start_coordinate)
//{
//    //refer to Prof Wang Jian-sheng's lecture notes Numerical Recipe
//    //std::cout << "here?";
//    double lambda, gamma;
//    coordinate = start_coordinate;
//    Target_function_and_negative_gradient(coordinate, current_search_direction);
//
//    unsigned int count_iteration = 0;
//    for(count_iteration = 0; count_iteration < _max_iteration; ++count_iteration)
//    {
//        std::cout << "starting " << count_iteration << " search" << std::endl;
//        //lambda must be bigger than 0, or the function is wrong
//        lambda = OneD_Minimum(coordinate, current_search_direction);
//        if(lambda < 0)
//        {
//            std::cout << "1D search gives negative lambda at iteration " << count_iteration << " of Conjugate Gradient function"<< std::endl;
//            break;
//        }
//        else
//        {
//            coordinate = lambda*current_search_direction + coordinate;
//            //
//            if(sqrt(arma::as_scalar(current_search_direction.t()*current_search_direction))<_epsilon)
//            {
//                break;
//            }
//            else
//            {
//                previous_search_direction = current_search_direction;
//                Target_function_and_negative_gradient(coordinate, current_search_direction);
//                gamma = arma::as_scalar(current_search_direction.t()*current_search_direction)/arma::as_scalar(previous_search_direction.t()*previous_search_direction);
//                current_search_direction = current_search_direction + gamma*previous_search_direction;
//            }
//        }
//    }
//    if(count_iteration >= _max_iteration)
//    {
//        std::cout << "Conjugate Gradient function hits maximum allowed iterations" << std::endl;
//    }
//
//    return coordinate;
//}



//optimization in 1D
template<typename real>
real Conj_Grad_Min<real>::OneD_Minimum(const arma::Col<real> start_coordinate, const arma::Col<real> search_direction_given) const
{
	real lambda;
    static const real Golden_Ratio = (sqrt(5.0)-1)/2;
    const real step_size = 1.0;//sqrt(as_scalar(search_direction_given.t()*search_direction_given) ) * 1.0;
    const real epsilon_scaled = _epsilon/sqrt(arma::as_scalar(search_direction_given.t()*search_direction_given));

    //left, middle and right values
	real left, middle, right, next;
    left = middle = right = 0.0;
    //corresponding target function values
	real f_left, f_middle, f_right, f_next;

    middle = Golden_Ratio*step_size;
    right = step_size;

    f_left = f->function_value(start_coordinate);
    f_middle = f->function_value(start_coordinate + middle*search_direction_given);
    f_right = f->function_value(start_coordinate + right*search_direction_given);

	//first choose a appropriate initial bracket with a convex-like shape

	int find_shape = 0;
	const unsigned int find_shape_max = 20;
	for (find_shape = 0; find_shape < find_shape_max; ++find_shape)
	{
		//we first need find a middle point has less function value than left
		//such point exists because search_direction_given points at the decreasing direction
		if ((f_middle > f_left))
		{
			//the middle point is the new right point
			right = middle;
			f_right = f_middle;
			//find new middle point
			middle *= Golden_Ratio;
			f_middle = f->function_value(start_coordinate + middle*search_direction_given);

		}
		//then find right one bigger than middle
		//seems works. BE CAREFUL
		else if ((f_right < f_middle))
		{
			middle = right;
			f_middle = f_right;

			right = right / Golden_Ratio;
			f_right = f_right = f->function_value(start_coordinate + right*search_direction_given);
		}
		else
		{
			break;
		}
	}
	if (find_shape >= find_shape_max)
	{
		std::cout << "Shit happened when looking for appropriate initial bracket!" << std::endl;
		return -1;
	}



    //makes sure the middle one is always less than side ones
    //real Golden ratio algorithm
    unsigned int count_iteration = 0;
    //the criteria does not contain absolute value because we know clearly which side the local minimum lies
    while( (right - left)>=epsilon_scaled && count_iteration < _max_iteration)
    {
        if(f_left >= f_right)
        {
            //next bracket is between left and middle
            next = left + Golden_Ratio*(middle - left);
            f_next = f->function_value(start_coordinate + next*search_direction_given);

            if(f_next >= f_middle)
            {
                left = next;
                f_left = f_next;
            }
            else
            {
                right = middle;
                f_right = f_middle;

                middle = next;
                f_middle = f_next;
            }
        }
        else
        {
            //next bracket between middle and right
            next = right - Golden_Ratio*(right - middle);
            f_next = f->function_value(start_coordinate + next*search_direction_given);

            if(f_next >= f_middle)
            {
                right = next;
                f_right = f_next;
            }
            else
            {
                left = middle;
                f_left = f_middle;

                middle = next;
                f_middle = f_next;
            }
        }
        ++count_iteration;
    }
    if (count_iteration >= _max_iteration)
    {
        std::cout << "Golden-section search hits maximum allowed iterations" << std::endl;
        return -1;
    }
    else
    {
        return (right + left)/2.0;
    }
}
