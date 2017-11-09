#include "Optimization.hpp"

using namespace Deng::Optimization;

//specilizing the abstract class
template class Target_function<float>;
template class Target_function<double>;

template class Min_Know_Function<float>;
template class Min_Know_Function<double>;

template class Min_Know_Gradient<float>;
template class Min_Know_Gradient<double>;

template class Min_Conj_Grad<float>;
template class Min_Conj_Grad<double>;


template <typename real>
arma::Col<real> Min_Conj_Grad<real>::Conj_Grad_Search(arma::Col<real> start_coordinate) const
{
	//refer to Prof Wang Jian-sheng's lecture notes Numerical Recipe
	std::cout << "Start search\n";
	real lambda, gamma;
	//C++03 14.6.2 Dependent names
    //In the definition of a class template or a member of a class template,
    //if a base class of the class template depends on a template-parameter,
    //the base class scope is not examined during unqualified name lookup either at the point of definition of the class template
    //or member or during an instantiation of the class template or member.
    coordinate = start_coordinate;

	real function_value;
	search_direction = f->negative_gradient(coordinate, function_value);

	unsigned int count_iteration = 0;
	int count_close_function_value = 0;
	const int count_close_function_value_max = start_coordinate.size();
	for (count_iteration = 0; count_iteration < _max_iteration; ++count_iteration)
	{
		if (sqrt(arma::as_scalar(search_direction.t()*search_direction)) < _epsilon_gradient)
		{
			std::cout << "\n";
			std::cout << "gradient is close to 0 at coordinate" << std::endl;
			std::cout << coordinate.t();
			std::cout << "with function value " << function_value << std::endl;
			break;
		}

		std::cout << "Search direction:" << search_direction.t();
		std::cout << "from coordinate :" << coordinate.t();
		std::cout << "function value  :" << function_value << std::endl;
		lambda = Opt_1D(coordinate, search_direction, f, _max_iteration, _epsilon);
		if (lambda < 0)
		{
			std::cout << "!!!1D search gives negative lambda at iteration " << count_iteration << " of Conjugate Gradient function" << std::endl;
			break;
		}
		if (lambda != lambda)
		{
			std::cout << "Shit happens. Possibly NaN" << std::endl;
			break;
		}

		//time costs for calculation derivative once
		clock_t my_time;
		my_time = clock();

		coordinate = lambda*search_direction + coordinate;
		previous_search_direction = search_direction;
		real temp_function_value;
		search_direction = f->negative_gradient(coordinate, temp_function_value);

		//conjugate gradient
		gamma = arma::as_scalar(search_direction.t()*search_direction) / arma::as_scalar(previous_search_direction.t()*previous_search_direction);
		search_direction = search_direction + gamma*previous_search_direction;

		std::cout << count_iteration << " th search stops at value: " << function_value << std::endl;
		std::cout << "with coordinate: " << coordinate.t();
		//time costs for calculation derivative once
		std::cout << "calculating derivative costs " << 1000 * ((clock() - (float)my_time) / CLOCKS_PER_SEC) << " ms" << "\n\n";

		//see if the new func value is close enough to old func value
		if (function_value - temp_function_value < _epsilon)
			++count_close_function_value;
		else
			count_close_function_value = 0;

		function_value = temp_function_value;

		if (count_close_function_value >= count_close_function_value_max)
		{
			std::cout << "\n";
			std::cout << "Consecutively generate " << count_close_function_value_max << " close function values around\n";
			std::cout << coordinate.t();
			std::cout << "with function value=" << function_value << " and gradient magnitue=" << sqrt(arma::as_scalar(search_direction.t()*search_direction)) << std::endl;
			break;
		}

	}
	if (count_iteration >= _max_iteration)
	{
		std::cout << "Conjugate Gradient function hits maximum allowed iterations" << std::endl;
	}

	return this->coordinate;
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


template float Deng::Optimization::OneD_Golden_Search(const arma::Col<float> start_coordinate, const arma::Col<float> search_direction_given, const Target_function<float>* const f, const unsigned int max_iteration, const float epsilon);
template double Deng::Optimization::OneD_Golden_Search(const arma::Col<double> start_coordinate, const arma::Col<double> search_direction_given, const Target_function<double>* const f, const unsigned int max_iteration, const double epsilon);

//optimization in 1D, alone given direction
template<typename real>
real Deng::Optimization::OneD_Golden_Search(const arma::Col<real> start_coordinate, const arma::Col<real> search_direction_given, const Target_function<real>* const f, const unsigned int max_iteration, const real epsilon)
{
    static const real Golden_Ratio = (sqrt(5.0)-1)/2;
	const real step_size = 1.0 / sqrt(as_scalar(search_direction_given.t()*search_direction_given));
    //const real epsilon_scaled = epsilon/sqrt(arma::as_scalar(search_direction_given.t()*search_direction_given));

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

	unsigned int find_shape = 0;
	//how many times we try to find an appopriate shape
	//golden ration to the power of 30 is about 5e-7, which is in enough for general purpose
	static const unsigned int find_shape_max = 30;
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
			f_right = f->function_value(start_coordinate + right*search_direction_given);
		}
		else
		{
			break;
		}
	}
	if (find_shape >= find_shape_max)
	{
		std::cout << "Cannot find smaller value " << "up to " << right << std::endl;
		//std::cout << "along " << search_direction_given.t() << " (gradient direction)" << std::endl;
		//std::cout << "start from " << start_coordinate.t() << std::endl;
		return -1;
	}


    //makes sure the middle one is always less than side ones
    //real Golden ratio algorithm
    unsigned int count_iteration = 0;
    //the criteria does not contain absolute value because we know clearly which side the local minimum lies
	//for (count_iteration = 0; count_iteration < max_iteration + find_shape; ++count_iteration)
    while( (f_right + f_left - 2.0*f_middle)/2>= epsilon )//&& count_iteration < max_iteration)
    {
		++count_iteration;
		if ((f_right + f_left - 2.0*f_middle) / 2 >= epsilon)
			break;

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
    }
    if (count_iteration >= max_iteration)
    {
        std::cout << "Golden-section search hits maximum allowed iterations" << std::endl;
    }    
	return (right + left)/2.0;

}

