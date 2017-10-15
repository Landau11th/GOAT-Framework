#include "Optimization_Know_Function.hpp"

using namespace Deng::Optimization::Know_Function;

//specilizing the abstract class
template class Target_function<float>;
template class Target_function<double>;

template class Minimization<float>;
template class Minimization<double>;