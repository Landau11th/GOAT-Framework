#include "Optimization_Know_Function.hpp"

using namespace Deng::Optimization::Know_Function;

//specilizing the abstract class
template class Deng::Optimization::Target_function<float>;
template class Deng::Optimization::Target_function<double>;

template class Minimization<float>;
template class Minimization<double>;