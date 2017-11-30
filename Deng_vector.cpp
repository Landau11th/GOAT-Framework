#include"Deng_vector.hpp"


using namespace Deng;

//explicit instantiations

//template class Col_vector<int>;
template class Col_vector<float>;
template class Col_vector<double>;
template class Col_vector<std::complex<float> >;
template class Col_vector<std::complex<double> >;

//block vectors
template class Col_vector<arma::Mat<float> >;
template class Col_vector<arma::Mat<double> >;
template class Col_vector<arma::Mat<std::complex<float> > >;
template class Col_vector<arma::Mat<std::complex<double> > >;


//#define DENG_VECTOR_COMPLEX //to determine whether the field is complex (aim for inner product)
template <typename Field>
Col_vector<Field>::Col_vector()//default constructor, set _vec to nullptr
{
    _dim = 0;
    _vec = nullptr;
}
template <typename Field>
Col_vector<Field>::Col_vector(unsigned int dim)
{
    _dim  = dim;
    _vec = new Field[_dim];
}
template <typename Field>
Col_vector<Field>::Col_vector(const Col_vector<Field>& c)//copy constructor
{
    _dim = c.dimension();
    _vec = new Field[_dim];

    for(unsigned int i = 0; i < _dim; ++i)
        _vec[i] = c._vec[i];
}
template <typename Field>
Col_vector<Field>::Col_vector(Col_vector<Field>&& c)//move constructor
{
    _dim = c.dimension();
    _vec = c._vec;

    c._vec = nullptr;
}
template <typename Field>
void Col_vector<Field>::set_size(unsigned int dim)
{
    //if the size is already dim, there is no need to change
    if(_dim != dim)
    {
        _dim = dim;
        delete[] _vec;
        _vec = new Field[_dim];
    }
}
template <typename Field>
Col_vector<Field>::~Col_vector()
{
    //std::cout << "releasing " << _vec << std::endl;
    delete[] _vec;
	_vec = nullptr;
}
template <typename Field>
Col_vector<Field>& Col_vector<Field>::operator=(const Col_vector<Field> & b)
{
    set_size(b.dimension());

    for(unsigned int i = 0; i < b.dimension(); ++i)
    {
        this->_vec[i] = b[i];
    }
    /*
    if(dddim == b.dimension())
    {
        for(i=0; i<dddim; i++)
        {
            this->_vec[i] = b[i];
        }
    }
    else
    {
        printf("Error in operator '='!\n");
    }
    */
    return *this;
}
template <typename Field>
Col_vector<Field>& Col_vector<Field>::operator=(Col_vector<Field> &&rhs) noexcept
{



//    if(this != &rhs)
//    {
    assert( (this != &rhs) && "Memory clashes in operator '=&&'!\n");
    delete[] this->_vec;
    this->_vec = rhs._vec;
    rhs._vec = nullptr;
//    }
//    else
//    {
//        std::cout << "Memory clashes in operator '=&&'!\n";
//    }

    return *this;
}
template <typename Field>
void Col_vector<Field>::operator+=(const Col_vector<Field>& b)
{
    /*
    unsigned int dim = this->dimension();
    Col_vector<Field> a(dim);

    if(dim==b.dimension())
    {
        for(unsigned int i = 0; i < dim; ++i)
        {
            this->_vec[i] += b[i];
        }
    }
    else
    {
        std::cout << "Dimension mismatch in operator '+='!" << std::endl;
    }
    */
    //here neglect all the dimension check etc
    //since +=, -=, *= are usually used for optimal performance
    for(unsigned int i = 0; i < this->_dim; ++i)
    {
        this->_vec[i] += b[i];
    }
}
template <typename Field>
void Col_vector<Field>::operator-=(const Col_vector<Field>& b)
{
    /*
    int dim = this->dimension();
    Col_vector<Field> a(dim);

    if(dim==b.dimension())
    {
        int i;

        for(i=0; i<dim; i++)
        {
            this->_vec[i] -= b[i];
        }
    }
    else
    {
        std::cout << "Dimension mismatch in operator '-='!" << std::endl;
    }
    */
    //here neglect all the dimension check etc
    //since +=, -=, *= are usually used for optimal performance
    for(unsigned int i = 0; i < this->_dim; ++i)
    {
        this->_vec[i] -= b[i];
    }
}
template <typename Field>
void Col_vector<Field>::operator*=(const Field k)
{
    //here neglect all the dimension check etc
    //since +=, -=, *= are usually used for optimal performance
    for(unsigned int i = 0; i < this->_dim; ++i)
    {
        this->_vec[i] *= k;
    }
}
template <typename Field>
void Col_vector<Field>::operator*=(const Col_vector<Field>& b)
{
    //here neglect all the dimension check etc
    //since +=, -=, *= are usually used for optimal performance
    for(unsigned int i = 0; i < this->_dim; ++i)
    {
        this->_vec[i] *= b[i];
    }
}


//#ifdef DENG_VECTOR_COMPLEX
//template <typename Field>
//Field Col_vector<Field>::operator%(const Col_vector<Field>& b)
//{
//
//    int dddim = this->dimension();
//    Field a = 0.0;
//
//    if(dddim==b.dimension())
//    {
//        int i;
//
//        for(i=0; i<dddim; i++)
//        {
//            a += std::conj(this->vec[i] )* b.vec[i];
//        }
//    }
//    else
//    {
//        printf("Error in operator '*'!\n");
//    }
//
//    return a;
//}
//#else
///*
//template <typename Field>
//Field Col_vector<Field>::operator%(const Col_vector<Field>& b)
//{
//    if((typeid(Field)==typeid(arma::Mat<float>))
//    || (typeid(Field)==typeid(arma::Mat<double>))
//    || (typeid(Field)==typeid(arma::Mat<std::complex<float> >))
//    || (typeid(Field)==typeid(arma::Mat<std::complex<double> >))
//      )
//    {
//        std::cout << "Inner product is not defined for block vectors" << std::endl;
//        return 0;
//    }
//    else
//    {
//        int dddim = this->dimension();
//        Field a = 0;
//
//        if(dddim==b.dimension())
//        {
//            int i;
//
//            for(i=0; i<dddim; i++)
//            {
//                a += this->_vec[i]* b[i];
//            }
//        }
//        else
//        {
//            printf("Error in operator '*'!\n");
//        }
//
//        return a;
//    }
//}
//*/
//#endif // Col_vector_COMPLEX
//vector operations
template <typename Field>
Field dot_product(Col_vector<Field> a, Col_vector<Field> b)
{
    int dim = a.dimension();
    Field dot;

    if(dim == b.dimension())
    {
        int i;

        dot = 0.0;

        for(i=0; i<dim; i++)
        {
            dot += a[i]*b[i];
        }
    }
    else
    {
        printf("Error in dot product!\n");
    }

    return dot;
}
/*
template <typename Field>
std::ostream& operator<<(std::ostream& out, const Col_vector<Field>& f)
{
    for(int i = 0; i < f.dimension(); ++i)
    {
        out << f[i];
    }

    return out << std::endl;
}
*/


