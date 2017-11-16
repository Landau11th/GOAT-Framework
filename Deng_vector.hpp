#ifndef DENG_VECTOR_HPP
#define DENG_VECTOR_HPP

#include <iostream>
#include <complex>
#include <armadillo>
//#include <typeinfo>
#include <cassert>

//define vectors
namespace Deng
{
    template <typename Field>
    class Col_vector;

    //addition
    template<typename Field_l, typename Field_r>
    Col_vector<Field_r> operator+(const Col_vector<Field_l>& l_vec, const Col_vector<Field_r> & r_vec)//addition
    {
        const unsigned int dim = l_vec.dimension();
        assert( (dim == r_vec.dimension()) && "Dimension mismatch in + (vector addition)!");

        Col_vector<Field_r> a(dim);
        for(unsigned int i = 0; i < dim; ++i)
        {
            a[i] = l_vec[i]+r_vec[i];
        }
        return a;
    }
    //subtraction
    template<typename Field_l, typename Field_r>
    Col_vector<Field_r> operator-(const Col_vector<Field_l>& l_vec, const Col_vector<Field_r> & r_vec)//subtraction
    {
        const unsigned int dim = l_vec.dimension();
        assert( (dim == r_vec.dimension()) && "Dimension mismatch in - (vector subtraction)!");

        Col_vector<Field_r> a(dim);
        for(unsigned int i = 0; i < dim; ++i)
        {
            a[i] = l_vec[i]-r_vec[i];
        }
        return a;
    }
    //scalar multiplication
    template<typename Scalar, typename Field_prime>
    Col_vector<Field_prime> operator*(const Scalar& k, const Col_vector<Field_prime> & r_vec)//scalar multiplication
    {
        const unsigned int dim = r_vec.dimension();
        Col_vector<Field_prime> a(dim);

        for(unsigned int i = 0; i < dim; ++i)
        {
            a[i] = k*r_vec[i];
        }
        return a;
    }
    //element-wise multiplication
    template<typename Field_l, typename Field_r>
    Col_vector<Field_r> operator%(const Col_vector<Field_l>& l_vec, const Col_vector<Field_r> & r_vec)//element-wise multiplication
    {
        const unsigned int dim = l_vec.dimension();
        assert( (dim == r_vec.dimension()) && "Dimension mismatch in % (element-wise multiplication)!");

        Col_vector<Field_r> a(dim);
        for(unsigned int i = 0; i < dim; ++i)
        {
            a[i] = l_vec[i]*r_vec[i];
        }
        return a;
    }
    //dot product. Field_l could be either Field_r or Scalar
    //for now only works for real/hermitian matrix!!!!!!!!!!!!!!!
    //choosing ^ is not quite appropriate
    template<typename Field_l, typename Field_r>
    Field_r operator^(const Col_vector<Field_l>& l_vec, const Col_vector<Field_r> & r_vec)
    {
        const unsigned int dim = l_vec.dimension();
        assert( (dim == r_vec.dimension()) && "Dimension mismatch in ^ (inner product)!" );

        Field_r a = r_vec[0];
        //only work for scalars and matrices
        a = 0*a;

        for(unsigned int i = 0; i < dim; ++i)
        {
            a += l_vec[i]*r_vec[i];
        }
        return a;
    }



    template <typename Field>
    class Col_vector
    {
    protected:
        unsigned int _dim;
        Field* _vec;//vector of the number field Field
    public:
        Col_vector();//default constructor, set _vec to nullptr
        Col_vector(unsigned int dim);//constructor
        Col_vector(const Col_vector<Field>& c);//copy constructor
        Col_vector(Col_vector<Field>&& c);//move constructor

        void set_size(unsigned int dim);//in case constructor could not be used, say in an array
        unsigned int dimension() const//returns _dim
        {
            return _dim;
        }
        //destructor. make it virtual in case we need to inherit
        virtual ~Col_vector();


        //overloading operators
        //member functions
        //copy assignment operator
        Col_vector<Field> &operator=(const Col_vector<Field> & rhs);
        //move constructor
        Col_vector<Field> &operator=(Col_vector<Field> &&rhs) noexcept;
        //other assignment operator
        //be careful with these operators!!!!!
        void operator+=(const Col_vector<Field>& rhs);
        void operator-=(const Col_vector<Field>& rhs);
        void operator*=(const Field k);//scalar multiplication
        void operator*=(const Col_vector<Field>& b);//element-wise multiplication
        //element access
        Field& operator[](unsigned int idx)
        {
			assert(idx < _dim && "error in []!");
			return _vec[idx];
        }
        const Field& operator[](unsigned int idx) const
        {
			assert(idx < _dim && "error in []!");
			return _vec[idx];
        }


        //non-member operator
        //addition
        template<typename Field_l, typename Field_r>
        friend Col_vector<Field_r> operator+(const Col_vector<Field_l>& l_vec, const Col_vector<Field_r> & r_vec);
        //subtraction
        template<typename Field_l, typename Field_r>
        friend Col_vector<Field_r> operator-(const Col_vector<Field_l>& l_vec, const Col_vector<Field_r> & r_vec);
        //scalar multiplication
        template<typename Scalar, typename Field_prime>
        friend Col_vector<Field_prime> operator*(const Scalar & k, const Col_vector<Field_prime> & r_vec);
        //scalar multiplication in another order
        //ambiguous!!!!!!
        //template<typename Scalar, typename Field_prime>
        //friend Col_vector<Field_prime> operator*(const Col_vector<Field_prime> & r_vec, const Scalar & k);
        //element-wise multiplication
        template<typename Field_l, typename Field_r>
        friend Col_vector<Field_r> operator%(const Col_vector<Field_l>& l_vec, const Col_vector<Field_r> & r_vec);
        //inner product. Field_l could be either Field_r or Scalar
        //for now only works for real matrix!!!!!!!!!!!!!!!
        template<typename Field_l, typename Field_r>
        friend Field_r operator^(const Col_vector<Field_l>& l_vec, const Col_vector<Field_r> & r_vec);


        //Field operator%(const Col_vector<Field>& b); //inner product



    //    template <typename Field2>
    //    friend std::ostream& operator<< (std::ostream& out, const Col_vector<Field2>& f);

    };
}

#endif //DENG_VECTOR_HPP
