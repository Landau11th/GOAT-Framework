#ifndef DENG_MISC_HPP
#define DENG_MISC_HPP

#include <iostream>
#include <cassert>
#include <string>
#include <regex>
#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <ctime>

namespace Deng
{
	namespace Misc
	{
		std::string TimeStamp();

		//initially for integers mod N
		//might be extended
		namespace FiniteGroup
		{
			template <class int_type, int_type order>
			class Abelian_Additive_Group_of_Z;

			template<typename int_type, int_type order>
			Abelian_Additive_Group_of_Z<int_type, order> operator+(
				const Abelian_Additive_Group_of_Z<int_type, order>& left,
				const Abelian_Additive_Group_of_Z<int_type, order>& right
				)
			{
				Abelian_Additive_Group_of_Z<int_type, order> result;
				result.value = left.value + right.value;
				result.value -= result.value > order ? order : 0;

				return result;
			};
			template<typename int_type, int_type order>
			Abelian_Additive_Group_of_Z<int_type, order> operator-(
				const Abelian_Additive_Group_of_Z<int_type, order>& left,
				const Abelian_Additive_Group_of_Z<int_type, order>& right
				)
			{
				//need this temp because right is const
				auto temp = right;
				return left + (-temp);
			};

			template <class int_type, int_type order>
			class Abelian_Additive_Group_of_Z
			{
			protected:
				int_type value;

			public:
				//modulo
				static const int_type n = order;
				//
				int_type v() const { return value; };

				Abelian_Additive_Group_of_Z(int_type i = 0)
					: value(i >= 0 ? (i%order) : (order - (i%order))) {	};
				//seems do not need to define copy/assign constructor

				//find the additive inverse
				Abelian_Additive_Group_of_Z<int_type, order>& operator-()
				{
					this->value = -this->value;
					return *this;
				}
				explicit operator int_type() const { return this->value; }

				//Abelian_Additive_Group_of_Z<int_type, order> operator+(
				//	const Abelian_Additive_Group_of_Z<int_type, order>& right
				//	)
				//{
				//	Abelian_Additive_Group_of_Z<int_type, order> result;
				//	result.value = this->value + right.value;
				//	result.value -= result.value > order ? order : 0;

				//	return result;
				//};

				//Abelian_Additive_Group_of_Z<int_type, order> operator-(
				//	const Abelian_Additive_Group_of_Z<int_type, order>& right
				//	)
				//{
				//	return *this + (-right);
				//};


				virtual ~Abelian_Additive_Group_of_Z() = default;

				template<typename int_type, int_type order>
				friend Abelian_Additive_Group_of_Z<int_type, order> operator+(const Abelian_Additive_Group_of_Z<int_type, order>& left,
					const Abelian_Additive_Group_of_Z<int_type, order>& right);

				template<typename int_type, int_type order>
				friend Abelian_Additive_Group_of_Z<int_type, order> operator-(const Abelian_Additive_Group_of_Z<int_type, order>& left,
					const Abelian_Additive_Group_of_Z<int_type, order>& right);
			protected:
				void Check_Consistency()
				{
					assert(value >= 0 && value < n && "range of value is incorrect!");
				};
			};
		}



		//read arguments from file
		namespace ReadArguments
		{
			//all arguments are turned to double
			class FromFile
			{
			protected:
				std::string _file_name;
				std::ifstream _file;
				//save the arguments in a map with key string
				std::map<std::string, double> _arguments;
			public:
				FromFile() {};
				FromFile(const std::string& file_name)
				{
					ObtainArguments(file_name);
				};
				std::string File_Name() const { return _file_name; };

				//read in arguments from file
				void ObtainArguments(const std::string& file_name);
				//output all read arguments to output_strm
				void operator() (std::ostream & output_strm) const;
				//get access to a certain arguments
				double operator() (const std::string& arg_name) const;
			};
		}
	}
}


#endif // !DENG_MISC_HPP