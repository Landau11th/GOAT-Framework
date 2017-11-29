#pragma once
#include<iostream>
#include<cassert>
#include<string>
#include<regex>
#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <ctime>

namespace Deng
{
	namespace Misc
	{
		//create time stamp
		//usually for filename not being covered
		std::string TimeStamp()
		{
			time_t t = time(0);   // get time now
			struct tm time_info;
			struct tm * now = &time_info;
			localtime_s(now, &t);
			char buffer[80];
			strftime(buffer, 80, "%Y%m%d-%H%M%S", now);

			return buffer;
		}
		
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


		namespace ReadArguments
		{
			//all arguments are turned to double first
			class FromFile
			{
			protected:
				std::string _file_name;
				std::ifstream _file;
				std::map<std::string, double> _arguments;
			public:
				FromFile() {};
				FromFile(const std::string& file_name)
				{
					ObtainArguments(file_name);
				};

				std::string File_Name() const { return _file_name; };

				double operator() (std::string& arg_name) const
				{
					try
					{
						return _arguments.at(arg_name);
					}
					catch (const std::out_of_range& oor)
					{
						std::cerr << "Error in " << __func__ << ": " << oor.what() << ": "
							<< "could not find value of " << arg_name << "\n";
						std::cerr << "Notice: argument name is case sensitive" << "\n";
						std::terminate();
						return 0;
					}
				};

				void operator() () const
				{
					for (auto map_iter = _arguments.begin(); map_iter != _arguments.end(); ++map_iter)
					{
						std::cout << map_iter->first << "=" << map_iter->second << "\n";
					}
					std::cout << "\n";
				};

				void ObtainArguments(const std::string& file_name)
				{
					_file_name = file_name;

					_file.open(_file_name);
					if (!_file.is_open())
						std::cerr << "Error in " << __func__ << " while opening file: " << _file_name << "\n\n";

					std::string line;
					while (std::getline(_file, line))
					{
						//in <regex>
						auto arg_info = std::regex_replace(line, std::regex(" "), "");
						auto pos = arg_info.find_first_of("#");
						if(pos < arg_info.length())
							arg_info.erase(pos, arg_info.length() - pos);

						pos = arg_info.find("=");

						if (pos >= arg_info.length())
							continue;


						auto arg_name = arg_info.substr(0, pos);
						//convert to lower case for format issue
						//in <algorithm>
						//std::transform(arg_name.begin(), arg_name.end(), arg_name.begin(), ::tolower);

						auto arg_value = arg_info.substr(pos + 1);

						try
						{
							auto insert_result = _arguments.insert(std::pair<std::string, double>(arg_name, std::stod(arg_value)));
							if (insert_result.second == false)
							{
								std::cerr << "duplicated value: " << arg_name << "\n"
									<< "stick with the first value encounted\n";
							}
						}
						catch (const std::invalid_argument& ia)
						{
							std::cerr << "Error in " << __func__ << ": " << ia.what() << ": could not convert value "
								<< arg_value << " of " << arg_name << "\n";
						}
						catch (const std::out_of_range& oor)
						{
							std::cerr << "Error in " << __func__ << ": " << oor.what() << ": could not convert value "
								<< arg_value << " of " << arg_name << "\n";
						}
					}

					_file.close();

				};

			};
		}
	}
}