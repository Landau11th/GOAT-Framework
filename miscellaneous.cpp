#include "miscellaneous.hpp"


std::string Deng::Misc::TimeStamp()
{
	time_t t = time(0);   // get time now
	struct tm time_info;
	struct tm * now = &time_info;
	localtime_s(now, &t);
	char buffer[80];
	strftime(buffer, 80, "%Y%m%d-%H%M%S", now);

	return buffer;
}


//read in arguments from file
void Deng::Misc::ReadArguments::FromFile::ObtainArguments(const std::string& file_name)
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
		if (pos < arg_info.length())
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
}
//output all read arguments to output_strm
void Deng::Misc::ReadArguments::FromFile::operator() (std::ostream & output_strm) const
{
	for (auto map_iter = _arguments.begin(); map_iter != _arguments.end(); ++map_iter)
	{
		output_strm << map_iter->first << " = " << map_iter->second << "\n";
	}
	output_strm << "\n";
}
//get access to a certain arguments
double Deng::Misc::ReadArguments::FromFile::operator() (const std::string& arg_name) const
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
}
