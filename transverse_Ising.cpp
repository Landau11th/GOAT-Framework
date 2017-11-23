#include"transverse_Ising.hpp"

template class Transverse_Ising<std::complex<float>, float>;
template class Transverse_Ising<std::complex<double>, double>;

template<typename Field, typename Parameter>
Transverse_Ising<Field, Parameter>::Transverse_Ising(const unsigned int num_spin, const unsigned int N_t, const Parameter tau,
	const unsigned int dim_para, const unsigned int dim_para_each_direction, const Parameter hbar)
	: Deng::GOAT::Hamiltonian<Field, Parameter>((1 << num_spin), N_t, tau, dim_para),
	_omega(2.0*Pi / tau), _hbar(hbar), _num_spin(num_spin), _dim_para_each_direction(dim_para_each_direction), minus_i_over_hbar(0.0, -1.0)
	//2 for the dimension of spin half system
{
	if ((dim_para%_dim_para_each_direction))
		std::cout << "Dimension of parametric space does not fit! Could cause potential problem\n";

	//set up Pauli matrices
	S.set_size(3);
	S[0].zeros(2, 2);
	S[0](0, 1) = 1;
	S[0](1, 0) = 1;
	S[1].zeros(2, 2);
	S[1](0, 1) = -imag_i;
	S[1](1, 0) = imag_i;
	S[2].zeros(2, 2);
	S[2](0, 0) = 1;
	S[2](1, 1) = -1;
	S_identity.zeros(2, 2);
	S_identity(0, 0) = 1;
	S_identity(1, 1) = 1;
	//std::cout << S_identity;
	//S = 0.5*_hbar*S;//Eq 3.2

	//initialize the interaction part
	const unsigned int N = this->_N;
	interaction.set_size(N, N);
	interaction.zeros();
	//calculate the interaction part
	for (unsigned int i = 0; i < (num_spin-1); ++i)
	{
		arma::Mat<Field> temp = i == 0 ? S[2] : S_identity;
		for (unsigned int j = 1; j < num_spin; ++j)
		{
			temp = arma::kron(temp, (j == i) || (j == i + 1) ? S[2] : S_identity);
		}
		interaction -= temp;
	}
	//interaction *= _J;

	//total S alone 3 directions
	S_total.set_size(3);
	for (unsigned int i = 0; i < 3; ++i)
	{
		S_total[i].set_size(N, N);
		S_total[i].zeros();
		//calculate the total S alone 3 directions
		for (unsigned int j = 0; j < num_spin; ++j)
		{
			arma::Mat<Field> temp_B = j == 0 ? S[i] : S_identity;
			for (unsigned int k = 1; k < num_spin; ++k)
			{
				temp_B = arma::kron(temp_B, j == k ? S[i] : S_identity);
			}
			S_total[i] += temp_B;
		}
	}
}
template<typename Field, typename Parameter>
Deng::Col_vector<Parameter> Transverse_Ising<Field, Parameter>::B(Parameter t) const
{
	Deng::Col_vector<Parameter> B_field(3);

	//rotating B field
	//B_field[0] = cos(0.25*_omega*t)*_B_x_max;
	//B_field[1] = _B_y_max;
	//B_field[2] = sin(0.25*_omega*t)*_B_z_max;

	//B_field[0] = this->_B_x_max *(1 + t / this->_tau);
	//B_field[1] = 0.0;
	//B_field[2] = this->_B_z_max *t / this->_tau *t / this->_tau;

	//B_field[0] = this->_B_x_max ;
	//B_field[1] = 0.0;
	//B_field[2] = this->_B_z_max *t / this->_tau;

	////benchmark constant B field
	//B_field[0] = this->_B_x_max;
	//B_field[1] = 0.0;
	//B_field[2] = this->_B_z_max;

	////benchmark quadratic
	//B_field[0] = this->_B_x_max;
	//B_field[1] = 0.0;
	//B_field[2] = this->_B_z_max*(t / this->_tau)*(1 + t / this->_tau);

	//benchmark trig
	B_field[0] = this->_B_x_max;
	B_field[1] = 0.0;
	B_field[2] = this->_B_z_max*sin(_omega *t);

	return B_field;
}
//give H_0, with time dependent parameter, but not control field
template<typename Field, typename Parameter>
arma::Mat<Field> Transverse_Ising<Field, Parameter>::H_0(Parameter t) const
{
	//bracket is necessary as operator ^ is of low priority
	return interaction + (B(t) ^ S_total);
}
//calculate control magnetic field based on the parameters
template<typename Field, typename Parameter>
Deng::Col_vector<Parameter> Transverse_Ising<Field, Parameter>::control_field(Parameter t) const
{
	Deng::Col_vector<Parameter> ctrl(3);

	for (unsigned int i = 0; i < 3; ++i)
		ctrl[i] = control_field_component(t, i*_dim_para_each_direction);

	return ctrl;
}
//calculate certain component with parameters starting from para_idx_begin
template<typename Field, typename Parameter>
Parameter Transverse_Ising<Field, Parameter>::control_field_component(const Parameter t, const unsigned int para_idx_begin) const
{
	assert(para_idx_begin%_dim_para_each_direction == 0 && "error in control_field_component");
	Parameter component = 0.0;
	const unsigned int if_has_const = _dim_para_each_direction % 2;
	unsigned int mode, trig;
	for (unsigned int i = 0; i < (_dim_para_each_direction - if_has_const); ++i)
	{
		mode = i / 2 + 1;
		trig = i % 2;
		//control field vanish at boundary
		component += trig == 0 ? this->parameters(i + para_idx_begin) * sin(mode * _omega * t) : this->parameters(i + para_idx_begin) * (cos(mode * _omega * t) - 1.0);
	}
	if (if_has_const)
		component += _dim_para_each_direction - 1;

	return component;
}
//control Hamiltonian
template<typename Field, typename Parameter>
arma::Mat<Field> Transverse_Ising<Field, Parameter>::H_control(Parameter t) const
{
	return control_field(t) ^ S_total;
}
//dynamics of GOAT
template<typename Field, typename Parameter>
Deng::Col_vector<arma::Mat<Field>> Transverse_Ising<Field, Parameter>::Dynamics(const Parameter t) const
{
	Deng::Col_vector<arma::Mat<Field> > iH_and_partial_H(this->_dim_para + 1);

	iH_and_partial_H[0] = Dynamics_U(t);

	for (unsigned int i = 1; i <= this->_dim_para; ++i)
	{
		//could be generalize?
		//double original_para = parameters[i];
		this->parameters[i - 1] += 0.0078125;
		Deng::Col_vector<Parameter> partial_control = control_field(t);
		this->parameters[i - 1] -= 0.0078125;
		partial_control = 128.0*(partial_control - control_field(t));

		//iH_and_partial_H[i] = (-imag_i / _hbar)*(partial_control^S_total);
		iH_and_partial_H[i] = minus_i_over_hbar*(partial_control^S_total);
	}

	return iH_and_partial_H;
}
//dynamics of plain Hamiltonian
template<typename Field, typename Parameter>
arma::Mat<Field> Transverse_Ising<Field, Parameter>::Dynamics_U(const Parameter t) const
{
	//return (-imag_i / _hbar)*(H_0(t) + H_control(t));
	return minus_i_over_hbar*(H_0(t) + H_control(t));
}






template class Transverse_Ising_Local_Control<std::complex<float>, float>;
template class Transverse_Ising_Local_Control<std::complex<double>, double>;


template<typename Field, typename Parameter>
Transverse_Ising_Local_Control<Field, Parameter>::Transverse_Ising_Local_Control(const unsigned int num_spin, const unsigned int N_t, const Parameter tau,
	const unsigned int dim_para, const unsigned int dim_para_each_direction, const Parameter hbar)
	: Transverse_Ising<Field, Parameter>(num_spin, N_t, tau, dim_para, dim_para_each_direction, hbar)
{
	if (dim_para != (2 * num_spin + 1)*dim_para_each_direction)
		std::cout << "Dimension of parametric space does not fit for local control with the most freedoms!" << std::endl;
	//S of each spin
	S_each = new Deng::Col_vector<arma::Mat<Field>>[num_spin];

	for (unsigned int i = 0; i < num_spin; ++i)
	{
		S_each[i].set_size(3);
		for (unsigned int j = 0; j < 3; ++j)
		{
			S_each[i][j].set_size(this->_N, this->_N);
			S_each[i][j].zeros();
		}
	}

	for (unsigned int i = 0; i < 3; ++i)
	{
		//calculate the total S alone 3 directions
		for (unsigned int j = 0; j < num_spin; ++j)
		{
			arma::Mat<Field> temp_B = j == 0 ? this->S[i] : this->S_identity;
			for (unsigned int k = 1; k < num_spin; ++k)
			{
				temp_B = arma::kron(temp_B, j == k ? this->S[i] : this->S_identity);
			}
			S_each[j][i] = temp_B;
		}
	}
}

template<typename Field, typename Parameter>
Deng::Col_vector<Parameter> Transverse_Ising_Local_Control<Field, Parameter>::local_control_field(Parameter t, unsigned int ith_spin) const
{
	Deng::Col_vector<Parameter> ctrl(3);

	//static const unsigned int dim_para_each_site = 2 * this->_dim_para_each_direction;
	if (ith_spin < this->_num_spin)
	{
		ctrl[0] = 0.0;
		ctrl[1] = this->control_field_component(t, (2 * ith_spin + 0)*this->_dim_para_each_direction);
		ctrl[2] = this->control_field_component(t, (2 * ith_spin + 1)*this->_dim_para_each_direction);
	}
	else if(ith_spin == this->_num_spin)
	{
		ctrl[0] = this->control_field_component(t, 2 * this->_num_spin *this->_dim_para_each_direction);
		ctrl[1] = 0.0;
		ctrl[2] = 0.0;
	}
	else
	{
		assert(false && "Wrong dimension in local_control_field");
	}

	return ctrl;
}

template<typename Field, typename Parameter>
arma::Mat<Field> Transverse_Ising_Local_Control<Field, Parameter>::H_control(const Parameter t) const
{
	arma::Mat<Field> h_c(this->_N, this->_N, arma::fill::zeros);

	for (unsigned int i = 0; i <this->_num_spin; ++i)
	{
		h_c += local_control_field(t, i) ^ S_each[i];
	}
	h_c += local_control_field(t, this->_num_spin) ^ this->S_total;

	return h_c;
}

template<typename Field, typename Parameter>
Deng::Col_vector<arma::Mat<Field>> Transverse_Ising_Local_Control<Field, Parameter>::Dynamics(Parameter t) const
{
	Deng::Col_vector<arma::Mat<Field> > iH_and_partial_H(this->_dim_para + 1);

	iH_and_partial_H[0] = this->Dynamics_U(t);

	for (unsigned int i = 1; i <= (2*this->_num_spin*this->_dim_para_each_direction); ++i)
	{
		unsigned int spin_index = (i - 1) / (2 * this->_dim_para_each_direction);
		//could be generalize?
		//double original_para = parameters[i];
		this->parameters(i - 1) += 0.0078125;
		Deng::Col_vector<Parameter> partial_control = local_control_field(t, spin_index);
		this->parameters(i - 1) -= 0.0078125;
		partial_control = 128.0*(partial_control - local_control_field(t, spin_index));

		//iH_and_partial_H[i] = (-imag_i)*(partial_control^S_each[spin_index]);
		iH_and_partial_H[i] = this->minus_i_over_hbar*(partial_control^S_each[spin_index]);
	}
	for (unsigned int i = (2 * this->_num_spin * this->_dim_para_each_direction)+1; i <= this->_dim_para; ++i)
	{
		//could be generalize?
		//double original_para = parameters[i];
		this->parameters(i - 1) += 0.0078125;
		Parameter partial_control = this->control_field_component(t, this->_dim_para - this->_dim_para_each_direction);
		this->parameters(i - 1) -= 0.0078125;
		partial_control = 128.0*(partial_control - this->control_field_component(t, this->_dim_para - this->_dim_para_each_direction));

		//iH_and_partial_H[i] = (-imag_i)*(partial_control*this->S_total[0]);
		iH_and_partial_H[i] = this->minus_i_over_hbar*(partial_control*this->S_total[0]);
	}

	return iH_and_partial_H;
}
