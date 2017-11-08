#include"transverse_Ising.hpp"

Transverse_Ising::Transverse_Ising(unsigned int num_spinor, unsigned int N_t, real tau, unsigned int dim_para, real hbar) :
	Deng::GOAT::Hamiltonian<elementtype, real>((1 << num_spinor), N_t, tau, dim_para), omega(2.0*Pi / tau), _hbar(hbar)
	//2 for the dimension of spin half system
{
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
	interaction.set_size(_N, _N);
	interaction.zeros();
	//calculate the interaction part
	for (unsigned int i = 0; i < (num_spinor - 1); ++i)
	{
		auto temp = i == 0 ? S[2] : S_identity;
		for (unsigned int j = 1; j < num_spinor; ++j)
		{
			temp = arma::kron(temp, (j == i) || (j == i + 1) ? S[2] : S_identity);
		}
		interaction -= temp;
	}

	//total S alone 3 directions
	S_total.set_size(3);
#ifdef ISING_CONTROL_INDIVIDUAL
	//S of each spin
	S_each = new Deng::Col_vector<arma::Mat<elementtype>>[num_spinor];
	for (unsigned int i = 0; i < num_spinor; ++i)
	{
		for (unsigned int j = 0; j < 3; ++j)
		{
			S_each[i][j].set_size(_N, _N);
			S_each[i][j].zeros();
		}
	}
#endif
	for (unsigned int i = 0; i < 3; ++i)
	{
		S_total[i].set_size(_N, _N);
		S_total[i].zeros();
		//calculate the total S alone 3 directions
		for (unsigned int j = 0; j < num_spinor; ++j)
		{
			auto temp_B = j == 0 ? S[i] : S_identity;
			for (unsigned int k = 1; k < num_spinor; ++k)
			{
				temp_B = arma::kron(temp_B, j == k ? S[i] : S_identity);
			}
#ifdef ISING_CONTROL_INDIVIDUAL
			S_each[j][i] = temp_B;
#endif	
			S_total[i] += temp_B;
		}
	}

}
Deng::Col_vector<real> Transverse_Ising::B(real t) const
{
	Deng::Col_vector<real> B_field(3);

	//rotating B field
	//B_field[0] = cos(0.25*omega*t)*B_x_max;
	//B_field[1] = B_y_max;
	//B_field[2] = sin(0.25*omega*t)*B_z_max;
	B_field[0] = B_x_max;// cos(0.25*omega*t)*B_x_max;
	B_field[1] = 0.0;// B_y_max;
	B_field[2] = B_z_max;// B_z_max*t / _tau;//sin(0.25*omega*t)*B_z_max;

	return B_field;
}
//calculate control magnetic field based on the parameters
Deng::Col_vector<real> Transverse_Ising::control_field(real t) const
{
	Deng::Col_vector<real> ctrl(3);
	ctrl[0] = 0.0;
	ctrl[1] = 0.0;
	ctrl[2] = 0.0;

	unsigned int mode = 0;
	unsigned int trig = 0;
	unsigned int direction = 0;

	for (unsigned int i = 0; i < _dim_para; ++i)
	{
		mode = i < 1 ? 0 : (i - 1) / 6 + 1;
		trig = (i - 1) % 6;
		direction = trig % 3;

		ctrl[direction] += trig < 3 ? parameters[i] * sin(mode * omega * t) : parameters[i] * cos(mode * omega * t);
	}
	return ctrl;
}
arma::Mat<elementtype> Transverse_Ising::H_0(real t) const
{
	//bracket is necessary as operator ^ is of low priority
	return interaction + (B(t) ^ S_total);
}
arma::Mat<elementtype> Transverse_Ising::H_control(real t) const
{
	return control_field(t) ^ S_total;
}
Deng::Col_vector<arma::Mat<elementtype> > Transverse_Ising::Dynamics(real t) const
{
	Deng::Col_vector<arma::Mat<elementtype> > iH_and_partial_H(_dim_para + 1);

	iH_and_partial_H[0] = Dynamics_U(t);

	for (unsigned int i = 1; i <= _dim_para; ++i)
	{
		//could be generalize?
		//double original_para = parameters[i];
		parameters[i - 1] += 0.01;
		Deng::Col_vector<real> partial_control = control_field(t);
		parameters[i - 1] -= 0.01;
		partial_control = (1 / 0.01)*(partial_control - control_field(t));

		iH_and_partial_H[i] = (-imag_i / _hbar)*(partial_control^S_total);
	}

	return iH_and_partial_H;
}
arma::Mat<elementtype> Transverse_Ising::Dynamics_U(real t) const
{
	return (-imag_i / _hbar)*(H_0(t) + H_control(t));
}