# GOAT-Framework

This project is a generic code to do quantum optimal control (QOC) in a way GOAT describes.
It is written in C++ and based on certain linear algebra libraries.
Currently it uses templates from Armadillo, and hopefully supports for Eigen will also be added later.

Intel MKL could be used to boost the performance. 

20171007-10:17
-Previously added inner product for block vectors. Only work for real/Hermitian vectors
-Plan to add Monte Carlo related functions
-Going to have a Metropolis and Ising model, together with several RNG in C++ style. 
-May update the RNG objects in an aggregation manner
