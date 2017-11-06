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


20171011-00:01
-for armadillo, need to add armadillo include folder to Project Property/Visual C\C++/Include Directory
-also add the path of lapack and blas into Linker Directory
-Add .lib files as additional dependencies (for C::B .dll works)
-Put .dll to the same dir with .exe


20171014-20:49
-definitely need to rewrite the dependence
-need to add an abstract class of target function, with at least a virtual function called target function
  *it could be functor, but considering I may have target functions with known gradient (for example in conjugate gradient), i may not do so
  *need to have another abstract class of target function and negative gradient, waited to be instanciation(??? is this the right term?)
  *use mutable!!!!!!!
-may also need to add a 1d search virtual class. I still did not figure out whether it is needed in general, or just in conjugate gradient method
-make the optimization template
-minor revisions for GOAT namespace
  *change Hamiltonian& to a member pointer
  *use mutable!!!!!!!
  
20171030-16:05
-may need to add Chebychev propagation for time evolution


20171106-2347
-x,y constant mag field are useless!
