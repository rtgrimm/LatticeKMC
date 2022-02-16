%module Nano
%{
#include "Native/base.hpp"
#include "Native/interop.hpp"
#include "Native/lattice.hpp"
#include "Native/kmc.hpp"
#include "Native/metropolis.hpp"
%}


%include "Native/base.hpp"
%include "Native/interop.hpp"
%include "Native/lattice.hpp"
%include "Native/kmc.hpp"
%include "Native/metropolis.hpp"

%include "typemaps.i"
%include "std_vector.i"
%include "std_string.i"
%include "std_array.i"
%include "stdint.i"


%template(IntVecData) Nano::Interop::VecData<int32_t>;
%template(DoubleVecData) Nano::Interop::VecData<double>;

%template(IntVector) std::vector<int32_t>;
%template(DoubleVector) std::vector<double>;
%template(IVec3DVector) std::vector<Nano::IVec3D>;