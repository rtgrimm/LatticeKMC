# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _Nano
else:
    import _Nano

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class IVec3D(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    x = property(_Nano.IVec3D_x_get, _Nano.IVec3D_x_set)
    y = property(_Nano.IVec3D_y_get, _Nano.IVec3D_y_set)
    z = property(_Nano.IVec3D_z_get, _Nano.IVec3D_z_set)
    dim = _Nano.IVec3D_dim

    def V(self):
        return _Nano.IVec3D_V(self)

    def __eq__(self, rhs):
        return _Nano.IVec3D___eq__(self, rhs)

    def __ne__(self, rhs):
        return _Nano.IVec3D___ne__(self, rhs)

    def wrap(self, size):
        return _Nano.IVec3D_wrap(self, size)

    def __init__(self):
        _Nano.IVec3D_swiginit(self, _Nano.new_IVec3D())
    __swig_destroy__ = _Nano.delete_IVec3D

# Register IVec3D in _Nano:
_Nano.IVec3D_swigregister(IVec3D)


def nearest(loc):
    return _Nano.nearest(loc)

def vector_data(*args):
    return _Nano.vector_data(*args)
class EnergyMap(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    beta = property(_Nano.EnergyMap_beta_get, _Nano.EnergyMap_beta_set)

    def get_type_count(self):
        return _Nano.EnergyMap_get_type_count(self)

    def get_particle_type(self, index):
        return _Nano.EnergyMap_get_particle_type(self, index)

    def add_particle_type(self, id):
        return _Nano.EnergyMap_add_particle_type(self, id)

    def reset(self):
        return _Nano.EnergyMap_reset(self)

    def set_uniform_binary(self, interaction, field):
        return _Nano.EnergyMap_set_uniform_binary(self, interaction, field)

    def set_interaction(self, i, j, J):
        return _Nano.EnergyMap_set_interaction(self, i, j, J)

    def get_interaction(self, i, j):
        return _Nano.EnergyMap_get_interaction(self, i, j)

    def get_field(self, i):
        return _Nano.EnergyMap_get_field(self, i)

    def set_field(self, i, B):
        return _Nano.EnergyMap_set_field(self, i, B)

    def __init__(self):
        _Nano.EnergyMap_swiginit(self, _Nano.new_EnergyMap())
    __swig_destroy__ = _Nano.delete_EnergyMap

# Register EnergyMap in _Nano:
_Nano.EnergyMap_swigregister(EnergyMap)

class Particle(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    invalid_particle = _Nano.Particle_invalid_particle
    type = property(_Nano.Particle_type_get, _Nano.Particle_type_set)

    def __init__(self):
        _Nano.Particle_swiginit(self, _Nano.new_Particle())
    __swig_destroy__ = _Nano.delete_Particle

# Register Particle in _Nano:
_Nano.Particle_swigregister(Particle)

class RandomGenerator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    gen = property(_Nano.RandomGenerator_gen_get, _Nano.RandomGenerator_gen_set)

    def __init__(self, seed):
        _Nano.RandomGenerator_swiginit(self, _Nano.new_RandomGenerator(seed))
    __swig_destroy__ = _Nano.delete_RandomGenerator

# Register RandomGenerator in _Nano:
_Nano.RandomGenerator_swigregister(RandomGenerator)

class Lattice(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    size = property(_Nano.Lattice_size_get)
    energy_map = property(_Nano.Lattice_energy_map_get, _Nano.Lattice_energy_map_set)

    def __init__(self, size_):
        _Nano.Lattice_swiginit(self, _Nano.new_Lattice(size_))

    def particle_at(self, loc):
        return _Nano.Lattice_particle_at(self, loc)

    def site_energy(self, center, type):
        return _Nano.Lattice_site_energy(self, center, type)

    def uniform_init(self, random_generator):
        return _Nano.Lattice_uniform_init(self, random_generator)

    def get_types(self):
        return _Nano.Lattice_get_types(self)
    __swig_destroy__ = _Nano.delete_Lattice

# Register Lattice in _Nano:
_Nano.Lattice_swigregister(Lattice)

Type_Swap = _Nano.Type_Swap
Type_Nothing = _Nano.Type_Nothing
class Dispatch(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    @staticmethod
    def execute(type, lattice, center, target):
        return _Nano.Dispatch_execute(type, lattice, center, target)

    @staticmethod
    def calc_rate(type, lattice, center, target):
        return _Nano.Dispatch_calc_rate(type, lattice, center, target)

    def __init__(self):
        _Nano.Dispatch_swiginit(self, _Nano.new_Dispatch())
    __swig_destroy__ = _Nano.delete_Dispatch

# Register Dispatch in _Nano:
_Nano.Dispatch_swigregister(Dispatch)

def Dispatch_execute(type, lattice, center, target):
    return _Nano.Dispatch_execute(type, lattice, center, target)

def Dispatch_calc_rate(type, lattice, center, target):
    return _Nano.Dispatch_calc_rate(type, lattice, center, target)

class SimulationParams(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    allowed_events = property(_Nano.SimulationParams_allowed_events_get, _Nano.SimulationParams_allowed_events_set)

    def default_events(self):
        return _Nano.SimulationParams_default_events(self)

    def __init__(self):
        _Nano.SimulationParams_swiginit(self, _Nano.new_SimulationParams())
    __swig_destroy__ = _Nano.delete_SimulationParams

# Register SimulationParams in _Nano:
_Nano.SimulationParams_swigregister(SimulationParams)

class Event(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    rate = property(_Nano.Event_rate_get, _Nano.Event_rate_set)
    type = property(_Nano.Event_type_get, _Nano.Event_type_set)
    center = property(_Nano.Event_center_get, _Nano.Event_center_set)
    target = property(_Nano.Event_target_get, _Nano.Event_target_set)

    def __init__(self):
        _Nano.Event_swiginit(self, _Nano.new_Event())
    __swig_destroy__ = _Nano.delete_Event

# Register Event in _Nano:
_Nano.Event_swigregister(Event)

class EventCache(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, size):
        _Nano.EventCache_swiginit(self, _Nano.new_EventCache(size))

    def empty(self):
        return _Nano.EventCache_empty(self)

    def add_event(self, event):
        return _Nano.EventCache_add_event(self, event)

    def clear_location(self, center):
        return _Nano.EventCache_clear_location(self, center)

    def choose_event(self, generator):
        return _Nano.EventCache_choose_event(self, generator)
    __swig_destroy__ = _Nano.delete_EventCache

# Register EventCache in _Nano:
_Nano.EventCache_swigregister(EventCache)

class Simulation(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    params = property(_Nano.Simulation_params_get, _Nano.Simulation_params_set)
    lattice = property(_Nano.Simulation_lattice_get, _Nano.Simulation_lattice_set)
    random_generator = property(_Nano.Simulation_random_generator_get, _Nano.Simulation_random_generator_set)

    def __init__(self, params_, lattice, randomGenerator):
        _Nano.Simulation_swiginit(self, _Nano.new_Simulation(params_, lattice, randomGenerator))

    def step(self):
        return _Nano.Simulation_step(self)

    def multi_step(self, count):
        return _Nano.Simulation_multi_step(self, count)

    def get_time(self):
        return _Nano.Simulation_get_time(self)
    __swig_destroy__ = _Nano.delete_Simulation

# Register Simulation in _Nano:
_Nano.Simulation_swigregister(Simulation)

class Metropolis(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, lattice_, rand_gen_):
        _Nano.Metropolis_swiginit(self, _Nano.new_Metropolis(lattice_, rand_gen_))

    def get_lattice(self):
        return _Nano.Metropolis_get_lattice(self)

    def acceptance_rate(self):
        return _Nano.Metropolis_acceptance_rate(self)

    def step(self, steps=1):
        return _Nano.Metropolis_step(self, steps)
    __swig_destroy__ = _Nano.delete_Metropolis

# Register Metropolis in _Nano:
_Nano.Metropolis_swigregister(Metropolis)

class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _Nano.delete_SwigPyIterator

    def value(self):
        return _Nano.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _Nano.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _Nano.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _Nano.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _Nano.SwigPyIterator_equal(self, x)

    def copy(self):
        return _Nano.SwigPyIterator_copy(self)

    def next(self):
        return _Nano.SwigPyIterator_next(self)

    def __next__(self):
        return _Nano.SwigPyIterator___next__(self)

    def previous(self):
        return _Nano.SwigPyIterator_previous(self)

    def advance(self, n):
        return _Nano.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _Nano.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _Nano.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _Nano.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _Nano.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _Nano.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _Nano.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _Nano:
_Nano.SwigPyIterator_swigregister(SwigPyIterator)

class IntVecData(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    ptr = property(_Nano.IntVecData_ptr_get, _Nano.IntVecData_ptr_set)
    size = property(_Nano.IntVecData_size_get, _Nano.IntVecData_size_set)

    def __init__(self):
        _Nano.IntVecData_swiginit(self, _Nano.new_IntVecData())
    __swig_destroy__ = _Nano.delete_IntVecData

# Register IntVecData in _Nano:
_Nano.IntVecData_swigregister(IntVecData)

class DoubleVecData(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    ptr = property(_Nano.DoubleVecData_ptr_get, _Nano.DoubleVecData_ptr_set)
    size = property(_Nano.DoubleVecData_size_get, _Nano.DoubleVecData_size_set)

    def __init__(self):
        _Nano.DoubleVecData_swiginit(self, _Nano.new_DoubleVecData())
    __swig_destroy__ = _Nano.delete_DoubleVecData

# Register DoubleVecData in _Nano:
_Nano.DoubleVecData_swigregister(DoubleVecData)

class IntVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _Nano.IntVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _Nano.IntVector___nonzero__(self)

    def __bool__(self):
        return _Nano.IntVector___bool__(self)

    def __len__(self):
        return _Nano.IntVector___len__(self)

    def __getslice__(self, i, j):
        return _Nano.IntVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _Nano.IntVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _Nano.IntVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _Nano.IntVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _Nano.IntVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _Nano.IntVector___setitem__(self, *args)

    def pop(self):
        return _Nano.IntVector_pop(self)

    def append(self, x):
        return _Nano.IntVector_append(self, x)

    def empty(self):
        return _Nano.IntVector_empty(self)

    def size(self):
        return _Nano.IntVector_size(self)

    def swap(self, v):
        return _Nano.IntVector_swap(self, v)

    def begin(self):
        return _Nano.IntVector_begin(self)

    def end(self):
        return _Nano.IntVector_end(self)

    def rbegin(self):
        return _Nano.IntVector_rbegin(self)

    def rend(self):
        return _Nano.IntVector_rend(self)

    def clear(self):
        return _Nano.IntVector_clear(self)

    def get_allocator(self):
        return _Nano.IntVector_get_allocator(self)

    def pop_back(self):
        return _Nano.IntVector_pop_back(self)

    def erase(self, *args):
        return _Nano.IntVector_erase(self, *args)

    def __init__(self, *args):
        _Nano.IntVector_swiginit(self, _Nano.new_IntVector(*args))

    def push_back(self, x):
        return _Nano.IntVector_push_back(self, x)

    def front(self):
        return _Nano.IntVector_front(self)

    def back(self):
        return _Nano.IntVector_back(self)

    def assign(self, n, x):
        return _Nano.IntVector_assign(self, n, x)

    def resize(self, *args):
        return _Nano.IntVector_resize(self, *args)

    def insert(self, *args):
        return _Nano.IntVector_insert(self, *args)

    def reserve(self, n):
        return _Nano.IntVector_reserve(self, n)

    def capacity(self):
        return _Nano.IntVector_capacity(self)
    __swig_destroy__ = _Nano.delete_IntVector

# Register IntVector in _Nano:
_Nano.IntVector_swigregister(IntVector)

class DoubleVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _Nano.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _Nano.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _Nano.DoubleVector___bool__(self)

    def __len__(self):
        return _Nano.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _Nano.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _Nano.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _Nano.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _Nano.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _Nano.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _Nano.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _Nano.DoubleVector_pop(self)

    def append(self, x):
        return _Nano.DoubleVector_append(self, x)

    def empty(self):
        return _Nano.DoubleVector_empty(self)

    def size(self):
        return _Nano.DoubleVector_size(self)

    def swap(self, v):
        return _Nano.DoubleVector_swap(self, v)

    def begin(self):
        return _Nano.DoubleVector_begin(self)

    def end(self):
        return _Nano.DoubleVector_end(self)

    def rbegin(self):
        return _Nano.DoubleVector_rbegin(self)

    def rend(self):
        return _Nano.DoubleVector_rend(self)

    def clear(self):
        return _Nano.DoubleVector_clear(self)

    def get_allocator(self):
        return _Nano.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _Nano.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _Nano.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        _Nano.DoubleVector_swiginit(self, _Nano.new_DoubleVector(*args))

    def push_back(self, x):
        return _Nano.DoubleVector_push_back(self, x)

    def front(self):
        return _Nano.DoubleVector_front(self)

    def back(self):
        return _Nano.DoubleVector_back(self)

    def assign(self, n, x):
        return _Nano.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _Nano.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _Nano.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _Nano.DoubleVector_reserve(self, n)

    def capacity(self):
        return _Nano.DoubleVector_capacity(self)
    __swig_destroy__ = _Nano.delete_DoubleVector

# Register DoubleVector in _Nano:
_Nano.DoubleVector_swigregister(DoubleVector)



