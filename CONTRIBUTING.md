## Exposing function and classes to Python
You can find full documentation about exposing on [pybind11 official page](https://pybind11.readthedocs.io/en/stable/basics.html), in this topic we just show code style recommendations:
1. Provide comments to the exposed functions and classes (copy them from the C++ code)
2. Provide the argument names and default values as in the C++ code
3. Use `MR::decorateExpected` if you expose a function that returns an `MR::Expected` value
4. For default arguments that have a C++ type use `pybind11::arg_v( arg_name, arg_value, "PythonArgValue" )` to get correct autocompletion
5. If an exposed class is used (as an argument or return type) by other classes and functions, consider declaring it with `MR_ADD_PYTHON_CUSTOM_CLASS` and `MR_PYTHON_CUSTOM_CLASS` macros
6. Add tests to the `test_python` folder for the exposed functions

Example:
```c++
// example class for exposing
class A
{
public:
    // example field 1
    int a;
    // example field 2
    float b;
    // example empty function
    void doSomething( int exampleArgA, int exampleArgB = 1 ){}
}

// if a.a is not zero
MR::Expected<MR::Mesh> foo(const A& a)
{
    if (a.a != 0) return MR::Mesh{};
    return unexpected( "a is zero" );
}

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, A, ::A )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, classA, [] ( pybind11::module_& m )
{
    MR_PYTHON_CUSTOM_CLASS( A ).doc() =
        "example class for exposing";
    MR_PYTHON_CUSTOM_CLASS( A ).
        def( pybind11::init<>() ).
        def_readwrite( "a", &A::a, "example field 1" ).
        def_readwrite( "b", &A::b, "example field 2" ).
        def( "doSomething", &A::doSomething, pybind11::arg( "exampleArgA" ), pybind11::arg( "exampleArgB" ) = 1, " example empty function" );

    m.def( "foo", MR::decorateExpected( &foo ), pybind11::arg( a ), "if a.a is not zero" );
} )
```

```python
from helper import *
import pytest


def test_a_exposing():
    a = mrmresh.A()
    a.a = 1
    a.doSomething(0)
    resMesh = mrmesh.foo(a)
    # usually here some assertions are placed about computation results
    assert (True)
```

You can find examples in `source/mrmeshpy` and `source/mrmeshnumpy`
