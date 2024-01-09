## Exopsing function and classes to python
You can find full documentation about exposing on [pybind11 official page](https://pybind11.readthedocs.io/en/stable/basics.html), in this topic we just show code style recommendations:
1. Provide comments to exposed functions and class (copy them from c++ code)
2. Provide argument names and default values as in c++ code
3. Use `MR::decorateExpected` if you expose function that returns `MR::Expected` value
4. Add tests to `test_python` folder for newly exposed functions

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

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, classA, [] ( pybind11::module_& m )
{
    pybind11::class_<A>( m, "A", "example class for exposing" ).
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
