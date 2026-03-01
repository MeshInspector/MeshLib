# How to maintain bindings and write binding-compatible code?

## Which projects are parsed?

Open [`scripts/mrbind/generate.mk`](./generate.mk) and search for `_InputProjects`. You can modify it to control which projects are parsed, and how they are grouped into modules.

Right now we parse `MRMesh` and some other projects to generate `mrmeshpy`, and separately parse `MRCuda` for `mrcudapy`.

Right now `mrmeshpy` also has old handwritten bindings that we don't release.

`MRViewer` is not parsed yet, it uses handwritten bindings that you can find in `source/mrviewerpy`.

Following things are not parsed and have no Python bindings: plugins (`MRCommonPlugins`), executables (`MRViewerApp`, `MRTest`), C and C# bindings (`MRMeshC`, `MRDotNet`), Python helpers (`MRPython`).

## Excluding things from bindings

### Namespaces

If the namespace with the same name in several headers, add `--ignore MR::MyNamespace` to [`scripts/mrbind/mrbind_flags.txt`](./mrbind_flags.txt).

If the namespace only appears in one place, you can alternatively mark it with `MR_BIND_IGNORE` (which is defined in `MRPch/MRBindingMacros.h`):
```cpp
namespace MR_BIND_IGNORE Blah {...}
```


### Functions, classes, etc

The recommended approach is to mark them with `MR_BIND_IGNORE`. This macro is defined in `MRPch/MRBindingMacros.h`. For example:

```cpp
MR_BIND_IGNORE void foo();

class MR_BIND_IGNORE Blah {...};

using Foo MR_BIND_IGNORE = ...;

struct A
{
    MR_BIND_IGNORE int x; // Skip a specific member.
};
```

Other options are:

* You can move them to `namespace MR::detail` if they are for internal use and even C++ users shouldn't use them.

* If this is a class member, you can make it non-`public`.

* You can also add `--ignore ...` to [`scripts/mrbind/mrbind_flags.txt`](./mrbind_flags.txt) to skip a certain entity by name, e.g. `--ignore MR::Blah`. But prefer `MR_BIND_IGNORE` when possible.

  `--ignore` is most useful with regular expressions, e.g. to skip all specializations of a certain template: `--ignore '/MR::Blah<.*>/'`.

* Base classes can't be marked with `MR_BIND_IGNORE`, but you can add `--skip-mentions-of` to [`scripts/mrbind/mrbind_flags.txt`](./mrbind_flags.txt) to skip them. This is useful when you inherit from library classes that can't be parsed for some reason.

* If you wish to ignore e.g. all fields with of certain type, or all functions with a certain parameter type, you can write complex rules using templates in [`scripts/mrbind/mrbind_target_pybind11.h`](./mrbind_target_pybind11.h).

Solutions that are **not** recommended:

* You can use `#if !MR_PARSING_FOR_PB11_BINDINGS`, but that's **not** recommended, since if any of that code is used in another header, the parser will fail on that.

* You can add your header to [`scripts/mrbind/input_file_blacklist.h`](./input_file_blacklist.txt), but that's **not** recommended, because if another non-blacklisted header will include yours, it will still be parsed.

## Adding new things only to the bindings

### Python aliases

You can add Python aliases for functions, classes, and even non-static class members.

Add them to [`scripts/mrbind/aliases.cpp`](./aliases.cpp).

### Python helper functions

You can add extra C++ functions to the bindings by adding them to [`scripts/mrbind/helpers.cpp`](./helpers.cpp).

## Customizing the type names

Adding a `using` alias for a type will add that alias in Python too.

If you want to change the primary name of a type in Python, this is only supported for templates, via [`MR_CANONICAL_TYPEDEFS(...)`](../../source/MRMesh/MRCanonicalTypedefs.h). Those usually should be added `MRMeshFwd.h` (if the class is from `MRMesh`) or elsewhere.

## How the C++ code has to be written

There are some limitations on how you must write your C++ code for the bindings to work.

### Export macros

Every function in a header in a parsed project needs to have the export macro on it (e.g. `MRMESH_API` in MRMesh, and similarly in other projects). Not doing this will cause `undefined reference`s in the bindings.

If a function isn't for public use (so you don't want to export it), move it to `namespace MR::detail`.

### No incomplete types

Bindings require types to be "complete" (the definition to be visible) even when C++ doesn't. E.g.:

```cpp
struct A; // Not a definition -> `A` is incomplete.

void foo(A *a);
```
This is ok in normal C++, but not ok for the bindings.

This will sometimes work, because all headers are combined into one big header and parsed together, so the missing types can happen to be defined in another header.

To fix this:

* If the function is **impossible to call** without this type being complete, add the missing `#include` that defines it. This is a good practice.

* If the function **can** be called without it and you don't want that include, you can wrap it in `#if MR_PARSING_FOR_PB11_BINDINGS`.

### Pointers are always assumed to point to single elements

Function parameters, return values, and class fields of pointer types are assumed to point to single objects, not arrays.

Eventually we want to support `std::span`, but for now use `std::vector` and `std::array` to pass arrays.

Note that function parameters that look like arrays: `void foo(int a[42])` (or `[]` without size) are actually pointers in C++, equivalent to `void foo(int *a)`.

## Lifetime annotations / keep-alive

If you're storing raw pointers or references in your classes, you have the risk of them dangling. Either due to user error, or in C# due to the compiler destroying local variables too early, incorrectly assuming that they aren't needed anymore.

To solve this, each C# and Python object stores a list of other C#/Python objects that it needs to "keep alive".

To automatically fill those lists, functions need to be annotated with special macros, to indicate the relationships between the parameters, `this`, and the return value: which references get stored where, etc.

Some (but not all) of those macros also help Clang emit useful warnings.

The annotations are deduced automatically for:

* Constructors taking references (we assume the resulting object stores that reference), other than copy/move constructors.

  This also happens for the implicitly generated constructors of aggregate types (structs that don't have custom constructors and are initialized with a list of their elements).

* Iterators: `begin()`/`end()`/`operator*`.

* The handwritten custom bindings, such as those for the standard containers.

The annotations need to be specified manually for any function that does any of the following:

* Returns a class reference (unless it refers to a global variable).

* Takes a class reference and stores it somewhere (perhaps in `this` or in another parameter).

Raw pointers need the same treatment as references. Classes storing references or raw pointers also need the same treatment, even if passed around by value.

In particular, custom containers like `MR::Vector` need those annotations on some of their methods. E.g. when inserting an object into a container, the original C#/Python object is lost, and its keep-alive list is lost with it, unless the `push_back()` function of the container is annotated to copy the element (or its list) into the list of the container itself.

The annotations will be silently ignored if applied to an incorrect type, which is convenient in templates. E.g. `MR::Vector<int>` will ignore all the annotations, but `MR::Vector<MR::Mesh>` will benefit from them.

Forgetting to specify those might not cause any issues for a while, and then blow up when the C# compiler feels like optimizing out your variables.

### How to annotate the functions?

For each function, the annotations are stored internally as a list of pairs `(a, b)`, and for each pair we do `a._KeepAlive.push_back(b)` (where `_KeepAlive` is the hidden keep-alive list in `a`).

`a` and `b` can be any of:

* Any function parameter.
* `this`
* The return value (only allowed for `a`, not `b`).

To add a pair, you must add a macro to `b`, and which macro to add depends on `a`. Include `<MRMesh/MRMacros.h>` to get access to the macros.

E.g. when `a` is the return value, the macro is `MR_LIFETIMEBOUND`:

* If `b` is a parameter, the macro must be added right after the parameter name:

  ```cpp
  MR::Mesh& foo( MR::Mesh& m MR_LIFETIMEBOUND )
  {
      return m;
  }

  MR::Mesh& bar( MR::Mesh& m1 MR_LIFETIMEBOUND, MR::Mesh& m2 MR_LIFETIMEBOUND )
  {
      if (...)
          return m1;
      else
          return m2;
  }
  ```
* If `b` is `this`, the macro must be added after the method parameter list: (and after `const` if any)
  ```cpp
  struct A
  {
      MR::Mesh m;

            MR::Mesh& getMesh()       MR_LIFETIMEBOUND { return m; }
      const MR::Mesh& getMesh() const MR_LIFETIMEBOUND { return m; }
  }
  ```

When `a` is a function parameter or `this`, the macro is `MR_LIFETIME_CAPTURE_BY(a)` (where `a` is that parameter name or `this`). And if `b` is `this`, you must use `MR_THIS_LIFETIME_CAPTURE_BY(a)` instead. E.g.:

```cpp
struct A
{
    MR::Mesh& mesh;
};

void foo( MR::Mesh& mesh MR_LIFETIME_CAPTURE_BY(vec), std::vector<A> &vec )
{
    vec.push_back( A{mesh} );
}

struct B
{
    std::vector<A> vec;

    void add( MR::Mesh& mesh MR_LIFETIME_CAPTURE_BY(this) )
    {
        vec.add( A{mesh} );
    }
}

struct C
{
    struct Ptr
    {
        C *c;
    };

    void bar(Ptr &ptr) MR_THIS_LIFETIME_CAPTURE_BY(ptr)
    {
        ptr.c = this;
    }
};
```

Incorrectly adding or not adding `THIS_` triggers an error, so you won't miss it.

The macros also exist in the `..._NESTED` variants. Those are used when you're not storing a reference to the object, but are instead copying it, but the copied object might store references to something else, so you want to copy the contents of its keep-alive list into the keep-alive list of your object. This is necessary e.g. for `MR::Vector<T>::push_back()`. For example:

```cpp
struct A
{
    MR::Mesh& mesh;
};

struct VecA
{
    std::vector<A> vec;

    void push_back( MR::Mesh& mesh MR_LIFETIME_CAPTURE_BY(this) )
    {
        vec.push_back( A{mesh} );
    }

    void push_back( A a MR_LIFETIME_CAPTURE_BY_NESTED(this) ) // Same for references to `A` as well.
    {
        vec.push_back( A{mesh} );
    }

    A& get_a() MR_LIFETIMEBOUND
    {
        return vec.front();
    }

    MR::Mesh& get_mesh() MR_THIS_LIFETIMEBOUND_NESTED
    {
        return vec.front().mesh;
    }
};
```

Notice that `MR_LIFETIMEBOUND_NESTED` and `MR_LIFETIME_CAPTURE_BY[_NESTED]` all have a `THIS` variant (you'll get an error if you incorrectly add or don't add `THIS`), but `MR_LIFETIMEBOUND` doesn't (it works in both places with the same syntax). This is sadly a technical limitation. (We could add `MR_THIS_LIFETIMEBOUND` with the same contents as `MR_LIFETIMEBOUND`, but there's no point.)

At the time of writing, all those annotations only affect C#, but they will be eventually ported to Python bindings too.

## How the C++ templates have to be written

Templates are more affected more than other things. There are some limitations on how you write them:

### Correct `requires` everywhere

The bindings parser will try to instantiate every template it sees, even if it's not called.

If a member function of a class template is only valid for **some** template arguments, the bindings will choke on it.

You must correctly annotate all your template functions with `requires`. Or, since not all of our platforms support C++20 at the moment, use `MR_REQUIRES_IF_SUPPORTED(...)` from `MRMesh/MRMacros.h`.

```cpp
template <typename T>
struct Pair
{
    T first, second;

    T sum() const MR_REQUIRES_IF_SUPPORTED( std::is_arithmetic_v<T> )
    {
        return first + second;
    }
};
```

This is normally only needed for class members. We don't instantiate free functions automatically, because it's hard to determine the right template arguments for that.

Of course, if literally **nothing** in the code instantiates the class with the wrong template arguments (e.g. if `Pair` in the example above is only instantiated with arithmetic types), you can omit the `requires`. But then it's a good practice to mark the entire class with `requires`.

### Manually instantiate classes and non-member functions

Non-member template functions have to be instantiated manually with all desired template arguments. We have a macro for that:

```cpp
template <typename T> T foo(T t) {...}
MR_BIND_TEMPLATE( int foo(int t) )     // Or `int foo<int>(int t)`
MR_BIND_TEMPLATE( float foo(float t) ) // Or `float foo<float>(float t)`
```
This is **not** needed if you already have an `extern template ...;` for this template in the header.

The same applies to class templates. But they are also instantiated automatically if a typedef is pointing to them: `using MyClassInt = MyClass<int>;`.

Alternatively, you can instantiate the templates in a separate file, [`mrbind/extra_headers/instantiate_templates.h`](./extra_headers/instantiate_templates.h). This is useful if you want to instantiate e.g. some class from `std`, e.g. `std::vector<MyType>`.

### Prefer `friend`-definitions to free functions

Prefer `friend`-definitions to free functions, because we automatically instantiate the friends.

```cpp
template <typename T>
struct A
{
    friend A operator+(A, A) {...}
};
```

This is better than making `template <typename T> A operator+(A, A) {...}` a free function, because you would have to `MR_BIND_TEMPLATE(...)` that free function, while the `friend` can be instantiated automatically by us.

This is a good practice in C++ anyway, because the friends can only be reached via ADL, meaning the compiler has to search through less functions.

## Missing bindings for `std::` classes and more in Python

There's a rare quirk that will sometimes happen to the Python bindings

Let's say you use `std::vector<std::vector<Blah>>` somewhere (e.g. as a function parameter). What can happen is that the binding for this type will be generated, but the one for `std::vector<Blah>` won't, rendering the generated one unusable and triggering Stubgen errors (`Invalid expression`, etc).

This is intentional behavior to speed up binding compilation, that we enable by defining `MB_PB11_NO_REGISTER_TYPE_DEPS`.

Fix this by manually instantiating the offending type (`std::vector<Blah>` in this example) [in `instantiate_templates.h` as was explained above](#manually-instantiate-classes-and-non-member-functions).
