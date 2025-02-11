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

* Base classes can't be marked with `MR_BIND_IGNORE`, but you can add `--skip-base` to [`scripts/mrbind/mrbind_flags.txt`](./mrbind_flags.txt) to skip them. This is useful when you inherit from library classes that can't be parsed for some reason.

* If you wish to ignore e.g. all fields with of certain type, or all functions with a certain parameter type, you can write complex rules using templates in [`scripts/mrbind/mrbind_target_pybind11.h`](./mrbind_target_pybind11.h).

Solutions that are **not** recommended:

* You can use `#if !MR_PARSING_FOR_PB11_BINDINGS`, but that's **not** recommended, since if any of that code is used in another header, the parser will fail on that.

* You can add your header to [`scripts/mrbind/input_file_blacklist.h`](./input_file_blacklist.txt), but that's **not** recommended, because if another non-blacklisted header will include yours, it will still be parsed.

## Adding new things only to the bindings

### Aliases

You can add Python aliases for functions, classes, and even non-static class members.

Add them to [`scripts/mrbind/aliases.cpp`](./aliases.cpp).

### Helper functions

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

This is a good practice in C++ anyway, because the friends can only be reaced via ADL, meaning the compiler has to search through less functions.

## Missing bindings for `std::` classes and more

There's a rare quirk that will sometimes happen.

Let's say you use `std::vector<std::vector<Blah>>` somewhere (e.g. as a function parameter). What can happen is that the binding for this type will be generated, but the one for `std::vector<Blah>` won't, rendering the generated one unusable and triggering Stubgen errors (`Invalid expression`, etc).

This is intentional behavior to speed up binding compilation, that we enable by defining `MB_PB11_NO_REGISTER_TYPE_DEPS`.

Fix this by manually instantiating the offending type (`std::vector<Blah>` in this example) [in `instantiate_templates.h` as was explained above](#manually-instantiate-classes-and-non-member-functions).
