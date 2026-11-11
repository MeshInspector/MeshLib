## Code style

* Indentation is 4 spaces. Braces on a separate line.

  ```cpp
  int foo()
  {
  ⸱⸱⸱⸱return 42;
  }
  ```

* Namespaces are not indented and should have a comment at the closing brace:

  ```cpp
  namespace MR
  {
  void foo() {}
  } // namespace MR
  ```

* Add spaces inside non-empty `(...)` and `{...}`:

  ```cpp
  int x = ( a + b ) / blah( c, d ) * foo();
  ```
  ```cpp
  int foo() { return 42; }
  ```

* Naming style:

  * Namespaces, types: `FooBar`.<br/>
    `namespace detail` is an exception, use it for things intended for internal use.

  * Functions, variables: `fooBar`.

  * Macros: `MR_FOO_BAR`

  * We never use `foo_bar`.

  * Non-public members (functions, variables, etc) should be suffixed with `_`, e.g. `fooBar_`.

  * Constant variables should be prefixed with `c`, e.g. `cFooBar`.

  * Enum constants are usually named `FooBar`, but some enums use other conventions.

## Export macros

Each project (`source/MRFoo/`) normally has its own `exports.h` that declares following macros:

* `MRFOO_API` — use it on all public functions in the headers.
* `MRFOO_CLASS` — sometimes you need to use this on classes and enums.

  If you're applying `typeid` to a class (or enum!) declared in a header, that class/enum **must** be marked `MRFOO_CLASS`, or on Arm Macs `typeid(...)` will return different IDs for it in different shared libraries, causing all sorts of issues.

CMake automatically defines `MRFoo_EXPORTS` when building each project, this is used in `exports.h`.

## Generated language bindings

We parse our headers to automatically generate Python/C/C# bindings, so you need to follow some additional rules when writing the headers. Consult [the bindings manual](./scripts/mrbind/README-coding.md) for that.

## Misc

### Flag enums

Flag enums should look like this:
```cpp
enum class MyFlags
{
    A = 1 << 0,
    B = 1 << 1,
    C = 1 << 2,
};
MR_MAKE_FLAG_OPERATORS( MyFlags )
```

The macro generates all the appropriate operators. Don't forget to include `<MRMesh/MRFlagOperators.h>`.

## Tests

A change in the library should come with a test that fails without it. There are three kinds of tests.

### C++ unit tests

They live in `source/MRTest/` and use GoogleTest.

* Put the tests of one topic in a file named `MR<Topic>Tests.cpp`. CMake takes all `*.cpp` files of the folder, but the Visual Studio project lists them, so add a new file to `MRTest.vcxproj` and `MRTest.vcxproj.filters` as well.
* The test suite is named after the library being tested, and the test after what it checks:

  ```cpp
  TEST( MRMesh, PolylineMaxDistance )
  {
      const Polyline3 a( Contour3f{ { 0, 0, 0 }, { 1, 0, 0 } } );
      EXPECT_EQ( findMaxDistanceSq( a, a ), 0.0f );
  }
  ```

* A test makes its input itself (`makeCube`, `makeUVSphere`, ... or a string with file content read through `std::istringstream`) and does not depend on files in the repository. If it needs files, it writes them in `UniqueTemporaryFolder`.
* Check what must be true on every platform: floating-point results differ in the last bits between compilers and processors, so compare them with a tolerance (`EXPECT_NEAR`), unless the exact value is guaranteed.
* The tests run in CI in both Release and Debug builds, where `assert`s are active, so a test must not pass wrong arguments to the library even if a Release build tolerates them.
* The code of an optional component is tested inside the same guard as the component itself, e.g. `#ifndef MRIOEXTRAS_NO_LAS`.
* To run the tests, start `MRTest` from the folder with the built binaries. `--gtest_filter=MRMesh.Polyline*` selects the tests to run, and `--with-python-tests` also checks the embedded Python.

### Python sanity tests

`test_python/` has small self-contained tests of the Python bindings; they also make their input themselves. See [test_python/readme.md](./test_python/readme.md) for how to run them.

### Python regression tests

`test_regression/` has the tests that compare results with reference data from the `test_data` submodule, for example the meshes from fixed issues. A new reference file is added in the repository of the submodule first, and then the pull request with the test updates the submodule to that commit.
