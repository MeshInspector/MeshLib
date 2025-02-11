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

## Python bindings

We parse our headers to automatically generate Python bindings, so you need to follow some additional rules when writing the headers. Consult [the bindings manual](./scripts/mrbind/README-coding.md) for that.
