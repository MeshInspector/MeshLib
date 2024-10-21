# Working with different measurement units

Basics: &nbsp; &nbsp; [Introduction](#introduction) — [Converting between units](#converting-between-units) — [Unit to string](#unit-to-string) — [Widgets](#widgets)

Advanced: &nbsp; &nbsp; [Creating a new measurement unit checklist](#creating-a-new-measurement-unit-checklist)

# Basics

### Introduction

We store all quantities as scalars (typically `float`s), there are no custom types to encode length/angle/etc.

Our convention is:
* Store angles in radians (converting to degrees in GUI).
* Store lengths as is (we append `mm` or `inch` suffix in GUI depending on the settings, without converting the value).
  * Same applies to area, volume, and movement speed, which is measured in mm/s or inches/s.
* Percentages should be stored as numbers between `0` and `1` (multiplied by 100 and displayed as percents in the GUI).

### Converting between units

You can use `convertUnits()` from `<MRViewer/MRUnits.h>` to convert between different units, e.g.:
```cpp
float a = convertUnits( AngleUnit::degrees, AngleUnit::radians, 180.f ); // a == pi
```
You usually don't need to call `convertUnits()` directly, prefer higher-level functions described below.

### Unit to string

Use `valueToString<E>( value )` from `<MRViewer/MRUnits.h>` to convert a value to a string.

Here `E` is one of: `LengthUnit`, `AngleUnit`, `NoUnit`, ... (see the list of types in `MRUnits.h`).

For example:
```cpp
std::string str = valueToString<AngleUnits>( PI_F ); // "180°"
```

See [Introduction](#introduction) for our default assumptions about units.

There's an optional parameter of type `UnitToStringParams<E>` with various settings. For example, you can override the input unit this way:

```cpp
// By default we assume that angles are stored as radians, but this
std::string str = valueToString<AngleUnits>( 180.f, { .sourceUnit = AngleUnit::degrees } ); // "180°"
```

By default `UnitToStringParams<E>` initializes each member to the global default, but you can override them individually. See the definition of that struct for for the full list of parameters.

Some useful parameters are:

* `.sourceUnit` (of type `std::optional<E>`) - The input measurement unit. Defaults to `AngleUnit::radians` for angles and to `std::nullopt` for everything else.

    When null, assumed to be the same as `.targetUnit`, so no conversion is performed.

* `.targetUnit` (of type `std::optional<E>`) - The output measurement unit.

    If this is null, no conversion is performed, and the unit name is taken from `sourceUnit` if any.

See the definition of `UnitToStringParams<E>` in `<MRViewer/MRUnits.h>` for the full list of parameters.

### Widgets

`<MRViewer/MRUIStyle.h>` defines a few measurement-unit-aware widgets:

* `UI::slider<E>( "Label", value )`
* `UI::drag<E>( "Label", value, min, max )` - also includes `+`/`-` buttons for integers like `ImGui::Input()`.
* `UI::readOnlyValue<E>( "Label", value )` - wraps `UI::inputTextCenteredReadOnly()`

Where:

* `E` is one of: `LengthUnit`, `AngleUnit`, `NoUnit`, ... (see the list of types in `MRUnits.h`).

  See [Introduction](#introduction) for our default assumptions about units.

* `value` is a reference to a scalar or `Vector2/3/4` or `ImVec2/4`.

Those automatically perform the measurement unit conversions, and display the units in the GUI. (Pass `E` = `NoUnit` if your value doesn't have a unit.)

NOTE: They are more sensitive to argument types. If the `value` is `float`, then `min`,`max` must also be `float`, not `int`.

They accept the same optional parameters as ImGui widgets, and additionally `UnitToStringParams<E>` to control the unit conversion. (See [Unit to string](#unit-to-string) for explanation.)

By default they clamp values that you input with Ctrl+Click to the same min/max bounds. (Pass `ImGuiSliderFlags_AlwaysClamp` by default.)

Some examples:

* Changing the default measurement unit assumptions:

  ```cpp
  static float f = PI_F / 2;
  // Assumes `f` is in radians, displays it as degrees.
  UI::slider<AngleUnit>( "Angle", f, 0.f, PI_F );

  static float fDeg = 90;
  // Displays `fDeg` directly as degrees.
  UI::slider<AngleUnit>( "Angle", fDeg, 0.f, 180.f, { .sourceUnit = AngleUnit::degrees } );
  ```

* Working with vectors:
  ```cpp
  static Vector3f v( 1, 2, 3 );
  // Same min/max for every component.
  UI::slider<LengthUnit>( "Size", v, 0.f, 10.f );
  // Different min/max per component.
  UI::slider<LengthUnit>( "Size", v, Vector3f( 0.f, 1.f, 2.f ), Vector3f( 10.f, 11.f, 12.f ) );
  ```

## Advanced

### Creating a new measurement unit checklist

#### Adding a unit to an existing enum

If you want to add a new unit to an existing enum: (say, add `LengthUnit::yards`)

* Add a new constant to the enum.

* In `MRViewer/MRUnits.cpp`, in the `getUnitInfo()` specialization for your unit, add an array element with information about your unit.

* If you're updating `LengthUnit` or anything that depends on length (such as `VolumeUnit`):

  * Add the respective units to all those enums (e.g. if you added `LengthUnit::yards`, add `VolumeUnit::yards3` and so on).
  * Add those units to `getDependentUnit()` in `MRUnitSettings.cpp`.

#### Creating a new unit enum

If you want to create a new unit enum: (say, `EnergyUnit { joules, electronvolts, ... }`)

* Add the enum to `MRViewer/MRUnits.h`, next to the other enums.
* Add your enum to the `#define DETAIL_MR_UNIT_ENUMS` macro in `MRViewer/MRUnits.h`.
* Add the default to-string parameters for your enum to `defaultUnitToStringParams` in `MRViewer/MRUnits.cpp`.

  The important thing you have to decide are: the default display unit and the default source code unit (which can be null to always match the display unit).

* Add a specialization of `getUnitInfo()` to `MRViewer/MRUnits.cpp` for your enum, next to the other ones.

* Add your enum to `forAll...Params()` in `MRUnitSettings.cpp` (choose `...` accordingly).

* If your enum depends on length (e.g. `VolumeUnit`), all units in your enum must have a respective unit in `LengthUnit`, and vice versa (e.g. `LengthUnit::mm` <-> `VolumeUnit::mm3`, `LengthUnit::meters` <-> `VolumeUnit::meters3`, etc). Then you must also update `getDependentUnit()` in `MRUnitSettings.cpp`.
