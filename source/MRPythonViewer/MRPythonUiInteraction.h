#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace MR::PythonViewer
{

/// UI entry type enum.
enum class UiEntryType
{
    button,
    group,
    valueInt,
    valueUint,
    valueReal,
    valueString,
};

struct UiEntry
{
    std::string name;
    UiEntryType type = UiEntryType::button;
    std::string status;
};

struct UiValueInt
{
    std::int64_t value = 0;
    std::int64_t min = 0;
    std::int64_t max = 0;
};

struct UiValueUint
{
    std::uint64_t value = 0;
    std::uint64_t min = 0;
    std::uint64_t max = 0;
};

struct UiValueReal
{
    double value = 0;
    double min = 0;
    double max = 0;
};

struct UiValueString
{
    std::string value;
    std::optional<std::vector<std::string>> allowed;
};

// `python_viewer.py` wraps the functions with a leading underscore, to return lists and tuples.

/// List existing UI entries at the specified path.
/// Pass an empty list to see top-level groups.
/// Add group name to the end of the vector to see its contents.
/// When you find the button you need, pass it to `uiPressButton()`.
[[nodiscard]] std::vector<UiEntry> _uiListEntries( const std::vector<std::string>& path );

/// Flat depth-first list of every UI entry in the subtree rooted at `rootPath`.
/// Pass an empty list for the whole tree.
/// Each element is a `(path, UiEntry)` tuple where `path[-1] == entry.name`.
[[nodiscard]] std::vector<std::pair<std::vector<std::string>, UiEntry>> _uiListAllEntries( const std::vector<std::string>& rootPath );

/// Simulate a button click. Use `uiListEntries()` to find button names.
void uiPressButton( const std::vector<std::string>& path );

/// Read a value from a drag/slider widget. This function is for signed integers.
[[nodiscard]] UiValueInt uiReadValueInt( const std::vector<std::string>& path );

/// Read a value from a drag/slider widget. This function is for unsigned integers.
[[nodiscard]] UiValueUint uiReadValueUint( const std::vector<std::string>& path );

/// Read a value from a drag/slider widget. This function is for real numbers.
[[nodiscard]] UiValueReal uiReadValueReal( const std::vector<std::string>& path );

/// Read a value from a drag/slider widget. This function is for strings.
[[nodiscard]] UiValueString uiReadValueString( const std::vector<std::string>& path );

/// Write a value to a drag/slider widget. This overload is for signed integers.
void uiWriteValue( const std::vector<std::string>& path, std::int64_t value );

/// Write a value to a drag/slider widget. This overload is for unsigned integers.
void uiWriteValue( const std::vector<std::string>& path, std::uint64_t value );

/// Write a value to a drag/slider widget. This overload is for real numbers.
void uiWriteValue( const std::vector<std::string>& path, double value );

/// Write a value to a drag/slider widget. This overload is for strings.
void uiWriteValue( const std::vector<std::string>& path, const std::string& value );

} // namespace MR::PythonViewer
