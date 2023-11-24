#pragma once

#include <functional>
#include <optional>

namespace MR
{

template <typename Result>
using Resumable = std::function<std::optional<Result> ()>;

} // namespace MR
