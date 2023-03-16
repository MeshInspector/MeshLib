#pragma once

#include "imgui.h"
#include "MRVector2.h"

namespace MR
{

template <typename T>
inline constexpr Vector2<T>::Vector2( ImVec2 i ) noexcept
    : x( i.x ), y( i.y )
{
}

template <typename T>
inline constexpr Vector2<T>::operator ImVec2() const noexcept
{
    return ImVec2( float( x ), float( y ) );
}

} //namespace MR
