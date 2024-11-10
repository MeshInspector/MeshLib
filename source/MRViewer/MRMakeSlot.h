#pragma once

#include <type_traits>

template<typename MemberFuncPtr, typename BaseClass>
auto bindSlotCallback( BaseClass* base, MemberFuncPtr func )
{
    static_assert( !( std::is_move_assignable_v<BaseClass> || std::is_move_constructible_v<BaseClass> ),
                   "MAKE_SLOT requires a non-movable type" );
    return[base, func] ( auto&&... args )
    {
        return ( base->*func )( std::forward<decltype( args )>( args )... );
    };
}

// you will not be able to move your struct after using this macro
#define MAKE_SLOT(func) bindSlotCallback(this,func)
