#pragma once

namespace MR
{

// Pass this as a second template argument to `boost::signals2::signal<...>` to stop the execution of handlers when one of them returns true.
struct StopOnTrueCombiner
{
    using result_type = bool;

    template<typename Iter>
    bool operator()( Iter first, Iter last ) const
    {
        while ( first != last )
        {
            if ( *first )
                return true; // The execution of slots stops if one returns true.
            ++first;
        }
        return false;
    }
};

}
