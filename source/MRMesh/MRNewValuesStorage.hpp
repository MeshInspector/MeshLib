#pragma once

#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRVector.h"

namespace MR
{

/// This class stores new values of a vertex field during one iteration of an algorithm:
/// all values if the zone is large, and only zone values if the zone is small (to avoid copying the whole field)
template<typename T>
class NewValuesStorage
{
public:
    NewValuesStorage( Vector<T, VertId> & field, const VertBitSet & zone ) : field_( field ), zone_( zone )
    {
        dense_ = 2 * zone.count() > field.size();
        if ( dense_ )
            return;
        zoneVerts_.reserve( zone.count() );
        for ( auto v : zone )
            zoneVerts_.push_back( v );
        newValues_.resize( zoneVerts_.size() );
    }

    /// computes new values f(v) for all zone vertices in parallel, and only then writes them in the field;
    /// \return false and leaves the field unchanged if interrupted by progress callback
    template<typename F>
    bool parallelProcess( F && f, const ProgressCallback & cb = {} )
    {
        if ( dense_ )
        {
            newField_ = field_;
            if ( !BitSetParallelFor( zone_, [&]( VertId v ) { newField_[v] = f( v ); }, cb ) )
                return false;
            field_.swap( newField_ );
            return true;
        }
        if ( !ParallelFor( zoneVerts_, [&]( size_t i ) { newValues_[i] = f( zoneVerts_[i] ); }, cb ) )
            return false;
        ParallelFor( zoneVerts_, [&]( size_t i ) { field_[zoneVerts_[i]] = newValues_[i]; } );
        return true;
    }

private:
    Vector<T, VertId> & field_;
    const VertBitSet & zone_;
    bool dense_ = false;
    Vector<T, VertId> newField_; // all values for a large zone
    std::vector<VertId> zoneVerts_; // zone vertices for a small zone
    std::vector<T> newValues_; // new values of zoneVerts_ for a small zone
};

} //namespace MR
