#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRVoxelsVolume.h"
#include "MRAffineXf3.h"
#include "MRphmap.h"

#include <array>
#include <optional>


namespace MR
{

/// This class represents tooth id
class DentalId
{
public:
    /// Creates id from FDI number known at compile time
    template <int id>
    static constexpr DentalId fromFDI()
    {
        const int t = id % 10;
        const int q = id / 10;
        static_assert( q >= 1 && q <= 4 && t >= 1 && t <= 8 );

        return DentalId( id );
    }

    /// Creates id from FDI number known only at runtime
    MRMESH_API static std::optional<DentalId> fromFDI( int id );

    /// Returns FDI representation of the id
    MRMESH_API int fdi() const;

    auto operator <=> ( const DentalId& other ) const = default;

private:
    constexpr explicit DentalId( int fdi ):
        fdi_( fdi )
    {}

    int fdi_;
};

}

template <>
struct std::hash<MR::DentalId>
{
    inline size_t operator() ( const MR::DentalId& id ) const noexcept
    {
        return hash<int>{}( id.fdi() );
    }
};


namespace MR
{

/// This class is an alternative to directly invoking \ref meshToDirectionVolume for the mesh retrieved from the teeth mask.
/// It is better because when a single mesh is created from mask, some neighboring teeth might fuse together, creating incorrect mask.
/// This class invokes meshing for each teeth separately, thus eliminating this problem.
class TeethMaskToDirectionVolumeConvertor
{
public:
    /// Initialize class
    MRMESH_API static Expected<TeethMaskToDirectionVolumeConvertor> create( const VdbVolume& volume );

    /// Returns all the teeth present in volume and corresponding bounding boxes
    MRMESH_API const HashMap<DentalId, Box3i>& getToothBounds() const;

    /// See \ref meshToDirectionVolume for details
    using DirectionVolume = std::array<SimpleVolume, 3>;
    struct ProcessResult
    {
        DirectionVolume volume;
        AffineXf3f xf;
    };

    /// Converts single tooth into direction volume
    MRMESH_API Expected<ProcessResult> convertTooth( DentalId id ) const;

    /// Converts all the teeth into direction volume
    MRMESH_API Expected<ProcessResult> convertAll() const;

private:
    MRMESH_API TeethMaskToDirectionVolumeConvertor();

    HashMap<DentalId, Box3i> presentTeeth_;
    SimpleVolume mask_;
};


}