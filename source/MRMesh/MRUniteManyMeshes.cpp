#include "MRUniteManyMeshes.h"
#include "MRPch/MRTBB.h"
#include "MRTimer.h"
#include "MRMeshCollide.h"
#include "MRAffineXf3.h"
#include "MRMeshBoolean.h"
#include "MRMeshFixer.h"
#include "MRMeshDecimate.h"
#include "MRMeshCollidePrecise.h"
#include "MRBox.h"
#include <random>

namespace MR
{

tl::expected<Mesh, std::string> unitePairOfMeshes( const Mesh& a, const Mesh& b, 
    bool fixDegenerations, float maxError, const Vector3f* shift = nullptr )
{
    if ( a.points.empty() )
        return b;
    else if ( b.points.empty() )
        return a;

    AffineXf3f xf = AffineXf3f::translation( shift ? *shift : Vector3f() );
    auto res = MR::boolean( a, b, BooleanOperation::Union, shift ? &xf : nullptr );
    if ( !res.valid() )
        return tl::make_unexpected( res.errorString );

    if ( fixDegenerations )
        resolveMeshDegenerations( res.mesh, { .maxIters = INT_MAX, .maxDeviation = maxError } ); // INT_MAX cause it will stop itself if no changes done

    return res.mesh;
}

class BooleanReduce
{
public:
    BooleanReduce( std::vector<Mesh>& mehses, const std::vector<Vector3f>& shifts, float maxError, bool fixDegenerations ) :
        maxError_{ maxError },
        fixDegenerations_{ fixDegenerations },
        mergedMeshes_{ mehses },
        shifts_{ shifts }
    {}

    BooleanReduce( BooleanReduce& x, tbb::split ) :
        error{ x.error },
        maxError_{ x.maxError_ },
        fixDegenerations_{ x.fixDegenerations_ },
        mergedMeshes_{ x.mergedMeshes_ },
        shifts_{ x.shifts_ }
    {
    }

    void join( const BooleanReduce& y )
    {
        if ( !error.empty() )
            return;
        if ( !y.error.empty() )
        {
            error = y.error;
            return;
        }
        Vector3f shift = y.resShift - resShift;
        auto res = unitePairOfMeshes( resultMesh, y.resultMesh, fixDegenerations_, maxError_, shifts_.empty() ? nullptr : &shift );
        if ( res.has_value() )
            resultMesh = std::move( res.value() );
        else
            error = std::move( res.error() );
    }

    void operator()( const tbb::blocked_range<int>& r )
    {
        assert( r.size() == 1 );
        assert( resultMesh.points.empty() );
        if ( !shifts_.empty() )
            resShift = shifts_[r.begin()];
        resultMesh = std::move( mergedMeshes_[r.begin()] );
    }

    Mesh resultMesh;
    std::string error;
    Vector3f resShift;
private:
    float maxError_{ 0.0f };
    bool fixDegenerations_{ false };
    std::vector<Mesh>& mergedMeshes_;
    const std::vector<Vector3f>& shifts_;
};

tl::expected<Mesh, std::string> uniteManyMeshes( 
    const std::vector<const Mesh*>& meshes, const UniteManyMeshesParams& params /*= {} */ )
{
    MR_TIMER
    if ( meshes.empty() )
        return Mesh{};

    // find non-intersecting groups for simple merge instead of union
    // the groups are represented with indices of input meshes
    std::vector<std::vector<int>> nonIntersectingGroups;
    std::vector<Box3d> meshBoxes( meshes.size() );
    for ( int m = 0; m < meshes.size(); ++m )
        if ( meshes[m] )
            meshBoxes[m] = Box3d( meshes[m]->getBoundingBox() );
    for ( int m = 0; m < meshes.size(); ++m )
    {
        const auto& mesh = meshes[m];
        if ( !mesh )
            continue;
        bool merged = false;
        for ( auto& group : nonIntersectingGroups )
        {
            // mesh intersects with current group
            std::atomic_bool intersects{ false };
            // mesh is included in one of group meshes
            std::atomic_bool included{ false };
            // mesh contains some group meshes
            tbb::enumerable_thread_specific<BitSet> nestedPerThread( meshes.size() );
            tbb::parallel_for( tbb::blocked_range<int>( 0, int( group.size() ) ),
                               [&] ( const tbb::blocked_range<int>& range )
            {
                if ( intersects.load( std::memory_order::relaxed ) || included.load( std::memory_order::relaxed ) )
                    return;
                auto& nested = nestedPerThread.local();

                for ( int i = range.begin(); i < range.end(); ++i )
                {
                    auto& groupMesh = meshes[group[i]];

                    Box3d box = meshBoxes[m];
                    box.include( meshBoxes[group[i]] );
                    auto intConverter = getToIntConverter( box );
                    auto collidingRes = findCollidingEdgeTrisPrecise( *mesh, *groupMesh, intConverter, nullptr, true );
                    if ( !collidingRes.edgesAtrisB.empty() || !collidingRes.edgesBtrisA.empty() )
                    {
                        intersects.store( true, std::memory_order::relaxed );
                        break;
                    }

                    if ( isNonIntersectingInside( *mesh, *groupMesh ) )
                    {
                        included.store( true, std::memory_order::relaxed );
                        break;
                    }
                    else if ( isNonIntersectingInside( *groupMesh, *mesh ) )
                    {
                        nested.set( group[i] );
                    }
                }
            } );
            BitSet nestedMeshes( meshes.size() );
            for ( auto&& nested : nestedPerThread )
                nestedMeshes |= nested;
            if ( intersects )
                continue;
            if ( included )
            {
                assert( nestedMeshes.count() == 0 );
                break;
            }
            if ( nestedMeshes.count() != 0 )
                std::erase_if( group, [&] ( int groupIndex ) { return nestedMeshes.test( groupIndex ); } );
            group.emplace_back( m );
            merged = true;
            break;
        }
        if ( !merged )
        {
            nonIntersectingGroups.emplace_back();
            nonIntersectingGroups.back().emplace_back( m );
        }
    }

    // merge non-intersecting groups
    std::vector<Mesh> mergedMeshes( nonIntersectingGroups.size() );
    tbb::parallel_for( tbb::blocked_range<int>( 0, int( nonIntersectingGroups.size() ) ),
                       [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            auto& mergedMesh = mergedMeshes[i];
            auto& mergeGroup = nonIntersectingGroups[i];
            for ( auto meshIndex : mergeGroup )
                mergedMesh.addPart( *meshes[meshIndex] );
        }
    } );

    std::vector<Vector3f> randomShifts;
    if ( params.useRandomShifts )
    {
        randomShifts.resize( mergedMeshes.size() );
        std::mt19937 mt( params.randomShiftsSeed );
        std::uniform_real_distribution<float> dist( -params.maxAllowedError * 0.5f, params.maxAllowedError * 0.5f );
        for ( auto& shift : randomShifts )
            for ( int i = 0; i < 3; ++i )
                shift[i] = dist( mt );
    }

    // parallel reduce unite merged meshes
    BooleanReduce reducer( mergedMeshes, randomShifts, params.maxAllowedError, params.fixDegenerations );
    tbb::parallel_deterministic_reduce( tbb::blocked_range<int>( 0, int( mergedMeshes.size() ), 1 ), reducer );
    if ( !reducer.error.empty() )
        return tl::make_unexpected( "Error while uniting meshes: " + reducer.error );

    return reducer.resultMesh;
}

}
