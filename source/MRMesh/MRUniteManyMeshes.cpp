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

Expected<Mesh> unitePairOfMeshes( Mesh&& a, Mesh&& b, 
    bool fixDegenerations, float maxError, bool mergeMode, const Vector3f* shift = nullptr, BooleanResultMapper* mapper = nullptr )
{
    if ( a.points.empty() )
        return std::move( b );
    else if ( b.points.empty() )
        return std::move( a );

    AffineXf3f xf = AffineXf3f::translation( shift ? *shift : Vector3f() );
    BooleanResultMapper mapper_;
    BooleanParameters params;
    params.rigidB2A = shift ? &xf : nullptr;
    params.mapper = fixDegenerations || mapper ? &mapper_ : nullptr;
    params.mergeAllNonIntersectingComponents = mergeMode;
    auto res = MR::boolean(
        std::move( a ),
        std::move( b ),
        BooleanOperation::Union,
        params
    );
    if ( !res.valid() )
        return unexpected( res.errorString );

    if ( fixDegenerations )
    {
        auto newFaces = mapper_.newFaces();
        resolveMeshDegenerations( res.mesh, {
            .maxDeviation = maxError,
            .region = &newFaces,
        } );
    }

    if ( mapper != nullptr )
        *mapper = std::move( mapper_ );

    return res.mesh;
}

class BooleanReduce
{
public:
    BooleanReduce( std::vector<Mesh>& mehses, const std::vector<Vector3f>& shifts, float maxError, bool fixDegenerations, bool collectNewFaces, bool mergeMode ) :
        maxError_{ maxError },
        fixDegenerations_{ fixDegenerations },
        mergedMeshes_{ mehses },
        shifts_{ shifts },
        collectNewFaces_{ collectNewFaces },
        mergeMode_{ mergeMode }
    {}

    BooleanReduce( BooleanReduce& x, tbb::split ) :
        error{ x.error },
        maxError_{ x.maxError_ },
        fixDegenerations_{ x.fixDegenerations_ },
        mergedMeshes_{ x.mergedMeshes_ },
        shifts_{ x.shifts_ },
        collectNewFaces_{ x.collectNewFaces_ },
        mergeMode_{ x.mergeMode_ }
    {
    }

    void join( BooleanReduce& y )
    {
        if ( !error.empty() )
            return;
        if ( !y.error.empty() )
        {
            error = y.error;
            return;
        }
        Vector3f shift = y.resShift - resShift;
        BooleanResultMapper mapper;
        Expected<Mesh> res;
        if ( mergeMode_ )
        {
            res = unitePairOfMeshes(
                Mesh( resultMesh ),
                Mesh( y.resultMesh ),
                fixDegenerations_, maxError_,
                mergeMode_,
                shifts_.empty() ? nullptr : &shift,
                collectNewFaces_ ? &mapper : nullptr );
        }
        else
        {
            res = unitePairOfMeshes(
                std::move( resultMesh ),
                std::move( y.resultMesh ),
                fixDegenerations_, maxError_,
                mergeMode_,
                shifts_.empty() ? nullptr : &shift,
                collectNewFaces_ ? &mapper : nullptr );
        }
        if ( !res.has_value() )
        {
            if ( !mergeMode_ )
            {
                error = std::move( res.error() );
                return;
            }
            else
            {
                FaceMap fMap;
                resultMesh.addPart( y.resultMesh, collectNewFaces_ ? &fMap : nullptr );
                if ( collectNewFaces_ )
                {
                    newFaces.resize( fMap.size() );
                    for ( auto f : y.newFaces )
                    {
                        if ( f >= fMap.size() )
                            continue;
                        if ( auto mF = fMap[f] )
                            newFaces.set( mF );
                    }
                }
                return;
            }
        }
        resultMesh = std::move( res.value() );
        if ( collectNewFaces_ )
        {
            // store faces created by the latest union operation and map faces created by previous ones
            newFaces = mapper.newFaces()
                    | mapper.map( newFaces, BooleanResultMapper::MapObject::A )
                    | mapper.map( y.newFaces, BooleanResultMapper::MapObject::B );
        }
    }

    void operator()( const tbb::blocked_range<int>& r )
    {
        assert( r.size() == 1 );
        assert( resultMesh.points.empty() );
        if ( !shifts_.empty() )
            resShift = shifts_[r.begin()];
        resultMesh = std::move( mergedMeshes_[r.begin()] );
        newFaces.resize( resultMesh.topology.faceSize() );
    }

    Mesh resultMesh;
    std::string error;
    Vector3f resShift;
    FaceBitSet newFaces;
private:
    float maxError_{ 0.0f };
    bool fixDegenerations_{ false };
    std::vector<Mesh>& mergedMeshes_;
    const std::vector<Vector3f>& shifts_;
    bool collectNewFaces_{ false };
    bool mergeMode_{ false };
};

Expected<Mesh> uniteManyMeshes( 
    const std::vector<const Mesh*>& meshes, const UniteManyMeshesParams& params /*= {} */ )
{
    MR_TIMER
    if ( meshes.empty() )
        return Mesh{};

    bool separateComponentsProcess = params.nestedComponentsMode != NestedComponenetsMode::Union;
    bool mergeNestedComponents = params.nestedComponentsMode == NestedComponenetsMode::Merge;

    // find non-intersecting groups for simple merge instead of union
    // the groups are represented with indices of input meshes
    std::vector<std::vector<int>> nonIntersectingGroups;
    if ( separateComponentsProcess )
    {
        std::vector<Box3d> meshBoxes( meshes.size() );
        for ( int m = 0; m < meshes.size(); ++m )
            if ( meshes[m] )
                meshBoxes[m] = Box3d( meshes[m]->getBoundingBox() );
        if ( !reportProgress( params.progressCb, 0.1f ) )
            return unexpectedOperationCanceled();
        auto sp = subprogress( params.progressCb, 0.1f, 0.5f );
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
                    if ( intersects.load( std::memory_order::relaxed ) || ( !mergeNestedComponents && included.load( std::memory_order::relaxed ) ) )
                        return;
                    auto& nested = nestedPerThread.local();

                    for ( int i = range.begin(); i < range.end(); ++i )
                    {
                        auto& groupMesh = meshes[group[i]];

                        Box3d box = meshBoxes[m];
                        box.include( meshBoxes[group[i]] );
                        auto intConverter = getToIntConverter( box );
                        auto collidingResAB = findCollidingEdgeTrisPrecise( *mesh, *groupMesh, intConverter, nullptr, true );
                        if ( !collidingResAB.edgesAtrisB.empty() || !collidingResAB.edgesBtrisA.empty() )
                        {
                            intersects.store( true, std::memory_order::relaxed );
                            break;
                        }
                        auto collidingResBA = findCollidingEdgeTrisPrecise( *groupMesh, *mesh, intConverter, nullptr, true );
                        if ( !collidingResBA.edgesAtrisB.empty() || !collidingResBA.edgesBtrisA.empty() )
                        {
                            intersects.store( true, std::memory_order::relaxed );
                            break;
                        }
                        if ( !mergeNestedComponents )
                        {
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
                    }
                } );
                if ( !mergeNestedComponents )
                {
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
                        std::erase_if( group, [&] ( int groupIndex )
                    {
                        return nestedMeshes.test( groupIndex );
                    } );
                }
                else if ( intersects )
                    continue;
                group.emplace_back( m );
                merged = true;
                break;
            }
            if ( !merged )
            {
                nonIntersectingGroups.emplace_back();
                nonIntersectingGroups.back().emplace_back( m );
            }
            if ( !reportProgress( sp, float( m + 1 ) / meshes.size() ) )
                return unexpectedOperationCanceled();
        }
    }


    // merge non-intersecting groups
    std::vector<Mesh> mergedMeshes( separateComponentsProcess ? nonIntersectingGroups.size() : meshes.size() );
    if ( separateComponentsProcess )
    {
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
    }
    else
    {
        tbb::parallel_for( tbb::blocked_range<int>( 0, int( meshes.size() ) ),
                   [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                mergedMeshes[i] = *meshes[i];
            }
        } );
    }

    float currentProgress = separateComponentsProcess ? 0.7f : 0.3f;
    if ( !reportProgress( params.progressCb, currentProgress ) )
        return unexpectedOperationCanceled();

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
    BooleanReduce reducer( mergedMeshes, randomShifts, params.maxAllowedError, params.fixDegenerations, params.newFaces != nullptr, mergeNestedComponents );
    tbb::parallel_deterministic_reduce( tbb::blocked_range<int>( 0, int( mergedMeshes.size() ), 1 ), reducer );
    if ( !reducer.error.empty() )
        return unexpected( "Error while uniting meshes: " + reducer.error );

    if ( !reportProgress( params.progressCb, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( params.newFaces != nullptr )
        *params.newFaces = std::move( reducer.newFaces );
    return reducer.resultMesh;
}

}
