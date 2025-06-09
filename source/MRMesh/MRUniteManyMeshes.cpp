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

struct UniteReducerBaseParams
{
    bool fixDegeneracies = false;
    float maxError = 1e-5f;
    bool mergeMode = false;
    bool forceCut = false;
};

struct UnitePairOfMehsesParams
{
    UniteReducerBaseParams commonParams;
    const Vector3f* shift{ nullptr };
    BooleanResultMapper* mapper{ nullptr };
};

Expected<Mesh> unitePairOfMeshes( Mesh&& a, Mesh&& b, const UnitePairOfMehsesParams& unitePairParams )
{
    if ( a.points.empty() )
        return std::move( b );
    else if ( b.points.empty() )
        return std::move( a );

    AffineXf3f xf = AffineXf3f::translation( unitePairParams.shift ? *unitePairParams.shift : Vector3f() );
    BooleanResultMapper mapper_;
    BooleanParameters params;
    params.rigidB2A = unitePairParams.shift ? &xf : nullptr;
    params.mapper = unitePairParams.commonParams.fixDegeneracies || unitePairParams.mapper ? &mapper_ : nullptr;
    params.mergeAllNonIntersectingComponents = unitePairParams.commonParams.mergeMode;
    params.forceCut = unitePairParams.commonParams.forceCut;
    auto res = MR::boolean(
        std::move( a ),
        std::move( b ),
        BooleanOperation::Union,
        params
    );
    if ( !res.valid() )
        return unexpected( res.errorString );

    if ( unitePairParams.commonParams.fixDegeneracies )
    {
        auto newFaces = mapper_.newFaces();
        auto e = fixMeshDegeneracies( res.mesh, {
            .maxDeviation = unitePairParams.commonParams.maxError,
            .region = &newFaces,
            .mode = FixMeshDegeneraciesParams::Mode::Decimate
        } );
        if ( !e )
            return unexpected( std::move( e.error() ) );
    }

    if ( unitePairParams.mapper != nullptr )
        *unitePairParams.mapper = std::move( mapper_ );

    return res.mesh;
}

struct UniteReducerParams
{
    UniteReducerBaseParams commonParams;
    bool mergeOnFail = false;
    bool collectNewFaces = false;
    const std::vector<Vector3f>* shifts{ nullptr };
};

class BooleanReduce
{
public:
    BooleanReduce( std::vector<Mesh>& mehses, const UniteReducerParams& params ) :
        params_{ params },
        mergedMeshes_{ mehses }
    {}

    BooleanReduce( BooleanReduce& x, tbb::split ) :
        error{ x.error },
        params_{ x.params_ },
        mergedMeshes_{ x.mergedMeshes_ }
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
        UnitePairOfMehsesParams upParams;
        upParams.commonParams = params_.commonParams;
        upParams.mapper = params_.collectNewFaces ? &mapper : nullptr;
        upParams.shift = params_.shifts ? &shift : nullptr;
        if ( params_.commonParams.mergeMode )
        {
            res = unitePairOfMeshes( Mesh( resultMesh ), Mesh( y.resultMesh ), upParams );
        }
        else
        {
            res = unitePairOfMeshes( std::move( resultMesh ), std::move( y.resultMesh ), upParams );
        }
        if ( !res.has_value() )
        {
            if ( !params_.mergeOnFail )
            {
                error = std::move( res.error() );
                return;
            }
            else
            {
                FaceMap fMap;
                resultMesh.addMesh( y.resultMesh, params_.collectNewFaces ? &fMap : nullptr );
                if ( params_.collectNewFaces )
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
        if ( params_.collectNewFaces )
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
        if ( params_.shifts )
            resShift = ( *params_.shifts )[r.begin()];
        resultMesh = std::move( mergedMeshes_[r.begin()] );
        newFaces.resize( resultMesh.topology.faceSize() );
    }

    Mesh resultMesh;
    std::string error;
    Vector3f resShift;
    FaceBitSet newFaces;
private:
    UniteReducerParams params_;

    std::vector<Mesh>& mergedMeshes_;
};

Expected<Mesh> uniteManyMeshes(
    const std::vector<const Mesh*>& meshes, const UniteManyMeshesParams& params /*= {} */ )
{
    MR_TIMER;
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
                        if ( !collidingResAB.empty() )
                        {
                            intersects.store( true, std::memory_order::relaxed );
                            break;
                        }
                        auto collidingResBA = findCollidingEdgeTrisPrecise( *groupMesh, *mesh, intConverter, nullptr, true );
                        if ( !collidingResBA.empty() )
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
                        assert( nestedMeshes.none() );
                        break;
                    }
                    if ( nestedMeshes.any() )
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
                    mergedMesh.addMesh( *meshes[meshIndex] );
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
    UniteReducerParams urParams;
    urParams.commonParams.fixDegeneracies = params.fixDegenerations;
    urParams.commonParams.forceCut = params.forceCut;
    urParams.commonParams.maxError = params.maxAllowedError;
    urParams.commonParams.mergeMode = mergeNestedComponents;
    urParams.collectNewFaces = params.newFaces != nullptr;
    urParams.mergeOnFail = params.mergeOnFail;
    urParams.shifts = params.useRandomShifts ? &randomShifts : nullptr;
    BooleanReduce reducer( mergedMeshes, urParams );
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
