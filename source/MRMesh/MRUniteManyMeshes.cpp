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
#include "MRParallelFor.h"
#include "MRMeshComponents.h"
#include "MRMeshFillHole.h"
#include "MRMapOrHashMap.h"
#include "MRPartMapping.h"
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
    BooleanResult res;
    if ( !unitePairParams.commonParams.forceCut )
    {
        res = MR::boolean(
            std::move( a ),
            std::move( b ),
            BooleanOperation::Union,
            params
        );
    }
    else
    {
        res = MR::forceBoolean( a, b, BooleanOperation::Union, params );
    }
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
    /// if set, the sources of the faces of each merged mesh
    Vector<Vector<ObjFaceId, FaceId>, ObjId>* new2oldFaces{ nullptr };
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
        if ( y.resultMesh.points.empty() )
            return;
        if ( resultMesh.points.empty() )
        {
            // unitePairOfMeshes would not fill the mapper
            resultMesh = std::move( y.resultMesh );
            resShift = y.resShift;
            newFaces = std::move( y.newFaces );
            new2oldFaces = std::move( y.new2oldFaces );
            return;
        }
        const bool collectNew2Old = params_.new2oldFaces != nullptr;
        Vector3f shift = y.resShift - resShift;
        BooleanResultMapper mapper;
        Expected<Mesh> res;
        UnitePairOfMehsesParams upParams;
        upParams.commonParams = params_.commonParams;
        upParams.mapper = params_.collectNewFaces || collectNew2Old ? &mapper : nullptr;
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
                resultMesh.addMesh( y.resultMesh, params_.collectNewFaces || collectNew2Old ? &fMap : nullptr );
                if ( params_.collectNewFaces )
                {
                    newFaces.resize( resultMesh.topology.faceSize() );
                    for ( auto f : y.newFaces )
                    {
                        if ( f >= fMap.size() )
                            continue;
                        if ( auto mF = fMap[f] )
                            newFaces.set( mF );
                    }
                }
                if ( collectNew2Old )
                {
                    new2oldFaces.resize( resultMesh.topology.faceSize() );
                    for ( FaceId f = 0_f; f < fMap.size(); ++f )
                        if ( auto mF = fMap[f] )
                            new2oldFaces[mF] = getAt( y.new2oldFaces, f );
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
        if ( collectNew2Old )
        {
            Vector<ObjFaceId, FaceId> resNew2Old( resultMesh.topology.faceSize() );
            auto mapSources = [&] ( BooleanResultMapper::MapObject obj, const Vector<ObjFaceId, FaceId>& old2src )
            {
                const auto new2old = mapper.getNew2OldFaceMap( obj );
                ParallelFor( new2old, [&] ( FaceId f )
                {
                    if ( auto oldF = new2old[f]; oldF && f < resNew2Old.size() )
                        resNew2Old[f] = getAt( old2src, oldF );
                } );
            };
            mapSources( BooleanResultMapper::MapObject::A, new2oldFaces );
            mapSources( BooleanResultMapper::MapObject::B, y.new2oldFaces );
            new2oldFaces = std::move( resNew2Old );
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
        if ( params_.new2oldFaces )
            new2oldFaces = std::move( ( *params_.new2oldFaces )[ObjId( r.begin() )] );
    }

    Mesh resultMesh;
    std::string error;
    Vector3f resShift;
    FaceBitSet newFaces;
    Vector<ObjFaceId, FaceId> new2oldFaces;
private:
    UniteReducerParams params_;

    std::vector<Mesh>& mergedMeshes_;
};

Expected<Mesh> uniteManyMeshes(
    const std::vector<const Mesh*>& meshes, const UniteManyMeshesParams& params /*= {} */ )
{
    MR_TIMER;
    if ( params.new2oldFaces )
        params.new2oldFaces->clear();
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
    Vector<Vector<ObjFaceId, FaceId>, ObjId> mergedNew2Old( params.new2oldFaces ? mergedMeshes.size() : 0 );
    if ( separateComponentsProcess )
    {
        ParallelFor( nonIntersectingGroups, [&] ( size_t i )
        {
            auto& mergedMesh = mergedMeshes[i];
            auto& mergeGroup = nonIntersectingGroups[i];
            for ( auto meshIndex : mergeGroup )
            {
                FaceMap fMap;
                mergedMesh.addMesh( *meshes[meshIndex], params.new2oldFaces ? &fMap : nullptr );
                if ( !params.new2oldFaces )
                    continue;
                auto& new2old = mergedNew2Old[ObjId( i )];
                new2old.resize( mergedMesh.topology.faceSize() );
                for ( FaceId f = 0_f; f < fMap.size(); ++f )
                    if ( auto mF = fMap[f] )
                        new2old[mF] = { ObjId( meshIndex ), f };
            }
        } );
    }
    else
    {
        ParallelFor( meshes, [&] ( size_t i )
        {
            mergedMeshes[i] = *meshes[i];
            if ( !params.new2oldFaces )
                return;
            auto& new2old = mergedNew2Old[ObjId( i )];
            new2old.resize( mergedMeshes[i].topology.faceSize() );
            for ( auto f : mergedMeshes[i].topology.getValidFaces() )
                new2old[f] = { ObjId( i ), f };
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
    urParams.new2oldFaces = params.new2oldFaces ? &mergedNew2Old : nullptr;
    BooleanReduce reducer( mergedMeshes, urParams );
    tbb::parallel_deterministic_reduce( tbb::blocked_range<int>( 0, int( mergedMeshes.size() ), 1 ), reducer );
    if ( !reducer.error.empty() )
        return unexpected( "Error while uniting meshes: " + reducer.error );

    if ( !reportProgress( params.progressCb, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( params.newFaces != nullptr )
        *params.newFaces = std::move( reducer.newFaces );
    if ( params.new2oldFaces )
    {
        reducer.new2oldFaces.resize( reducer.resultMesh.topology.faceSize() );
        *params.new2oldFaces = std::move( reducer.new2oldFaces );
    }
    return reducer.resultMesh;
}

namespace
{
/// returns false if the mesh was replaced by the result of self-boolean, so its faces do not correspond to the original ones
bool normalizeUniteMesh( Mesh& mesh, bool needClosed, const UniteMeshNormalizationParams& params )
{
    bool facesKept = true;
    if ( needClosed )
    {
        FillHoleParams fhp;
        fhp.metric = getMinAreaMetric( mesh );
        fillHoles( mesh, mesh.topology.findHoleRepresentiveEdges(), fhp );
    }

    if ( params.trySelfBoolean )
    {
        auto sbRes = selfBoolean( mesh );
        if ( sbRes.has_value() )
        {
            sbRes->deleteFaces( sbRes->topology.getValidFaces() - MeshComponents::getLargestComponent( *sbRes ) );
            mesh = std::move( *sbRes );
            facesKept = false;
        }
    }
    if ( params.flipInverted && mesh.volume() < 0.0f )
        mesh.topology.flipOrientation();

    if ( params.expansionRatio != 0.0f )
    {
        auto center = mesh.findCenterFromFaces();
        mesh.transform( AffineXf3f::xfAround( Matrix3f::scale( 1.0f + params.expansionRatio ), center ) );
    }
    return facesKept;
}
}

Expected<Mesh> uniteManyMeshesMutable( const std::vector<Mesh*>& meshes,
    const UniteManyMeshesParams& params, const UniteMeshNormalizationParams& normalizeParams )
{
    MR_TIMER;
    auto keepGoing = ParallelFor( meshes, [&] ( size_t i )
    {
        normalizeUniteMesh( *meshes[i], params.forceCut || normalizeParams.flipInverted, normalizeParams );
    }, subprogress( params.progressCb, 0.0f, 0.3f ) );
    if ( !keepGoing )
        return unexpectedOperationCanceled();
    UniteManyMeshesParams ump = params;
    ump.progressCb = subprogress( params.progressCb, 0.3f, 1.0f );
    // reinterpret_cast here to avoid allocation new vector with same pointers
    return uniteManyMeshes( reinterpret_cast< const std::vector<const Mesh*>& >( meshes ), ump );
}

Expected<Mesh> uniteComponents( const Mesh& mesh,
    const UniteManyMeshesParams& params, const UniteMeshNormalizationParams& normalizeParams )
{
    MR_TIMER;
    const auto comps = MeshComponents::getAllComponentsFaces( mesh );
    if ( !reportProgress( params.progressCb, 0.1f ) )
        return unexpectedOperationCanceled();
    const auto numComps = comps.offsets.size() - 1;
    std::vector<Mesh> components( numComps );
    std::vector<const Mesh*> meshPtrs( numComps );
    // component face -> input mesh face
    Vector<FaceMapOrHashMap, ObjId> comp2orig( params.new2oldFaces ? numComps : 0 );
    auto keepGoing = ParallelFor( components, [&] ( size_t i )
    {
        FaceBitSet compBs( mesh.topology.faceSize() );
        comps.setComponentBits( RegionId( i ), compBs );
        components[i].addMeshPart( MeshPart( mesh, &compBs ), false, {}, {}, { .tgt2srcFaces = params.new2oldFaces ? &comp2orig[ObjId( i )] : nullptr } );
        if ( !normalizeUniteMesh( components[i], params.forceCut || normalizeParams.flipInverted, normalizeParams ) && params.new2oldFaces )
            comp2orig[ObjId( i )].clear();
        meshPtrs[i] = &components[i];

        compBs = {}; // reduce peek memory
    }, subprogress( params.progressCb, 0.1f, 0.4f ) );
    if ( !keepGoing )
        return unexpectedOperationCanceled();
    auto np = params;
    np.progressCb = subprogress( params.progressCb, 0.4f, 1.0f );
    auto res = uniteManyMeshes( meshPtrs, np );
    if ( res && params.new2oldFaces )
    {
        ParallelFor( *params.new2oldFaces, [&] ( FaceId f )
        {
            auto& src = ( *params.new2oldFaces )[f];
            if ( src.objId.valid() )
                src.fId = getAt( *comp2orig[src.objId].getMap(), src.fId );
        } );
    }
    return res;
}

}
