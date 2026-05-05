#include "MRSequentialNester.h"
#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRTimer.h"

namespace MR
{

namespace Nesting
{

SequentialNester::SequentialNester( const NestingBaseParams& params, float voxelSize ) :
    baseParams_{ params },
    tetrisOptions_{ .voxelSize = voxelSize,.nestVoxelsCache = &voxelsCache_,.nestDimensionsCache = &dimensionsCache_,.occupiedVoxelsCache = &occupiedVoxelsCache_ }
{
}

Expected<NestingResult> SequentialNester::nestMesh( const MeshXf& meshXf, const BoxNestingOptions& options, const std::vector<OutEdge>* densificationSequence )
{
    auto res = nestMeshes( { meshXf }, options, densificationSequence );
    if ( !res.has_value() )
        return unexpected( std::move( res.error() ) );
    if ( res->empty() )
    {
        assert( false );
        return {};
    }
    return ( *res )[ObjId( 0 )];
}

Expected<Vector<NestingResult, ObjId>> SequentialNester::nestMeshes( const Vector<MeshXf, ObjId>& meshes, const BoxNestingOptions& options, const std::vector<OutEdge>* densificationSequence )
{
    MR_TIMER;
    BoxNestingParams bnParams;
    bnParams.baseParams = baseParams_;
    bnParams.options = options;
    bnParams.options.preNestedVolumes = &nestedBoxesCache_;
    bnParams.options.additinalSocketCorners = &addBoxCornersCache_;

    bool doTetris = ( !densificationSequence || !densificationSequence->empty() ) && tetrisOptions_.voxelSize > 0;

    bnParams.options.cb = subprogress( options.cb, 0.0f, doTetris ? 0.3f : 0.9f );

    auto boxNestingRes = boxNesting( meshes, bnParams );
    if ( !boxNestingRes.has_value() )
        return boxNestingRes;

    int numPlaced = 0;
    for ( const auto& r : *boxNestingRes )
        if ( r.nested )
            ++numPlaced;

    if ( numPlaced == 0 )
        return boxNestingRes;

    Vector<MeshXf, ObjId> nestedMeshes;
    if ( doTetris )
        nestedMeshes.resize( numPlaced );

    auto prevNestedSize = nestedBoxesCache_.size();
    nestedBoxesCache_.resize( prevNestedSize + numPlaced );
    auto* nestedBoxes = nestedBoxesCache_.data() + prevNestedSize;

    ObjId no( 0 );
    for ( ObjId io( 0 ); io < meshes.size(); ++io )
    {
        if ( !( *boxNestingRes )[io].nested )
            continue;
        if ( doTetris )
            nestedMeshes[no] = meshes[io];
        const auto& resXf = ( *boxNestingRes )[io].xf;
        nestedBoxes[no] = meshes[io].mesh->computeBoundingBox( &resXf ); // nested box before densification
        if ( options.expansionFactor )
        {
            AffineXf3f scaleXf = AffineXf3f::xfAround( Matrix3f::scale( *options.expansionFactor ), nestedBoxes[no].min );
            nestedBoxes[no].max = scaleXf( nestedBoxes[no].max );
            if ( doTetris )
                nestedMeshes[no].xf = scaleXf * resXf;
        }
        else
        {
            if ( doTetris )
                nestedMeshes[no].xf = resXf;
        }
        ++no;
    }

    if ( doTetris )
    {
        TetrisDensifyParams tdParams;
        tdParams.baseParams = baseParams_;
        tdParams.options = tetrisOptions_;
        if ( densificationSequence )
            tdParams.options.densificationSequence = *densificationSequence;
        else
            tdParams.options.densificationSequence = { OutEdge::MinusZ,OutEdge::MinusY,OutEdge::MinusX,
                                                       OutEdge::MinusZ,OutEdge::MinusY,OutEdge::MinusX,
                                                       OutEdge::MinusZ };
        tdParams.options.cb = subprogress( options.cb, 0.3f, 0.8f );

        auto tetrisDensifyRes = tetrisNestingDensify( nestedMeshes, tdParams );

        if ( !tetrisDensifyRes.has_value() )
            return unexpected( std::move( tetrisDensifyRes.error() ) );

        no = ObjId( 0 );
        for ( ObjId io( 0 ); io < meshes.size(); ++io )
        {
            if ( !( *boxNestingRes )[io].nested )
                continue;
            const auto& resXf = ( *tetrisDensifyRes )[no];
            ( *boxNestingRes )[io].xf = resXf * ( *boxNestingRes )[io].xf;
            nestedBoxes[no].min += resXf.b;
            nestedBoxes[no].max += resXf.b;
            nestedBoxes[no].intersect( baseParams_.nest );
            ++no;
        }
        if ( !reportProgress( options.cb, 0.9f ) )
            return unexpectedOperationCanceled();
    }
    addBoxCornersCache_.clear();

    auto updateCornersRes = fillNestingSocketCorneres( nestedBoxesCache_, addBoxCornersCache_, subprogress( options.cb, 0.9f, 1.0f ) );
    if ( !updateCornersRes.has_value() )
        return unexpected( std::move( updateCornersRes.error() ) );

    return boxNestingRes;
}

}

}
