#include "MRMovementBuildBody.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRAffineXf3.h"
#include "MRBox.h"
#include "MRTimer.h"
#include "MRMatrix3.h"
#include "MRContour.h"

namespace MR
{

Mesh makeMovementBuildBody( const Contours3f& bodyContours, const Contours3f& trajectoryContoursOrg,
    const MovementBuildBodyParams& params )
{
    MR_TIMER;

    // copy to clear duplicates (mb leave it to user?)
    auto trajectoryContours = trajectoryContoursOrg;
    // filter same points in trajectory
    for ( auto& trajC : trajectoryContours )
        trajC.erase( std::unique( trajC.begin(), trajC.end() ), trajC.end() );

    AffineXf3f xf;
    std::optional<AffineXf3f> xf0Inv;
    Vector3f trans;
    Matrix3f prevHalfRot;
    Matrix3f accumRot;
    Vector3f center;
    Matrix3f scaling;
    if ( params.center )
        center = *params.center;
    else
    {
        Box3f box;
        for ( const auto& c : bodyContours )
            for ( const auto& p : c )
                box.include( p );
        center = box.center();
    }
    if ( params.b2tXf )
        center = ( *params.b2tXf )( center );
    Vector3f normal;
    if ( params.bodyNormal )
        normal = *params.bodyNormal;
    else
    {
        for ( const auto& bc : bodyContours )
            normal += calcOrientedArea( bc );
    }
    if ( params.b2tXf )
        normal = params.b2tXf->A.inverse().transposed() * normal;
    // minus to have correct orientation of result mesh
    normal = -normal.normalized();

    Vector3f prevVec, nextVec; // needed for soft rotation


    auto halfRot = [] ( const Vector3f& from, const Vector3f& to, Vector3f& axis, float& halfAng )->Matrix3f
    {
        axis = cross( from, to );
        halfAng = 0.0f;
        if ( axis.lengthSq() > 0 )
        {
            halfAng = angle( from, to ) * 0.5f;
            return Matrix3f::rotation( axis, halfAng );
        }
        if ( dot( from, to ) >= 0 )
            return {}; // identity matrix
        halfAng = PI2_F;
        return Matrix3f::rotation( cross( from, from.furthestBasisVector() ), halfAng );
    };

    Mesh res;
    int numEdges = 0;
    auto connectBlocks = [&] ( EdgeId newFirstEdge, EdgeId prevFirstEdge )
    {
        assert( numEdges != 0 );
        auto& tp = res.topology;
        EdgeId firstNewEdgeInLoop;
        for ( int i = 0; i < numEdges; ++i )
        {
            // first edge
            auto newEdge = tp.makeEdge();
            tp.splice( tp.prev( prevFirstEdge + i * 2 ), newEdge );
            tp.splice( newFirstEdge + i * 2, newEdge.sym() );

            if ( !firstNewEdgeInLoop )
                firstNewEdgeInLoop = newEdge;
            else
                tp.setLeft( newEdge.sym(), tp.addFaceId() );

            // diagonal edge
            newEdge = tp.makeEdge();
            tp.splice( tp.prev( prevFirstEdge + i * 2 ), newEdge );
            tp.splice( tp.prev( ( newFirstEdge + i * 2 ).sym() ), newEdge.sym() );
            tp.setLeft( newEdge.sym(), tp.addFaceId() );

            auto diagEdgePrev = tp.prev( newEdge.sym() );
            
            if ( diagEdgePrev == ( newFirstEdge + ( i + 1 ) * 2 ) )
                continue;
            if ( diagEdgePrev == firstNewEdgeInLoop.sym() )
            {
                tp.setLeft( firstNewEdgeInLoop.sym(), tp.addFaceId() );
                firstNewEdgeInLoop = {};
                continue;
            }
            // same as first if path is not finished
            // last in non closed path
            firstNewEdgeInLoop = {};
            newEdge = tp.makeEdge();
            tp.splice( ( prevFirstEdge + i * 2 ).sym(), newEdge );
            tp.splice( diagEdgePrev, newEdge.sym() );
            tp.setLeft( newEdge.sym(), tp.addFaceId() );
        }
    };
    for ( const auto& trajectoryCont : trajectoryContours )
    {
        bool closed = trajectoryCont.size() > 2 && trajectoryCont.front() == trajectoryCont.back();
        EdgeId firstBodyEdge;
        EdgeId prevBodyEdge;
        bool initRotationDone = false;
        prevHalfRot = accumRot = Matrix3f();
        for ( int i = 0; i + ( closed ? 1 : 0 ) < trajectoryCont.size(); ++i )
        {
            const auto& trajPoint = trajectoryCont[i];
            nextVec = prevVec = Vector3f();
            trans = trajPoint - center;
            if ( params.allowRotation )
            {
                if ( i > 0 )
                    prevVec = trajPoint - trajectoryCont[i - 1];
                else if ( closed )
                    prevVec = trajPoint - trajectoryCont[trajectoryCont.size() - 2];
                if ( i + 1 < trajectoryCont.size() )
                    nextVec = trajectoryCont[i + 1] - trajPoint;
                else if ( closed )
                    prevVec = trajPoint - trajectoryCont[1];

                if ( prevVec == Vector3f() )
                    prevVec = nextVec;
                if ( nextVec == Vector3f() )
                    nextVec = prevVec;
                
                if ( !initRotationDone )
                {
                    initRotationDone = true;
                    accumRot = Matrix3f::rotation( normal, prevVec );
                }
                float outHalfAng = 0.0f;
                Vector3f axis;
                auto curHalfRot = halfRot( prevVec, nextVec, axis, outHalfAng );
                accumRot = curHalfRot * prevHalfRot * accumRot;
                prevHalfRot = curHalfRot;
                scaling = Matrix3f{};
                if ( axis.lengthSq() > 0 )
                {
                    auto scaleDir = cross( axis, accumRot * normal );
                    Vector3f basisVec = Vector3f::plusX();
                    if ( std::abs( scaleDir.y ) > std::abs( scaleDir.x ) &&
                        std::abs( scaleDir.y ) > std::abs( scaleDir.z ) )
                        basisVec = Vector3f::plusY();
                    else if ( std::abs( scaleDir.z ) > std::abs( scaleDir.x ) &&
                        std::abs( scaleDir.z ) > std::abs( scaleDir.y ) )
                        basisVec = Vector3f::plusZ();

                    float additionalScale = std::min( 1.0f / std::cos( outHalfAng ), 2.0f ) - 1.0f;

                    scaling = 
                        Matrix3f::rotation( basisVec, scaleDir ) *
                        Matrix3f::scale(Vector3f::diagonal(1) + basisVec * additionalScale ) *
                        Matrix3f::rotation( scaleDir, basisVec );
                }
            }
            xf = AffineXf3f::translation( trans ) * AffineXf3f::xfAround( scaling * accumRot, center );
            if ( params.startMeshFromBody )
            {
                if ( !xf0Inv )
                    xf0Inv = xf.inverse();
                xf = *xf0Inv * xf;
            }
            if ( params.b2tXf )
                xf = xf * ( *params.b2tXf );


            auto curBodyEdge = res.addSeparateContours( bodyContours, &xf );
            if ( !firstBodyEdge )
                firstBodyEdge = curBodyEdge;

            if ( prevBodyEdge )
            {
                if ( numEdges == 0 )
                    numEdges = ( curBodyEdge - prevBodyEdge ) / 2;
                connectBlocks( curBodyEdge, prevBodyEdge );
            }
            prevBodyEdge = curBodyEdge;
        }
        if ( closed )
            connectBlocks( firstBodyEdge, prevBodyEdge );
    }

    return res;
}

}