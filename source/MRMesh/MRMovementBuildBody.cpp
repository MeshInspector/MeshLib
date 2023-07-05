#include "MRMovementBuildBody.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRAffineXf3.h"
#include "MRTimer.h"
#include "MRMatrix3.h"

namespace MR
{

Mesh makeMovementBuildBody( const Polyline3& body, const Polyline3& trajectory, bool allowRotation, const AffineXf3f* b2tXf /*= nullptr */ )
{
    MR_TIMER;

    auto bodyContours = body.contours();
    auto trajectoryContours = body.contours();
    // filter same points in trajectory
    for ( auto& trajC : trajectoryContours )
        trajC.erase( std::unique( trajC.begin(), trajC.end() ), trajC.end() );

    AffineXf3f xf;
    Vector3f trans;
    Matrix3f prevHalfRot;
    Matrix3f accumRot;
    auto rotationCenter = body.computeBoundingBox().center();
    Vector3f basePos = trajectoryContours.front().front();
    Vector3f prevVec, nextVec; // needed for soft rotation


    auto halfRot = [] ( const Vector3f& from, const Vector3f& to )->Matrix3f
    {
        auto axis = cross( from, to );
        if ( axis.lengthSq() > 0 )
            return Matrix3f::rotation( axis, angle( from, to ) * 0.5f );
        if ( dot( from, to ) >= 0 )
            return {}; // identity matrix
        return Matrix3f::rotation( cross( from, from.furthestBasisVector() ), PI2_F );
    };

    Mesh res;
    EdgeId firstBodyEdge;
    EdgeId prevBodyEdge;
    int numEdges = 0;
    auto connectBlocks = [&] ( EdgeId newEdge, EdgeId prevEdge )
    {
        assert( numEdges != 0 );
        auto& tp = res.topology;
        for ( int i = 0; i < numEdges; ++i )
        {
            // connect pairs
        }
    };
    for ( const auto& trajectoryCont : trajectoryContours )
    {
        EdgeId curBodyEdge;
        bool closed = trajectoryCont.size() > 2 && trajectoryCont.front() == trajectoryCont.back();
        firstBodyEdge = {};
        for ( int i = 0; i + ( closed ? 1 : 0 ) < trajectoryCont.size(); ++i )
        {
            const auto& trajPoint = trajectoryCont[i];
            nextVec = prevVec = Vector3f();
            trans = trajPoint - basePos;
            if ( allowRotation )
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
                
                auto curHalfRot = halfRot( prevVec, nextVec );
                accumRot = curHalfRot * prevHalfRot * accumRot;
            }
            xf = AffineXf3f::translation( trans ) * AffineXf3f::xfAround( accumRot, rotationCenter );

            curBodyEdge = res.addSeparateContours( bodyContours, &xf );
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