#include "MRContoursCut.h"
#include "MRMesh.h"
#include "MRAffineXf.h"
#include "MRCoordinateConverters.h"
#include "MRVector3.h"

#pragma managed( push, off )
#include <MRMesh/MRContoursCut.h>
#pragma managed( pop )
MR_DOTNET_NAMESPACE_BEGIN

OneMeshContours^ ContoursCut::GetOneMeshIntersectionContours( Mesh^ meshA, Mesh^ meshB, ContinousContours^ contours, bool getMeshAIntersections,
    CoordinateConverters^ converters )
{
    return GetOneMeshIntersectionContours( meshA, meshB, contours, getMeshAIntersections, converters, nullptr );
}

OneMeshContours^ ContoursCut::GetOneMeshIntersectionContours( Mesh^ meshA, Mesh^ meshB, ContinousContours^ contours, bool getMeshAIntersections,
    CoordinateConverters^ converters, AffineXf3f^ rigidB2A )
{

    MR::ContinuousContours nativeContours(contours->Count);
    for ( int i = 0; i < contours->Count; i++ )
    {
        auto contour = contours[i];
        auto& nativeContour = nativeContours[i];

        nativeContour.reserve( contour->Count);
        for ( int j = 0; j < contour->Count; j++ )
        {
            auto vet = contour[j];            

            nativeContour.emplace_back();
            auto& nativeVet = nativeContour.back();
            nativeVet.edge = MR::EdgeId( vet.edge );
            nativeVet.tri = MR::FaceId( vet.tri );
            nativeVet.isEdgeATriB = vet.isEdgeATriB;
        }
    }

    auto nativeRes = MR::getOneMeshIntersectionContours( *meshA->getMesh(), *meshB->getMesh(), nativeContours, getMeshAIntersections, converters->ToNative(), rigidB2A ? rigidB2A->xf() : nullptr );
    auto res = gcnew OneMeshContours( int( nativeRes.size() ) );
    for ( size_t i = 0; i < nativeRes.size(); i++ )
    {
        auto contour = OneMeshContour();
        contour.intersections = gcnew System::Collections::Generic::List<OneMeshIntersection>( int( nativeRes[i].intersections.size() ) );
        contour.closed = nativeRes[i].closed;

        for ( size_t j = 0; j < nativeRes[i].intersections.size(); j++ )
        {
            OneMeshIntersection omi;
            omi.coordinate = gcnew Vector3f( new MR::Vector3f( nativeRes[i].intersections[j].coordinate ) );

            const auto& nativeOmi = nativeRes[i].intersections[j];
            auto nativeId = nativeOmi.primitiveId;
            switch ( nativeId.index() )
            {
            case 0:
                omi.variantIndex = VariantIndex::Edge;
                omi.index = std::get<0>( nativeId );
                break;
            case 1:
                omi.variantIndex = VariantIndex::Face;
                omi.index = std::get<1>( nativeId );
                break;
            case 2:
                omi.variantIndex = VariantIndex::Vertex;
                omi.index = std::get<2>( nativeId );
                break;
            }
            contour.intersections->Add( omi );
        }
        res->Add( contour );
    }

    return res;
}

MR_DOTNET_NAMESPACE_END