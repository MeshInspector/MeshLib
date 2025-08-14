#include "MRViewportGlobalBasis.h"
#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRViewport.h"
#include "MRMesh/MRMesh.h"

namespace
{
constexpr MR::Vector3f cPlusAxis[3] = {
    MR::Vector3f( 1.0f, 0.0f, 0.0f ),
    MR::Vector3f( 0.0f, 1.0f, 0.0f ),
    MR::Vector3f( 0.0f, 0.0f, 1.0f ) };

constexpr std::array<MR::Vector3i, 3> cRotOrders =
{
    MR::Vector3i{2,0,1},
    MR::Vector3i{1,2,0},
    MR::Vector3i{0,1,2}
};
}

namespace MR
{

ViewportGlobalBasis::ViewportGlobalBasis()
{
    auto cylinder = std::make_shared<Mesh>( makeCylinder( 1.0f, 1.0f ) ); // 1/1 is ok, because we actually control size by transform
    for ( int i = 0; i < 3; ++i )
    {
        auto child = std::make_shared<ObjectMesh>();
        child->setMesh( cylinder ); // same mesh for all objects (not really needed to keep it)
        child->setAncillary( true );
        child->setVisualizePropertyMask( MeshVisualizePropertyType::EnableShading, ViewportMask() );
        axes_.emplace_back( std::move( child ) );
    }
    setVisible( false );
    setColors( Color::red(), Color::green(), Color::blue() );
    setAxesProps( 1.0f, 0.01f );
}

float ViewportGlobalBasis::getAxesLength( ViewportId id ) const
{
    if ( axes_.empty() || !axes_[0] )
        return 1.0f;
    return axes_[0]->xf( id ).A.x.z;
}

float ViewportGlobalBasis::getAxesWidth( ViewportId id /*= {} */ ) const
{
    if ( axes_.empty() || !axes_[0] )
        return 1.0f;
    return axes_[0]->xf( id ).A.y.x;
}

void ViewportGlobalBasis::setAxesProps( float length, float width, ViewportId id /*= {} */ )
{
    int i = 0;
    for ( auto& child : visualChildren() )
    {
        child->setXf(
            AffineXf3f::linear(
                Matrix3f( cPlusAxis[cRotOrders[i].x], cPlusAxis[cRotOrders[i].y], cPlusAxis[cRotOrders[i].z] ) * // rotate to correct axis
                Matrix3f::scale( Vector3f( width, width, length ) ) // control width and length
            ) *
            AffineXf3f::translation( Vector3f( 0.0f, 0.0f, -0.1f ) ) // first move 10% backwards
            , id
        );
        ++i;
    }
}

void ViewportGlobalBasis::setColors( const Color& xColor, const Color& yColor, const Color& zColor )
{
    const auto& visChild = visualChildren();
    visChild[0]->setFrontColor( xColor, true ); visChild[0]->setFrontColor( xColor, false );
    visChild[1]->setFrontColor( yColor, true ); visChild[1]->setFrontColor( yColor, false );
    visChild[2]->setFrontColor( zColor, true ); visChild[2]->setFrontColor( zColor, false );
}

const std::vector<std::shared_ptr<MR::VisualObject>>& ViewportGlobalBasis::visualChildren() const
{
    return axes_;
}

bool ViewportGlobalBasis::getRedrawFlag( ViewportMask vpMask ) const
{
    for ( const auto& child : visualChildren() )
        if ( child->getRedrawFlag( vpMask ) )
            return true;
    return false;
}

void ViewportGlobalBasis::resetRedrawFlag() const
{
    for ( const auto& child : visualChildren() )
        child->resetRedrawFlag();
}

void ViewportGlobalBasis::draw( const Viewport& vp ) const
{
    for ( const auto& child : visualChildren() )
        vp.draw( *child, child->xf( vp.id ) );
}

void ViewportGlobalBasis::setVisible( bool on, ViewportMask vpMask /*= ViewportMask::all() */ )
{
    for ( const auto& child : visualChildren() )
        child->setVisible( on, vpMask );
}

}