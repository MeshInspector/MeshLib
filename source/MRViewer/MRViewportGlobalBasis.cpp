#include "MRViewportGlobalBasis.h"
#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRViewport.h"
#include "MRMesh/MRMesh.h"
#include "MRSymbolMesh/MRObjectLabel.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRPositionedText.h"
#include "MRMesh/MRSceneColors.h"
#include "MRColorTheme.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"

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
    auto menu = getViewerInstance().getMenuPlugin();
    auto cylinder = std::make_shared<Mesh>( makeCylinder( 1.0f, 1.0f ) ); // 1/1 is ok, because we actually control size by transform
    for ( int i = 0; i < 3; ++i )
    {
        auto child = std::make_shared<ObjectMesh>();
        child->setMesh( cylinder ); // same mesh for all objects (not really needed to keep it)
        child->setAncillary( true );
        child->setVisualizePropertyMask( MeshVisualizePropertyType::EnableShading, ViewportMask() );
        
        auto label = std::make_shared<ObjectLabel>();
        label->setPivotPoint( Vector2f( 0.5f, 0.5f ) );
        label->setLabel( PositionedText( ( i == 0 ? "X" : ( i == 1 ? "Y" : "Z" ) ), Vector3f( 0, 0, 1.05f ) ) );
        label->setFontHeight( menu ? 20 * menu->menu_scaling() : 20.0f );
        label->setAncillary( true );
        child->addChild( label );

        axes_.emplace_back( std::move( child ) );
    }
    setVisible( false );
    setAxesProps( 1.0f, 0.01f );

    auto updateColors = [this] ()
    {
        const Color& colorX = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisX );
        const Color& colorY = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisY );
        const Color& colorZ = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisZ );
        const Color& labelColor = SceneColors::get( SceneColors::Type::Labels );

        setColors( colorX, colorY, colorZ, labelColor );
    };
    updateColors();

    connections_.emplace_back( ColorTheme::onChanged( updateColors ) );
    connections_.emplace_back( getViewerInstance().postRescaleSignal.connect( [this] ( float, float )
    {
        auto menu = getViewerInstance().getMenuPlugin();
        if ( !menu )
            return;
        for ( const auto& child : axes_ )
            for ( const auto& label : child->children() )
                if ( auto* visLabel = label->asType<ObjectLabel>() )
                    visLabel->setFontHeight( 20.0f * menu->menu_scaling() );
    } ) );

    creteGrids_();
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
    for ( auto& child : axesChildren() )
    {
        child->setXf(
            AffineXf3f::linear(
                Matrix3f( cPlusAxis[cRotOrders[i].x], cPlusAxis[cRotOrders[i].y], cPlusAxis[cRotOrders[i].z] ) * // rotate to correct axis
                Matrix3f::scale( Vector3f( width, width, length ) ) // control width and length
            )
            , id
        );
        ++i;
    }
}

void ViewportGlobalBasis::setColors( const Color& xColor, const Color& yColor, const Color& zColor, const Color& labelColors )
{
    const auto& visChild = axesChildren();
    visChild[0]->setFrontColor( xColor, true ); visChild[0]->setFrontColor( xColor, false );
    visChild[1]->setFrontColor( yColor, true ); visChild[1]->setFrontColor( yColor, false );
    visChild[2]->setFrontColor( zColor, true ); visChild[2]->setFrontColor( zColor, false );
    for ( const auto& child : visChild )
        for ( const auto& label : child->children() )
            if ( auto* visLabel = label->asType<VisualObject>() )
            {
                visLabel->setFrontColor( labelColors, true ); visLabel->setFrontColor( labelColors, false );
            }
}

const std::vector<std::shared_ptr<MR::VisualObject>>& ViewportGlobalBasis::axesChildren() const
{
    return axes_;
}

bool ViewportGlobalBasis::getRedrawFlag( ViewportMask vpMask ) const
{
    for ( const auto& child : axesChildren() )
        if ( child->getRedrawFlag( vpMask ) )
            return true;
    for ( const auto& child : grids_ )
        if ( child->getRedrawFlag( vpMask ) )
            return true;
    return false;
}

void ViewportGlobalBasis::resetRedrawFlag() const
{
    for ( const auto& child : axesChildren() )
        child->resetRedrawFlag();
    for ( const auto& child : grids_ )
        child->resetRedrawFlag();
}

void ViewportGlobalBasis::draw( const Viewport& vp ) const
{
    for ( const auto& child : axesChildren() )
    {
        const auto& xf = child->xf( vp.id );
        vp.draw( *child, xf );
        for ( const auto& label : child->children() )
            if ( auto* visLabel = label->asType<VisualObject>() )
                vp.draw( *visLabel, xf );
    }
    if ( !isGridVisible( vp.id ) )
        return;
    updateGridXfs_( vp );
    for ( const auto& child : grids_ )
    {
        const auto& xf = child->xf( vp.id );
        vp.draw( *child, xf );
    }
}

void ViewportGlobalBasis::setVisible( bool on, ViewportMask vpMask /*= ViewportMask::all() */ )
{
    for ( const auto& child : axesChildren() )
        child->setVisible( on, vpMask );
}

void ViewportGlobalBasis::setGridVisible( bool on, ViewportMask vpMask /*= ViewportMask::all() */ )
{
    for ( const auto& child : grids_ )
        child->setVisible( on, vpMask );
}

void ViewportGlobalBasis::creteGrids_()
{
    grids_.clear();
    grids_.push_back( std::make_shared<ObjectLines>() ); // thin grid
    grids_.push_back( std::make_shared<ObjectLines>() ); // thick grid
    auto& thinGrid = grids_[0];
    auto& thickGrid = grids_[1];

    constexpr int cNumSegments = 50;
    Contours3f conts( cNumSegments * 4 + 2, Contour3f( 2 ) );
    int id = 0;
    for ( int i = -cNumSegments; i <= cNumSegments; ++i )
    {
        auto otherId = id + 2 * cNumSegments + 1;
        conts[id][0].x = conts[id][1].x = conts[otherId][0].y = conts[otherId][1].y = float( i );
        conts[id][0].y = conts[otherId][0].x = -cNumSegments;
        conts[id][1].y = conts[otherId][1].x = cNumSegments;
        id++;
    }
    auto pl = std::make_shared<Polyline3>( conts );
    thinGrid->setPolyline( pl );
    thickGrid->setPolyline( pl );
    thinGrid->setLineWidth( 0.5f );
    thickGrid->setLineWidth( 1.5f );
    thinGrid->setFrontColor( Color::gray(), true );
    thinGrid->setFrontColor( Color::gray(), false );
    thickGrid->setFrontColor( Color::gray(), true );
    thickGrid->setFrontColor( Color::gray(), false );
}

void ViewportGlobalBasis::updateGridXfs_( const Viewport& vp ) const
{
    if ( grids_.empty() )
        return;
    auto forwardVec = -vp.getBackwardDirection().normalized();
    Matrix3f rot = cachedGridRotation_.get( vp.id );
    if ( std::abs( dot( forwardVec, rot.col( 2 ) ) ) < 0.2f ) // change grid orientation only on degeneracy
    {
        auto dotX = std::abs( dot( forwardVec, Vector3f::plusX() ) );
        auto dotY = std::abs( dot( forwardVec, Vector3f::plusY() ) );
        auto dotZ = std::abs( dot( forwardVec, Vector3f::plusZ() ) );
        if ( dotZ >= dotX && dotZ >= dotY )
            rot = Matrix3f();
        else if ( dotY >= dotX && dotY >= dotZ )
            rot = Matrix3f::fromColumns( Vector3f::plusZ(), Vector3f::plusX(), Vector3f::plusY() );
        else
            rot = Matrix3f::fromColumns( Vector3f::plusY(), Vector3f::plusZ(), Vector3f::plusX() );
        cachedGridRotation_.set( rot, vp.id );
    }

    auto halfScreenWorldSize = vp.getPixelSizeAtPoint( Vector3f() ) * vp.getViewportRect().diagonal() * 0.3f;
    auto gridScaling = std::pow( 10.0f, std::round( std::log10( halfScreenWorldSize ) ) );

    int mainAxis = 2;
    if ( rot.z.y == 1.0f )
        mainAxis = 0;
    else if ( rot.z.x == 1.0f )
        mainAxis = 1;
    auto camera = vp.getCameraPoint();
    auto zeroPlanePoint = camera - forwardVec * ( camera[mainAxis] / forwardVec[mainAxis] );

    auto closestNicePlanePoint = zeroPlanePoint / gridScaling;
    for ( int i = 0; i < 3; ++i )
        closestNicePlanePoint[i] = std::round( closestNicePlanePoint[i] ) * gridScaling;

    grids_[0]->setXf( AffineXf3f::translation( closestNicePlanePoint ) * AffineXf3f::linear( rot * Matrix3f::scale( gridScaling * 0.2f ) ), vp.id );
    grids_[1]->setXf( AffineXf3f::translation( closestNicePlanePoint ) * AffineXf3f::linear( rot * Matrix3f::scale( gridScaling ) ), vp.id );
}

}