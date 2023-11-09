#include "MRObjectLabel.h"
#ifndef MRMESH_NO_LABEL
#include "MRObjectFactory.h"
#include "MRSerializer.h"
#include "MRMesh.h"
#include "MRSceneColors.h"
#include "MRMeshSave.h"
#include "MRMeshLoad.h"
#include "MRStringConvert.h"
#include "MR2DContoursTriangulation.h"
#include "MRPch/MRAsyncLaunchType.h"
#include "MRPch/MRJson.h"
#include "MRString.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectLabel )

void ObjectLabel::setLabel( const PositionedText& label )
{
    if ( label == label_ )
        return;
    label_ = label;
    needRebuild_ = true;
    setDirtyFlags( DIRTY_POSITION | DIRTY_FACE );
}

void ObjectLabel::setFontPath( const std::filesystem::path& pathToFont )
{
    if ( pathToFont_ == pathToFont )
        return;
    pathToFont_ = pathToFont;
    needRebuild_ = true;
    setDirtyFlags( DIRTY_POSITION | DIRTY_FACE );
}

void ObjectLabel::setPivotPoint( const Vector2f& pivotPoint )
{
    if ( pivotPoint == pivotPoint_ )
        return;

    pivotPoint_ = pivotPoint;
    updatePivotShift_();
}

ObjectLabel::ObjectLabel()
{
    setDefaultColors_();

    // set default path to font if available
#ifndef __EMSCRIPTEN__
    pathToFont_ = GetFontsDirectory() / "NotoSansSC-Regular.otf";
#else
    pathToFont_ = GetFontsDirectory() / "NotoSans-Regular.ttf";
#endif
    std::error_code ec;
    if ( !std::filesystem::is_regular_file( pathToFont_, ec ) )
        pathToFont_.clear();
}

void ObjectLabel::swapBase_( Object& other )
{
    if ( auto otherLabelObject = other.asType<ObjectLabel>() )
        std::swap( *this, *otherLabelObject );
    else
        assert( false );
}

Box3f ObjectLabel::computeBoundingBox_() const
{
    Box3f box;
    box.include( label_.position );
    return box;
}

void ObjectLabel::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    root["Text"] = label_.text;
    serializeToJson( label_.position, root["Position"] );
    root["FontHeight"] = fontHeight_;
    
    root["PathToFontFile"] = utf8string( pathToFont_ );

    root["SourcePoint"] = sourcePoint_.value();
    root["Background"] = background_.value();
    root["Contour"] = contour_.value();
    root["LeaderLine"] = leaderLine_.value();

    // append base type
    root["Type"].append( ObjectLabel::TypeName() );
    
    root["SourcePointSize"] = sourcePointSize_;
    root["LeaderLineWidth"] = leaderLineWidth_;
    root["BackgroundPadding"] = backgroundPadding_;

    serializeToJson( pivotPoint_, root["PivotPoint"] );

    serializeToJson( sourcePointColor_.get(), root["Colors"]["SourcePoint"] );
    serializeToJson( leaderLineColor_.get(), root["Colors"]["LeaderLine"] );
    serializeToJson( contourColor_.get(), root["Colors"]["Contour"] );
}

void ObjectLabel::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );

    deserializeFromJson( root["Position"], label_.position );
    if ( root["FontHeight"].isDouble() )
        fontHeight_ = root["FontHeight"].asFloat();
    if ( root["Text"].isString() )
        label_.text = root["Text"].asString();
    if ( root["PathToFontFile"].isString() )
        pathToFont_ = root["PathToFontFile"].asString();

    if ( root["SourcePoint"].isUInt() )
        sourcePoint_ = ViewportMask( root["SourcePoint"].asUInt() );
    if ( root["Background"].isUInt() )
        background_ = ViewportMask( root["Background"].asUInt() );
    if ( root["Contour"].isUInt() )
        contour_ = ViewportMask( root["Contour"].asUInt() );
    if ( root["LeaderLine"].isUInt() )
        leaderLine_ = ViewportMask( root["LeaderLine"].asUInt() );

    if ( root["SourcePointSize"].isDouble() )
        sourcePointSize_ = root["SourcePointSize"].asFloat();
    if ( root["LeaderLineWidth"].isDouble() )
        leaderLineWidth_ = root["LeaderLineWidth"].asFloat();
    if ( root["BackgroundPadding"].isDouble() )
        backgroundPadding_ = root["BackgroundPadding"].asFloat();

    deserializeFromJson( root["PivotPoint"], pivotPoint_ );

    deserializeFromJson( root["Colors"]["SourcePoint"], sourcePointColor_.get() );
    deserializeFromJson( root["Colors"]["LeaderLine"], leaderLineColor_.get() );
    deserializeFromJson( root["Colors"]["Contour"], contourColor_.get() );

    needRebuild_ = true;
}

void ObjectLabel::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<ObjectLabel>( *this );

    if ( needRebuild_ && !label_.text.empty() && !pathToFont_.empty() )
        buildMesh_();
}

void ObjectLabel::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::Labels ), true );
    setFrontColor( SceneColors::get( SceneColors::Labels ), false );
    setSourcePointColor( Color::gray() );
    setLeaderLineColor( Color::gray() );
    setContourColor( Color::gray() );
}

void ObjectLabel::buildMesh_() const
{
    std::vector<std::string> splited = split( label_.text, "\n" );

    mesh_ = std::make_shared<Mesh>();
    for ( int i = 0; i < splited.size(); ++i )
    {
        const auto& s = splited[i];
        if ( s.empty() )
            continue;
        SymbolMeshParams params;
        params.text = s;
        params.pathToFontFile = pathToFont_;
        auto contours = createSymbolContours( params );
        auto mesh = PlanarTriangulation::triangulateContours( contours );
        // 1.3f - line spacing
        mesh.transform( AffineXf3f::translation( 
            Vector3f::minusY() * SymbolMeshParams::MaxGeneratedFontHeight * 1.3f * float( i ) ) );

        mesh_->addPart( mesh );
    }

    meshBox_ = mesh_->computeBoundingBox();
    updatePivotShift_();

    // important to call before bindAllVisualization to avoid recursive calls
    needRebuild_ = false;

    // we can always clear cpu model for labels
    bindAllVisualization();
    mesh_.reset();
}

void ObjectLabel::updatePivotShift_() const
{
    if ( !meshBox_.valid() )
        return;
    Vector3f  diagonal = meshBox_.max + meshBox_.min; // (box.max - box.min) + box.min * 2 - because box.min != 0
    pivotShift_.x = pivotPoint_.x * diagonal.x;
    pivotShift_.y = pivotPoint_.y * diagonal.y;
}

Box3f ObjectLabel::getWorldBox( ViewportId id ) const
{
    Box3f box;
    box.include( worldXf( id )( label_.position ) );
    return box;
}

size_t ObjectLabel::heapBytes() const
{
    return VisualObject::heapBytes() +
        sizeof( pathToFont_ ) * pathToFont_.native().capacity() +
        label_.text.capacity() +
        MR::heapBytes( mesh_ );
}

void ObjectLabel::applyScale( float scaleFactor )
{
    fontHeight_ *= scaleFactor;
}

std::shared_ptr<MR::Object> ObjectLabel::clone() const
{
    auto res = std::make_shared<ObjectLabel>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    return res;
}

std::shared_ptr<MR::Object> ObjectLabel::shallowClone() const
{
    auto res = std::make_shared<ObjectLabel>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    return res;
}

AllVisualizeProperties ObjectLabel::getAllVisualizeProperties() const
{
    AllVisualizeProperties res;
    res.resize( LabelVisualizePropertyType::LabelVisualizePropsCount );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = getVisualizePropertyMask( unsigned( i ) );
    return res;
}

const ViewportMask &ObjectLabel::getVisualizePropertyMask( unsigned int type ) const
{
    switch ( type )
    {
    case LabelVisualizePropertyType::SourcePoint:
        return sourcePoint_;
    case LabelVisualizePropertyType::Background:
        return background_;
    case LabelVisualizePropertyType::Contour:
        return contour_;
    case LabelVisualizePropertyType::LeaderLine:
        return leaderLine_;
    default:
        return VisualObject::getVisualizePropertyMask( type );
    }
}

void ObjectLabel::setLeaderLineWidth( float width )
{
    if ( leaderLineWidth_ == width )
        return;

    leaderLineWidth_ = width;
    needRedraw_ = true;
}

void ObjectLabel::setSourcePointSize( float size )
{
    if ( sourcePointSize_ == size )
        return;

    sourcePointSize_ = size;
    needRedraw_ = true;
}

void ObjectLabel::setBackgroundPadding( float padding )
{
    if ( backgroundPadding_ == padding )
        return;

    backgroundPadding_ = padding;
    needRedraw_ = true;
}

void ObjectLabel::setSourcePointColor( const Color &color, ViewportId id )
{
    if ( sourcePointColor_.get( id ) == color )
        return;

    sourcePointColor_.set( color, id );
    needRedraw_ = true;
}

void ObjectLabel::setLeaderLineColor( const Color &color, ViewportId id )
{
    if ( leaderLineColor_.get( id ) == color )
        return;

    leaderLineColor_.set( color, id );
    needRedraw_ = true;
}

void ObjectLabel::setContourColor( const Color& color, ViewportId id )
{
    if ( contourColor_.get( id ) == color )
        return;

    contourColor_.set( color, id );
    needRedraw_ = true;
}

const ViewportProperty<Color>& ObjectLabel::getSourcePointColorsForAllViewports() const
{
    return sourcePointColor_;
}

void ObjectLabel::setSourcePointColorsForAllViewports( ViewportProperty<Color> val )
{
    sourcePointColor_ = std::move( val );
    needRedraw_ = true;
}

const ViewportProperty<Color>& ObjectLabel::getLeaderLineColorsForAllViewports() const
{
    return leaderLineColor_;
}

void ObjectLabel::setLeaderLineColorsForAllViewports( ViewportProperty<Color> val )
{
    leaderLineColor_ = std::move( val );
    needRedraw_ = true;
}

const ViewportProperty<Color>& ObjectLabel::getContourColorsForAllViewports() const
{
    return contourColor_;
}

void ObjectLabel::setContourColorsForAllViewports( ViewportProperty<Color> val )
{
    contourColor_ = std::move( val );
    needRedraw_ = true;
}

}
#endif
