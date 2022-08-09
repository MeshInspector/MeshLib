#include "MRObjectLabel.h"
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
    if ( !pathToFont_.empty() )
        buildMesh_();
}

void ObjectLabel::setFontPath( const std::filesystem::path& pathToFont )
{
    if ( pathToFont_ == pathToFont )
        return;
    pathToFont_ = pathToFont;
    if ( !label_.text.empty() )
        buildMesh_();
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
    pathToFont_ = GetFontsDirectory() / "DroidSans.ttf";
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
    if ( !mesh_ )
        return {};
    Box3f box;
    box.include( label_.position );
    return box;
}

Box3f ObjectLabel::computeBoundingBoxXf_() const
{
    if ( !mesh_ )
        return {};
    Box3f box;
    box.include( worldXf()( label_.position ) );
    return box;
}

tl::expected<std::future<void>, std::string> ObjectLabel::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !mesh_ )
        return {};

    auto save = [mesh = mesh_, filename = path.u8string() + u8".ctm", this]()
    {
        MR::MeshSave::toCtm( *mesh, filename, {}, vertsColorMap_.empty() ? nullptr : &vertsColorMap_ );
    };

    return std::async( getAsyncLaunchType(), save );
}

tl::expected<void, std::string> ObjectLabel::deserializeModel_( const std::filesystem::path& path )
{
    auto res = MeshLoad::fromCtm( path.u8string() + u8".ctm", &vertsColorMap_ );
    if ( !res.has_value() )
        return tl::make_unexpected( res.error() );

    mesh_ = std::make_shared<Mesh>( std::move( res.value() ) );
    return {};
}

void ObjectLabel::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    root["Text"] = label_.text;
    serializeToJson( label_.position, root["Position"] );
    root["FontHeight"] = fontHeight_;
    
    root["PathToFontFile"] = utf8string( pathToFont_ );

    // append base type
    root["Type"].append( ObjectLabel::TypeName() );
    
    root["SourcePointSize"] = sourcePointSize_;
    root["LeaderLineWidth"] = leaderLineWidth_;
    root["BackgroundPadding"] = backgroundPadding_;

    serializeToJson( sourcePointColor_, root["Colors"]["SourcePoint"] );
    serializeToJson( leaderLineColor_, root["Colors"]["LeaderLine"] );
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

    if ( root["SourcePointSize"].isDouble() )
        sourcePointSize_ = root["SourcePointSize"].asFloat();
    if ( root["LeaderLineWidth"].isDouble() )
        leaderLineWidth_ = root["LeaderLineWidth"].asFloat();
    if ( root["BackgroundPadding"].isDouble() )
        backgroundPadding_ = root["BackgroundPadding"].asFloat();

    deserializeFromJson( root["Colors"]["SourcePoint"], sourcePointColor_ );
    deserializeFromJson( root["Colors"]["LeaderLine"], leaderLineColor_ );
}

void ObjectLabel::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<ObjectLabel>( *this );
}

void ObjectLabel::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::Labels ) );
    setFrontColor( SceneColors::get( SceneColors::Labels ), false );
    setSourcePointColor( Color::gray() );
    setLeaderLineColor( Color::gray() );
}

void ObjectLabel::buildMesh_()
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
    setDirtyFlags( DIRTY_POSITION | DIRTY_FACE );

    updatePivotShift_();
}

void ObjectLabel::updatePivotShift_()
{
    if ( !mesh_ )
        return;
    const Box3f box = mesh_->computeBoundingBox();
    if ( box.valid() )
    {
        Vector3f  diagonal = box.max + box.min; // (box.max - box.min) + box.min * 2 - because box.min != 0
        pivotShift_.x = pivotPoint_.x * diagonal.x;
        pivotShift_.y = pivotPoint_.y * diagonal.y;
    }
}

Box3f ObjectLabel::getWorldBox() const
{
    return computeBoundingBoxXf_();
}

size_t ObjectLabel::heapBytes() const
{
    return VisualObject::heapBytes() +
        sizeof( std::filesystem::path::value_type ) * pathToFont_.native().capacity() +
        label_.text.capacity() +
        MR::heapBytes( mesh_ );
}

void ObjectLabel::applyScale( float scaleFactor )
{
    fontHeight_ *= scaleFactor;
}

bool ObjectLabel::valid() const
{
    return true;
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
    switch ( LabelVisualizePropertyType::Type( type ) )
    {
    case LabelVisualizePropertyType::SourcePoint:
        return sourcePoint_;
    case LabelVisualizePropertyType::Background:
        return background_;
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

void ObjectLabel::setSourcePointColor( const Color &color )
{
    if ( sourcePointColor_ == color )
        return;

    sourcePointColor_ = color;
}

void ObjectLabel::setLeaderLineColor( const Color &color )
{
    if ( leaderLineColor_ == color )
        return;

    leaderLineColor_ = color;
}

}
