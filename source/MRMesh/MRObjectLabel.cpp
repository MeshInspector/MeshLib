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
}

void ObjectLabel::buildMesh_()
{
    SymbolMeshParams params;
    params.text = label_.text;
    params.pathToFontFile = pathToFont_;
    auto contours = createSymbolContours( params );
    mesh_ = std::make_shared<Mesh>( PlanarTriangulation::triangulateContours( contours ) );
    setDirtyFlags( DIRTY_POSITION | DIRTY_FACE );
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

}