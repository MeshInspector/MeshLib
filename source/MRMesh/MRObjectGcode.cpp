#include "MRObjectGcode.h"
#include "MRObjectFactory.h"
#include "MRPolyline.h"
#include "MRPch/MRJson.h"
#include "MRSerializer.h"
#include "MRSceneSettings.h"
#include "MRTimer.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectGcode )

ObjectGcode::ObjectGcode()
{
    setVisualizeProperty( true, LinesVisualizePropertyType::Smooth, ViewportMask::all() );
    setColoringType( ColoringType::VertsColorMap );
    setLineWidth( 3.f );
    cncMachineSettings_ = SceneSettings::getCNCMachineSettings();
}



std::shared_ptr<Object> MR::ObjectGcode::clone() const
{
    auto res = std::make_shared<ObjectGcode>( ProtectedStruct{}, *this );
    if ( gcodeSource_ )
        res->setGcodeSource( std::make_shared<GcodeSource>( *gcodeSource_ ) );
    return res;
}

std::shared_ptr<Object> ObjectGcode::shallowClone() const
{
    auto res = std::make_shared<ObjectGcode>( ProtectedStruct{}, *this );
    if ( gcodeSource_ )
        res->setGcodeSource( gcodeSource_ );
    return res;
}

void ObjectGcode::setCNCMachineSettings( const CNCMachineSettings& cncSettings )
{
    cncMachineSettings_ = cncSettings;
    updateAll_();
}

void ObjectGcode::setGcodeSource( const std::shared_ptr<GcodeSource>& gcodeSource )
{
    gcodeSource_ = gcodeSource;
    updateAll_();
}

void ObjectGcode::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    ObjectLinesHolder::setDirtyFlags( mask, invalidateCaches );

    if ( mask & DIRTY_POSITION || mask & DIRTY_PRIMITIVES )
    {
        if ( polyline_ )
        {
            gcodeChangedSignal( mask );
        }
    }
}

std::vector<std::string> ObjectGcode::getInfoLines() const
{
    std::vector<std::string> res = ObjectLinesHolder::getInfoLines();

    std::stringstream ss;
    if ( polyline_ )
    {
        ss << "vertices : " << polyline_->topology.numValidVerts();
        res.push_back( ss.str() );

        if ( !totalLength_ )
            totalLength_ = polyline_->totalLength();
        res.push_back( "total length : " + std::to_string( *totalLength_ ) );

        boundingBoxToInfoLines_( res );
    }
    else
    {
        res.push_back( "no polyline" );
    }

    return res;
}

void ObjectGcode::switchFeedrateGradient( bool isFeedrateGradientEnabled )
{
    if ( feedrateGradientEnabled_ == isFeedrateGradientEnabled )
        return;
    feedrateGradientEnabled_ = isFeedrateGradientEnabled;
    updateColors_();
}

void ObjectGcode::setIdleColor( const Color& color )
{
    if ( idleColor_ == color )
        return;
    idleColor_ = color;

    updateColors_();
}

bool ObjectGcode::select( bool isSelected )
{
    if ( !ObjectLinesHolder::select( isSelected ) )
        return false;
    float width = getLineWidth();
    if ( isSelected )
        width += std::clamp( width, 3.0f, 6.0f );
    else
    {
        width -= std::clamp( width * 0.5f, 3.0f, 6.0f );
        if ( width < 0.5f )
            width = 0.5f; // minimum width
    }
    setLineWidth( width );
    return true;
}

void ObjectGcode::setFrontColor( const Color& color, bool selected, ViewportId viewportId /*= {} */ )
{
    ObjectLinesHolder::setFrontColor( color, selected, viewportId );
    if ( selected )
        updateColors_();
}

size_t ObjectGcode::heapBytes() const
{
    return ObjectLinesHolder::heapBytes() +
        MR::heapBytes( segmentToSourceLineMap_ ) +
        nonTrivialHeapUsageCache_;
}

void ObjectGcode::swapBase_( Object& other )
{
    if ( auto otherGcode = other.asType<ObjectGcode>() )
        std::swap( *this, *otherGcode );
    else
        assert( false );
}

void ObjectGcode::swapSignals_( Object& other )
{
    ObjectLinesHolder::swapSignals_( other );
    if ( auto otherGcode = other.asType<ObjectGcode>() )
        std::swap( gcodeChangedSignal, otherGcode->gcodeChangedSignal );
    else
        assert( false );
}

void ObjectGcode::serializeFields_( Json::Value& root ) const
{
    ObjectLinesHolder::serializeBaseFields_( root );
    root["Type"].append( ObjectGcode::TypeName() );
    root["FeedrateGradientEnable"] = feedrateGradientEnabled_;
    root["MaxFeedrate"] = maxFeedrate_;
    serializeToJson( idleColor_, root["IdleColor"] );

    auto& gcodeSourceRoot = root["GcodeSource"];
    for ( const auto& str : *gcodeSource_ )
        gcodeSourceRoot.append( str );
}

void ObjectGcode::deserializeFields_( const Json::Value& root )
{
    ObjectLinesHolder::deserializeBaseFields_( root );
    deserializeFromJson( root["IdleColor"], idleColor_ );

    if ( root["FeedrateGradientEnable"].isBool() )
        feedrateGradientEnabled_ = root["FeedrateGradientEnable"].asBool();
    if ( root["MaxFeedrate"].isDouble() )
        maxFeedrate_ = float( root["MaxFeedrate"].asDouble() );

    const auto& gcodeSourceRoot = root["GcodeSource"];
    if ( !gcodeSourceRoot.isArray() )
        return;
    GcodeSource gcodeSource( gcodeSourceRoot.size() );
    for ( int i = 0; i < gcodeSource.size(); ++i )
    {
        if ( gcodeSourceRoot[i].isString() )
            gcodeSource[i] = gcodeSourceRoot[i].asString();
    }
    setGcodeSource( std::make_shared<GcodeSource>( std::move( gcodeSource ) ) );
}

void ObjectGcode::updateHeapUsageCache_()
{
    nonTrivialHeapUsageCache_ = 0;
    if ( gcodeSource_ )
    {
        nonTrivialHeapUsageCache_ += sizeof( GcodeSource );
        nonTrivialHeapUsageCache_ += sizeof( std::string ) * gcodeSource_->capacity();
        for ( int i = 0; i < gcodeSource_->size(); ++i )
            nonTrivialHeapUsageCache_ += ( *gcodeSource_ )[i].capacity();
    }
    nonTrivialHeapUsageCache_ += sizeof( GcodeProcessor::MoveAction ) * actionList_.capacity();
    for ( int i = 0; i < actionList_.size(); ++i )
    {
        nonTrivialHeapUsageCache_ += actionList_[i].action.warning.capacity();
        nonTrivialHeapUsageCache_ += MR::heapBytes( actionList_[i].action.path );
    }
}

void ObjectGcode::updateColors_()
{
    MR_TIMER
    const bool feedrateValid = maxFeedrate_ > 0.f;
    VertColors colors;
    const Color workColor = getFrontColor( true );
    for ( int i = 0; i < actionList_.size(); ++i )
    {
        const auto& part = actionList_[i];
        if ( part.action.path.empty() )
            continue;
        Color color = idleColor_;
        if ( !part.idle )
        {
            if ( feedrateGradientEnabled_ && feedrateValid )
            {
                color = workColor * ( 0.3f + 0.7f * part.feedrate / maxFeedrate_ );
                color.a = 255;
            }
            else
                color = workColor;
        }
        colors.autoResizeSet( VertId( colors.size() ), part.action.path.size(), color );
    }
    setVertsColorMap( colors );
}

void ObjectGcode::updateAll_()
{
    if ( !gcodeSource_ )
    {
        polyline_ = std::make_shared<Polyline3>();
        setDirtyFlags( DIRTY_ALL );
        return;
    }
    MR_TIMER
    GcodeProcessor executor;
    executor.setCNCMachineSettings( cncMachineSettings_ );
    executor.setGcodeSource( *gcodeSource_ );
    actionList_ = executor.processSource();

    maxFeedrate_ = 0.f;
    std::shared_ptr<Polyline3> polyline = std::make_shared<Polyline3>();
    for ( int i = 0; i < actionList_.size(); ++i )
    {
        const auto& part = actionList_[i];
        if ( part.action.path.empty() )
            continue;
        polyline->addFromPoints( part.action.path.data(), part.action.path.size(), false );
        segmentToSourceLineMap_.insert( segmentToSourceLineMap_.end(), part.action.path.size() - 1, i );
        if ( !part.idle && part.feedrate > maxFeedrate_ )
            maxFeedrate_ = part.feedrate;
    }
    polyline_ = polyline;
    updateColors_();
    updateHeapUsageCache_();
    setDirtyFlags( DIRTY_ALL );
}

} //namespace MR
