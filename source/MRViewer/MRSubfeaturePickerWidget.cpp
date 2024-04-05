#include "MRSubfeaturePickerWidget.h"

#include "MRMesh/MRObjectDimensionsEnum.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRVisualSubfeatures.h"

#include <unordered_set>

namespace MR
{

const Color cColorActive( 0, 255, 0, 255 );
const Color cColorHover( 255, 128, 0, 255 );

static void hideFeatureDimensions( VisualObject& vis )
{
    for ( std::size_t i = 0; i < std::size_t( DimensionsVisualizePropertyType::_count ); i++ )
    {
        auto enumValue = DimensionsVisualizePropertyType( i );
        if ( vis.supportsVisualizeProperty( enumValue ) )
            vis.setVisualizePropertyMask( enumValue, {} );
    }
}

static void highlightFeature( FeatureObject& feat, Color color )
{
    feat.setVisible( true );

    float hoverPointSize = 12;
    float hoverLineWidth = 5;

    feat.setFrontColor( color, false );
    feat.setFrontColor( color, true );
    feat.setBackColor( color );
    feat.setDecorationsColor( color, false );
    feat.setDecorationsColor( color, true );
    feat.setPointSize( std::max( hoverPointSize, feat.getPointSize() ) );
    feat.setLineWidth( std::max( hoverLineWidth, feat.getLineWidth() ) );

    hideFeatureDimensions( feat );
}

float SubfeaturePickerWidgetLow::defaultInfiniteExtent()
{
    float ret = 0.1f; // An arbitrary small starting value.
    for ( const auto& viewport : getViewerInstance().viewport_list )
    {
        auto box = viewport.getSceneBox();
        if ( box.valid() )
            ret = std::max( ret, box.diagonal() );
    }
    return ret * 0.5f;
}

const SubfeaturePickerWidgetLow::ObjectMap& SubfeaturePickerWidgetLow::getObjects() const
{
    return objects_;
}

void SubfeaturePickerWidgetLow::updateTemporaryObjects( ObjectPred objectPredicate )
{
    std::optional<float> infiniteExtent;

    // Returns null on failure.
    auto addSubfeatures = [&]( const std::shared_ptr<FeatureObject>& target ) -> std::shared_ptr<ObjectData>
    {
        std::shared_ptr<ObjectData> ret;
        if ( auto opt = Features::primitiveFromObject( *target ) )
        {
            ret = std::make_shared<ObjectData>();
            ret->feature = std::move( *opt );
        }
        else
        {
            return ret; // Always null here.
        }

        std::optional<AffineXf3f> invTargetWorldXf;

        Features::forEachVisualSubfeature( ret->feature, [&]( const Features::SubfeatureInfo& info )
        {
            if ( info.isInfinite )
                return;

            if ( !infiniteExtent )
                infiniteExtent = getInfiniteExtent();

            SubfeatureData newSubfeatureData;
            newSubfeatureData.name = info.name;
            newSubfeatureData.subfeature = info.create();
            newSubfeatureData.object = Features::primitiveToObject( newSubfeatureData.subfeature, *infiniteExtent );
            if ( !newSubfeatureData.object )
                return;

            if ( !invTargetWorldXf )
                invTargetWorldXf = target->worldXf().inverse();

            newSubfeatureData.object->setXf( *invTargetWorldXf * newSubfeatureData.object->xf() );

            // Disable subfeatures.
            newSubfeatureData.object->setVisualizePropertyMask( FeatureVisualizePropertyType::Subfeatures, {} );
            // Disable measurements, just in case.
            for ( int i = 0; i < int( DimensionsVisualizePropertyType::_count ); i++ )
            {
                auto enumValue = DimensionsVisualizePropertyType( i );
                if ( newSubfeatureData.object->supportsVisualizeProperty( enumValue ) )
                    newSubfeatureData.object->setVisualizePropertyMask( enumValue, {} );
            }

            // Hide by default.
            newSubfeatureData.object->setVisible( false );

            newSubfeatureData.object->setAncillary( true );
            newSubfeatureData.object->setPickable( false );

            target->addChild( newSubfeatureData.object );

            ret->subfeatures.push_back( std::move( newSubfeatureData ) );
        } );

        return ret;
    };

    std::unordered_set<std::shared_ptr<Object>> visitedObjects;

    auto lambda = [&]( auto lambda, const std::shared_ptr<Object>& cur, ViewportMask visibilityMask, bool isAncillary ) -> void
    {
        visibilityMask &= cur->visibilityMask();
        if ( visibilityMask.empty() )
            return;

        isAncillary = isAncillary || cur->isAncillary();
        if ( !allowAncillaryObjects && isAncillary )
            return;

        if ( auto feature = std::dynamic_pointer_cast<FeatureObject>( cur ) )
        {
            if ( !findSubfeature( *feature ) ) // Refuse to recurse into our own subfeatures.
            {
                ViewportMask subfeatureVisMask = visibilityMask;

                if ( respectSubfeatureVisualProperty )
                    subfeatureVisMask &= feature->getVisualizePropertyMask( FeatureVisualizePropertyType::Subfeatures );

                if ( !subfeatureVisMask.empty() )
                {
                    if ( !objectPredicate || objectPredicate( *feature, isAncillary ) )
                    {
                        auto iter = objects_.find( cur );
                        bool isNew = iter == objects_.end();

                        bool ok = true;
                        if ( isNew )
                        {
                            if ( auto data = addSubfeatures( feature ) )
                                iter = objects_.insert_or_assign( cur, std::move( data ) ).first;
                            else
                                ok = false;
                        }

                        if ( ok )
                            visitedObjects.insert( feature );
                    }
                }
            }
        }

        for ( const auto& child : cur->children() )
            lambda( lambda, child, visibilityMask, isAncillary );
    };
    lambda( lambda, SceneRoot::getSharedPtr(), getViewerInstance().getPresentViewports(), false );

    for ( auto it = objects_.begin(); it != objects_.end(); )
    {
        if ( !visitedObjects.contains( it->first ) )
        {
            for ( const SubfeatureData& sub : it->second->subfeatures )
                sub.object->detachFromParent();

            it = objects_.erase( it );
        }
        else
        {
            it++;
        }
    }
}

void SubfeaturePickerWidgetLow::removeTemporaryObjects()
{
    for ( const auto& elem : objects_ )
    {
        for ( const SubfeatureData& sub : elem.second->subfeatures )
            sub.object->detachFromParent();
    }

    objects_.clear();
}

ObjAndPick SubfeaturePickerWidgetLow::pickObject( std::function<bool( const Object& )> pred )
{
    ObjAndPick ret{};

    if ( ImGui::GetIO().WantCaptureMouse )
        return {};

    temporarilyAdjustSubfeaturesForPicking( [&]{
        std::vector<VisualObject*> candidates;

        Viewport& viewport = getViewerInstance().viewport();

        // First pass: all pickable objects, all subfeatures, but none of the full features.

        auto lambda = [&]( auto& lambda, Object& cur ) -> void
        {
            if ( auto visual = dynamic_cast<VisualObject*>( &cur ) )
            {
                if (
                    // Matches the predicate and...
                    ( !pred || pred( cur ) ) &&
                    // Not a target feature, and...
                    !objects_.contains( { std::shared_ptr<void>{}, visual } ) &&
                    (
                        // Pickable objects.
                        visual->isPickable( viewport.id ) ||
                        // Our subfeatures.
                        findSubfeature( cur )
                    )
                )
                    candidates.push_back( visual );
            }
            for ( const auto& child : cur.children() )
                lambda( lambda, *child );
        };
        lambda( lambda, SceneRoot::get() );

        ret = viewport.pick_render_object( candidates, 0 );
        if ( ret.first && findSubfeature( *ret.first ) )
            return;

        // Second pass: all pickable objects, all full features, none of subfeatures.

        candidates.clear();

        auto lambda2 = [&]( auto& lambda2, Object& cur ) -> void
        {
            if ( auto visual = dynamic_cast<VisualObject*>( &cur ) )
            {
                if (
                    // Matches the predicate and...
                    ( !pred || pred( cur ) ) &&
                    // Not a subfeature, and...
                    !findSubfeature( cur ) &&
                    (
                        // Pickable objects.
                        visual->isPickable( viewport.id ) ||
                        // Our subfeatures.
                        !objects_.contains( { std::shared_ptr<void>{}, visual } )
                    )
                )
                    candidates.push_back( visual );
            }
            for ( const auto& child : cur.children() )
                lambda2( lambda2, *child );
        };
        lambda2( lambda2, SceneRoot::get() );

        ret = viewport.pick_render_object( candidates, 0 );
    } );

    return ret;
}

void SubfeaturePickerWidgetLow::temporarilyAdjustSubfeaturesForPicking( std::function<void()> func )
{
    ViewportId viewportId = getViewerInstance().viewport().id;

    struct FeatureRollbackEntry
    {
        std::shared_ptr<FeatureObject> subfeature;
        ViewportMask visMask;
        float pointSize = 0;
        float lineWidth = 0;
    };
    std::vector<FeatureRollbackEntry> featureRollbackEntries;

    struct MeshRollbackEntry
    {
        std::shared_ptr<ObjectMeshHolder> meshHolder;
        ViewportMask polygonOffsetMask;
    };
    std::vector<MeshRollbackEntry> meshRollbackEntries;

    // Adjust features.
    for ( auto& [object, objectData] : objects_ )
    {
        for ( SubfeatureData& subfeature : objectData->subfeatures )
        {
            featureRollbackEntries.push_back( {
                .subfeature = subfeature.object,
                .visMask = subfeature.object->visibilityMask(),
                .pointSize = subfeature.object->getPointSize(),
                .lineWidth = subfeature.object->getLineWidth(),
            } );
            subfeature.object->setVisibilityMask( dynamic_cast<VisualObject &>( *object ).getVisualizePropertyMask( FeatureVisualizePropertyType::Subfeatures ) );
            subfeature.object->setPointSize( pickerPointSize );
            subfeature.object->setLineWidth( pickerLineWidth );
        }
    }
    // Add polygon offset to meshes.
    auto lambda = [&]( auto& lambda, const std::shared_ptr<Object>& cur ) -> void
    {
        if ( auto meshHolder = std::dynamic_pointer_cast<ObjectMeshHolder>( cur ) )
        {
            auto polygonOffsetMask = meshHolder->getVisualizePropertyMask( MeshVisualizePropertyType::PolygonOffsetFromCamera );
            if ( !polygonOffsetMask.contains( viewportId ) )
            {
                meshRollbackEntries.push_back( {
                    .meshHolder = meshHolder,
                    .polygonOffsetMask = polygonOffsetMask,
                } );
                meshHolder->setVisualizePropertyMask( MeshVisualizePropertyType::PolygonOffsetFromCamera, ViewportMask::all() );
            }
        }
        for ( const auto& child : cur->children() )
            lambda( lambda, child );
    };
    lambda( lambda, SceneRoot::getSharedPtr() );

    func();

    for ( FeatureRollbackEntry& rollback : featureRollbackEntries )
    {
        rollback.subfeature->setVisibilityMask( rollback.visMask );
        rollback.subfeature->setPointSize( rollback.pointSize );
        rollback.subfeature->setLineWidth( rollback.lineWidth );
    }
    for ( MeshRollbackEntry& rollback : meshRollbackEntries )
    {
        rollback.meshHolder->setVisualizePropertyMask( MeshVisualizePropertyType::PolygonOffsetFromCamera, rollback.polygonOffsetMask );
    }
}

SubfeaturePickerWidgetLow::FindResult SubfeaturePickerWidgetLow::findSubfeature( const Object& subfeature )
{
    auto parent = subfeature.parent();
    if ( !parent )
        return {};

    auto it = objects_.find( std::shared_ptr<Object>( std::shared_ptr<void>(), const_cast<Object *>( parent ) ) );
    if ( it == objects_.end() )
        return {};

    auto subIt = std::find_if( it->second->subfeatures.begin(), it->second->subfeatures.end(), [&]( const SubfeatureData& sub ){ return sub.object.get() == &subfeature; } );
    if ( subIt == it->second->subfeatures.end() )
        return {}; // A child of the tracked object, but not one of the subfeatures.

    return { &*it, &*subIt };
}

void SubfeaturePickerWidget::enable()
{
    if ( isEnabled_ )
        return;

    isEnabled_ = true;

    connect( &getViewerInstance() );
}

void SubfeaturePickerWidget::disable()
{
    if ( !isEnabled_ )
        return;

    isEnabled_ = false;

    disconnect();
    underlyingPicker.removeTemporaryObjects();

    targets.clear();

    for ( TemporaryObject& elem : temporaryObjects_ )
    {
        if ( elem.object )
            elem.object->detachFromParent();
    }
    temporaryObjects_.clear();
}

bool SubfeaturePickerWidget::drawGui( std::function<void()> preDrawGui )
{
    if ( !isEnabled_ )
        return false;

    if ( ImGui::GetIO().WantCaptureMouse )
        mouseMovePick_ = {};

    disableFeatureHoverModifierIsHeld_ = ImGui::GetIO().KeyAlt;

    if ( targets.empty() )
        return false;

    return drawSubfeatureGuiFor( targets.back().stack, std::move( preDrawGui ) );
}

bool SubfeaturePickerWidget::highlight( const std::shared_ptr<const Object>& object, ObjectHighlighter::ModifyFunc modify )
{
    if ( !isEnabled_ )
        return false;

    // Active features.
    if ( std::any_of( targets.begin(), targets.end(), [&]( const Target& t ){ return t.stack.back().object == object; } ) )
    {
        auto& feat = dynamic_cast<FeatureObject&>( modify( "subfeat_picker:active" ) );
        highlightFeature( feat, cColorActive );
        return true;
    }

    // Hovered feature.
    if ( shouldAllowFeatures() && mouseMovePick_.first == object && dynamic_cast<const FeatureObject*>( object.get() ) )
    {
        auto& feat = dynamic_cast<FeatureObject&>( modify( "subfeat_picker:hovered" ) );
        highlightFeature( feat, cColorHover );
        return true;
    }

    // Parent of hovered feature.
    if ( shouldAllowFeatures() && mouseMovePick_.first && object.get() == mouseMovePick_.first->parent() )
    {
        if ( auto sub = underlyingPicker.findSubfeature( *mouseMovePick_.first ).feature; sub && sub->first == object )
        {
            auto& feat = dynamic_cast<FeatureObject&>( modify( "subfeat_picker:parent" ) );
            hideFeatureDimensions( feat );
            return true;
        }
    }

    return false;
}

bool SubfeaturePickerWidget::addTargetFromFeature( const std::shared_ptr<FeatureObject>& feature )
{
    Target target;
    target.stack.emplace_back();

    if ( auto sub = underlyingPicker.findSubfeature( *feature ) )
    {
        target.stack.back().name = sub.subfeature->name;
        target.stack.back().feature = sub.subfeature->subfeature;
        target.parentObject = sub.feature->first->parent()->getSharedPtr();
    }
    else
    {
        if ( auto opt = Features::primitiveFromObject( *feature ) )
        {
            target.stack.back().feature = *opt;
            target.stack.back().name = Features::name( *opt );
        }
        else
        {
            return false; // Hmm.
        }

        target.parentObject = feature->parent()->getSharedPtr();
    }

    target.stack.back().object = feature;

    targets.push_back( std::move( target ) );

    return true;
}

void SubfeaturePickerWidget::addTargetFromPoint( const std::shared_ptr<Object>& object, Vector3f localPoint )
{
    targets.push_back( {
        .stack = {
            TargetStackEntry{
                .name = "Clicked point",
                .feature = Features::toPrimitive( object->worldXf()( localPoint ) ),
            },
        },
        .parentObject = object,
    } );
}

bool SubfeaturePickerWidget::addTargetFromHover()
{
    if ( !isEnabled_ )
        return false;

    if ( !mouseMovePick_.first )
        return false; // Nothing is hovered.

    if ( shouldAllowFeatures() )
    {
        if ( auto feature = std::dynamic_pointer_cast<FeatureObject>( mouseMovePick_.first ) )
            return addTargetFromFeature( feature );
    }

    if ( shouldAllowNonFeatures() )
    {
        switch ( nonFeatureMode )
        {
        case NonFeatureMode::point:
            addTargetFromPoint( mouseMovePick_.first, mouseMovePick_.second.point );
            return true;
        }
    }

    return false;
}

bool SubfeaturePickerWidget::drawSubfeatureGuiFor( std::vector<TargetStackEntry>& stack, std::function<void()> preDrawGui )
{
    bool ret = false;

    if ( stack.empty() )
        return ret;

    bool first = true;
    auto beginSubfeatureGui = [&]
    {
        if ( first )
        {
            first = false;
            if ( preDrawGui )
                preDrawGui();
            UI::transparentText( fmt::format( "Subfeatures of {}:", stack.back().name ).c_str() );
        }
    };

    if ( stack.size() > 1 )
    {
        beginSubfeatureGui();
        ImGui::Bullet();
        if ( ImGui::Selectable( fmt::format( "[ Back to {} ]", stack[stack.size() - 2].name ).c_str() ) )
        {
            stack.pop_back();
            ret = true;
        }
    }

    Features::forEachSubfeature( stack.back().feature, [&]( const Features::SubfeatureInfo& info )
    {
        beginSubfeatureGui();
        ImGui::Bullet();
        if ( ImGui::Selectable( info.name.data() ) )
        {
            stack.push_back( {
                .name = std::string( info.name ),
                .feature = info.create(),
            } );
            ret = true;
        }
    } );

    return ret;
}

void SubfeaturePickerWidget::preDraw_()
{
    if ( !isEnabled_ )
        return;

    underlyingPicker.updateTemporaryObjects( [&]( const Object& object, bool isAncillary )
    {
        (void)isAncillary;
        return std::none_of( temporaryObjects_.begin(), temporaryObjects_.end(), [&]( const TemporaryObject& t ){ return t.object.get() == &object; } );
    } );

    { // Update our own temporary objects.
        if ( temporaryObjects_.size() < targets.size() )
            temporaryObjects_.resize( targets.size() );

        for ( std::size_t i = 0; i < targets.size(); i++ )
        {
            Target& thisTarget = targets[i];
            TemporaryObject& thisTempObject = temporaryObjects_[i];

            std::optional<Features::Primitives::Variant> desiredFeature;
            if ( thisTarget.stack.back().object == nullptr )
                desiredFeature = thisTarget.stack.back().feature;

            if ( desiredFeature != thisTempObject.feature )
            {
                if ( thisTempObject.object )
                {
                    thisTempObject.object->detachFromParent();
                    thisTempObject.object = nullptr;
                    thisTempObject.feature = {};
                }

                if ( desiredFeature )
                {
                    thisTempObject.feature = desiredFeature;
                    thisTempObject.object = Features::primitiveToObject( *desiredFeature, underlyingPicker.getInfiniteExtent() );
                    highlightFeature( *thisTempObject.object, cColorActive );

                    Object* parent = nullptr;
                    for ( auto it = thisTarget.stack.end(); it != thisTarget.stack.begin(); )
                    {
                        --it;
                        if ( it->object )
                        {
                            parent = it->object.get();
                            if ( auto sub = underlyingPicker.findSubfeature( *parent ) )
                                parent = parent->parent();
                            // And again:
                            parent = parent->parent();
                            break;
                        }
                    }
                    if ( parent )
                        thisTempObject.object->setXf( parent->worldXf().inverse() * thisTempObject.object->xf() );
                    else
                        parent = &SceneRoot::get();

                    thisTempObject.object->setPickable( false );
                    thisTempObject.object->setAncillary( true );

                    parent->addChild( thisTempObject.object );
                }
            }
        }

        while ( temporaryObjects_.size() > targets.size() )
        {
            if ( temporaryObjects_.back().object )
                temporaryObjects_.back().object->detachFromParent();
            temporaryObjects_.pop_back();
        }
    }
}

bool SubfeaturePickerWidget::onMouseMove_( int x, int y )
{
    (void)x;
    (void)y;

    if ( !isEnabled_ )
        return false;

    mouseMovePick_ = underlyingPicker.pickObject( [&]( const Object& obj )
    {
        return std::none_of( temporaryObjects_.begin(), temporaryObjects_.end(), [&]( const TemporaryObject& t ){ return t.object.get() == &obj; } );
    } );
    return false;
}

}
