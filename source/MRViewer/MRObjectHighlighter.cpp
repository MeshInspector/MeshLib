#include "MRObjectHighlighter.h"

#include "MRMesh/MRFeatureObject.h"
#include "MRMesh/MRObjectDimensionsEnum.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRVisualObject.h"

#include <unordered_set>

namespace MR
{

ObjectHighlighter::RollbackFunc ObjectHighlighter::ruleObjectVisibility( const Object& object )
{
    return [
        mask = object.visibilityMask()
    ]( Object& target )
    {
        target.setVisibilityMask( mask );
    };
}

ObjectHighlighter::RollbackFunc ObjectHighlighter::ruleVisualObjectColors( const Object& object )
{
    auto visual = dynamic_cast<const VisualObject*>( &object );
    if ( !visual )
        return nullptr;

    return [
        colorUnsel = visual->getFrontColorsForAllViewports( false ),
        colorSel = visual->getFrontColorsForAllViewports( true ),
        colorBack = visual->getBackColorsForAllViewports()
    ]( Object& target )
    {
        auto& visualTarget = dynamic_cast<VisualObject&>( target );
        visualTarget.setFrontColorsForAllViewports( colorUnsel, false );
        visualTarget.setFrontColorsForAllViewports( colorSel, true );
        visualTarget.setBackColorsForAllViewports( colorBack );
    };
}

ObjectHighlighter::RollbackFunc ObjectHighlighter::ruleFeatures( const Object& object )
{
    auto feature = dynamic_cast<const FeatureObject*>( &object );
    if ( !feature )
        return nullptr;

    std::array<ViewportMask, std::size_t( DimensionsVisualizePropertyType::_count )> dimensions{};
    for ( std::size_t i = 0; i < std::size_t( DimensionsVisualizePropertyType::_count ); i++ )
    {
        auto enumValue = DimensionsVisualizePropertyType( i );
        if ( feature->supportsVisualizeProperty( enumValue ) )
            dimensions[i] = feature->getVisualizePropertyMask( enumValue );
    }

    return [
        dimensions,
        decoColorUnsel = feature->getDecorationsColorForAllViewports( false ),
        decoColorSel = feature->getDecorationsColorForAllViewports( true ),
        subfeatureMask = feature->getVisualizePropertyMask( FeatureVisualizePropertyType::Subfeatures ),
        pointSize = feature->getPointSize(),
        lineWidth = feature->getLineWidth(),
        subPointSize = feature->getSubfeaturePointSize(),
        subLineWidth = feature->getSubfeatureLineWidth(),
        alpha = feature->getMainFeatureAlpha(),
        subAlphaPoints = feature->getSubfeatureAlphaPoints(),
        subAlphaLines = feature->getSubfeatureAlphaLines(),
        subAlphaMesh = feature->getSubfeatureAlphaMesh()
    ]( Object& target )
    {
        auto& featureTarget = dynamic_cast<FeatureObject&>( target );

        for ( std::size_t i = 0; i < std::size_t( DimensionsVisualizePropertyType::_count ); i++ )
        {
            auto enumValue = DimensionsVisualizePropertyType( i );
            if ( featureTarget.supportsVisualizeProperty( enumValue ) )
                featureTarget.setVisualizePropertyMask( enumValue, dimensions[i] );
        }

        featureTarget.setDecorationsColorForAllViewports( decoColorUnsel, false );
        featureTarget.setDecorationsColorForAllViewports( decoColorSel, true );
        featureTarget.setVisualizePropertyMask( FeatureVisualizePropertyType::Subfeatures, subfeatureMask );
        featureTarget.setPointSize( pointSize );
        featureTarget.setLineWidth( lineWidth );
        featureTarget.setSubfeaturePointSize( subPointSize );
        featureTarget.setSubfeatureLineWidth( subLineWidth );
        featureTarget.setMainFeatureAlpha( alpha );
        featureTarget.setSubfeatureAlphaPoints( subAlphaPoints );
        featureTarget.setSubfeatureAlphaLines( subAlphaLines );
        featureTarget.setSubfeatureAlphaMesh( subAlphaMesh );
    };
}

ObjectHighlighter::ObjectHighlighter( std::vector<RuleFunc> rules )
    : rules_( std::move( rules ) )
{}

void ObjectHighlighter::highlight( std::function<void( const std::shared_ptr<const Object>& object, ModifyFunc modify )> highlightObject )
{
    std::unordered_set<std::shared_ptr<Object>> visitedObjects;

    auto lambda = [&]( auto& lambda, const std::shared_ptr<Object>& cur ) -> void
    {
        bool once = true;

        highlightObject( cur, [&]( std::string name ) -> Object&
        {
            if ( once )
            {
                once = false;

                visitedObjects.insert( cur );

                auto [iter, isNew] = rollbackEntries_.try_emplace( cur );

                if ( !isNew && iter->second.name != name )
                {
                    iter->second.func( *iter->first );
                    iter->second = {};
                    isNew = true;
                }

                if ( isNew )
                {
                    std::vector<RollbackFunc> objRollbackFuncs;
                    for ( const auto& rule : rules_ )
                    {
                        if ( auto func = rule( *cur ) )
                            objRollbackFuncs.push_back( std::move( func ) );
                    }
                    assert( !objRollbackFuncs.empty() && "ObjectHighlighter: The object passed the filter, but I don't know which information about it to save for rollback." );

                    iter->second.func = [funcs = std::move(objRollbackFuncs)]( Object& target )
                    {
                        for ( auto& func : funcs )
                            func( target );
                    };
                    iter->second.name = std::move( name );
                }
            }

            return *cur;
        } );

        for ( const auto& child : cur->children() )
            lambda( lambda, child );
    };
    lambda( lambda, SceneRoot::getSharedPtr() );

    for ( auto it = rollbackEntries_.begin(); it != rollbackEntries_.end(); )
    {
        if ( !visitedObjects.contains( it->first ) )
        {
            it->second.func( *it->first );

            it = rollbackEntries_.erase( it );
        }
        else
        {
            it++;
        }
    }
}

void ObjectHighlighter::restoreOriginalState()
{
    for ( auto& elem : rollbackEntries_ )
        elem.second.func( *elem.first );
    rollbackEntries_.clear();
}

}
