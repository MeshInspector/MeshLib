#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRStatePluginUpdate.h"
#include "MRViewer/MRRibbonRegisterItem.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRUnits.h"
#include "MRViewer/MRI18n.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRFeatureObject.h"
#include "MRMesh/MRFeatures.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"

namespace MR
{

class FeatureAlignment : public StatePlugin, public PluginCloseOnSelectedObjectRemove
{
public:
    FeatureAlignment() : StatePlugin( "Feature Alignment", StatePluginTabs::Basic ) {}

    std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const override
    {
        if ( objs.size() == 2 && !objs[0]->isAncestor( objs[1].get() ) && !objs[1]->isAncestor( objs[0].get() ) )
            return {};
        return "Select two separate objects: the upper one stays, the lower one moves";
    }

    void drawDialog( ImGuiContext* ) override;

private:
    bool onEnable_() override
    {
        auto objs = getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        if ( objs.size() != 2 )
            return false;
        parts_ = { objs[0], objs[1] };
        return true;
    }

    bool onDisable_() override
    {
        parts_ = {};
        rows_ = {};
        return true;
    }

    // feature of the given row and side (0 - fixed, 1 - moving) in world space
    std::optional<Features::Primitives::Variant> primitive_( int row, int side ) const
    {
        const auto& f = rows_[row][side];
        auto res = f ? Features::primitiveFromObject( *f ) : std::nullopt;
        if ( res && f->parent() ) // feature getters are in the parent space
            res = Features::transformPrimitive( f->parent()->worldXf(), *res );
        return res;
    }

    std::optional<Features::AlignPair> pair_( int row ) const
    {
        auto f = primitive_( row, 0 ), m = primitive_( row, 1 );
        if ( !f || !m )
            return {};
        return Features::AlignPair{ *m, *f };
    }

    std::array<std::shared_ptr<Object>, 2> parts_; // fixed, moving
    std::array<std::array<std::shared_ptr<FeatureObject>, 2>, 4> rows_; // primary, locator 1, locator 2, clocking
    bool flip_ = false;
    float gap_ = 0;
};

void FeatureAlignment::drawDialog( ImGuiContext* )
{
    if ( !ImGuiBeginWindow_( { .width = 400 * UI::scale() } ) )
        return;

    ImGui::Text( "%s: %s", _tr( "Fixed" ), parts_[0]->name().c_str() );
    ImGui::Text( "%s: %s", _tr( "Moving" ), parts_[1]->name().c_str() );
    if ( UI::button( _tr( "Swap" ) ) )
    {
        std::swap( parts_[0], parts_[1] );
        for ( auto& row : rows_ )
            std::swap( row[0], row[1] );
    }

    const char* rowNames[] = { "Primary", "Locator 1", "Locator 2", "Clocking" };
    std::array<std::vector<std::shared_ptr<FeatureObject>>, 2> features;
    std::array<std::vector<std::string>, 2> names;
    for ( int side = 0; side < 2; ++side )
    {
        features[side] = getAllObjectsInTree<FeatureObject>( parts_[side].get() );
        names[side] = { "-" };
        for ( const auto& f : features[side] )
            names[side].push_back( f->name() );
    }
    ImGui::PushItemWidth( ImGui::GetContentRegionAvail().x * 0.35f );
    for ( int row = 0; row < 4; ++row )
    {
        for ( int side = 0; side < 2; ++side )
        {
            const auto& fs = features[side];
            auto it = std::find( fs.begin(), fs.end(), rows_[row][side] );
            int i = it == fs.end() ? 0 : int( it - fs.begin() ) + 1;
            if ( side )
                ImGui::SameLine();
            if ( UI::combo( ( ( side ? "" : "##" ) + s_tr( rowNames[row] ) ).c_str(), &i, names[side] ) || !i )
                rows_[row][side] = i ? fs[i - 1] : nullptr;
        }
    }
    ImGui::PopItemWidth();
    UI::checkbox( _tr( "Flip" ), &flip_ );
    UI::setTooltipIfHovered( _tr( "Planes: same normals instead of opposite ones. Axes: the farther direction instead of the closer one." ) );
    UI::drag<LengthUnit>( _tr( "Gap" ), gap_ );

    const auto primary = pair_( 0 );
    if ( !primary )
        UI::transparentTextWrapped( "%s", _tr( "Choose the primary features of both objects." ) );
    else
    {
        Features::AlignParams params{ .primary = *primary, .clocking = pair_( 3 ), .flip = flip_, .gap = gap_ };
        for ( int row : { 1, 2 } )
            if ( auto l = pair_( row ) )
                params.locators.push_back( *l );
        const auto xf = Features::align( params );
        const auto center = [] ( const Features::Primitives::Variant& v )
        {
            return std::visit( overloaded{ [] ( const Features::Primitives::ConeSegment& c ) { return c.centerPoint().center; },
                                           [] ( const auto& p ) { return p.center; } }, v );
        };
        const auto* plane = std::get_if<Features::Primitives::Plane>( &primary->fixed );
        for ( int row : { 1, 2 } )
        {
            if ( auto l = pair_( row ) )
            {
                auto d = AffineXf3f( xf )( center( l->moving ) ) - center( l->fixed );
                if ( plane ) // the plane gap sets the depth, locators only the position along the plane
                    d -= dot( d, plane->normal ) * plane->normal;
                ImGui::Text( "%s %s: %s", s_tr( rowNames[row] ).c_str(), _tr( "mismatch" ), valueToString<LengthUnit>( d.length() ).c_str() );
            }
        }
        if ( UI::button( _tr( "Apply" ), !parts_[1]->isLocked(), Vector2f( -1, 0 ) ) )
        {
            AppendHistory<ChangeXfAction>( s_tr( "Feature Alignment" ), parts_[1] );
            parts_[1]->setWorldXf( AffineXf3f( xf * AffineXf3d( parts_[1]->worldXf() ) ) );
        }
    }
    ImGui::EndCustomStatePlugin();
}

MR_REGISTER_RIBBON_ITEM( FeatureAlignment )

}
