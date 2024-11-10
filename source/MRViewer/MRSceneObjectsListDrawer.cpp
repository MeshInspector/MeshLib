#include "MRSceneObjectsListDrawer.h"
#include "MRSceneCache.h"
#include "MRViewer.h"
#include "MRViewerInstance.h"
#include "MRViewport.h"
#include "MRUIStyle.h"
#include "MRShowModal.h"
#include "MRRibbonConstants.h"
#include "MRColorTheme.h"
#include "ImGuiMenu.h"
#include "MRAppendHistory.h"
#include "MRRibbonSchema.h"
#include "MRUITestEngine.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRPch/MRSpdlog.h"
#include "imgui_internal.h"
#include "imgui.h"
#include <stack>
#include <iterator>

namespace MR
{

// helper class to optimaze render (skip elements outside draw area)
class SkippableRenderer
{
public:
    SkippableRenderer()
    {
        drawBoxMinY = ImGui::GetScrollY();
        drawBoxMaxY = drawBoxMinY + ImGui::GetWindowHeight();

        cursorPosY = ImGui::GetCursorPosY();
        skipedCursorPosY = cursorPosY;
    }

    void draw( float height, float spacingY, std::function<void( void )> drawFunc, std::function<void( void )> hiddenFunc = []{} )
    {
        if ( skipedCursorPosY + height <= drawBoxMinY || skipedCursorPosY > drawBoxMaxY )
        {
            skipedCursorPosY += height + spacingY;
            lastSpacingY = spacingY;
            hiddenFunc();
        }
        else
        {
            if ( skipedCursorPosY > cursorPosY )
                ImGui::Dummy( ImVec2( 0, skipedCursorPosY - cursorPosY - lastSpacingY ) );
            
            drawFunc();

            cursorPosY = ImGui::GetCursorPosY();
            skipedCursorPosY = cursorPosY;
        }
    }

    void endDraw()
    {
        if ( skipedCursorPosY > cursorPosY )
            ImGui::Dummy( ImVec2( 0, skipedCursorPosY - cursorPosY - lastSpacingY ) );
        cursorPosY = ImGui::GetCursorPosY();
        skipedCursorPosY = cursorPosY;
    }

    float getCursorPosY() { return skipedCursorPosY; }

private:
    float drawBoxMinY = 0.f;
    float drawBoxMaxY = 0.f;

    float cursorPosY = 0.f;
    float skipedCursorPosY = 0.f;

    float lastSpacingY = 0.f;
};

////////////////////////////////////////////////////

void SceneObjectsListDrawer::draw( float height, float scaling )
{
    menuScaling_ = scaling;

    ImGui::BeginChild( "SceneObjectsList", ImVec2( -1, height ), false );
    updateSceneWindowScrollIfNeeded_();
    drawObjectsList_();
    // any click on empty space below Scene Tree removes object selection
    const auto& selected = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>();
    ImGui::BeginChild( "EmptySpace" );
    if ( ImGui::IsWindowHovered() && ImGui::IsMouseClicked( 0 ) )
    {
        for ( const auto& s : selected )
            if ( s )
                s->select( false );
    }
    ImGui::EndChild();

    ImGui::EndChild();
    sceneOpenCommands_.clear();
    reorderSceneIfNeeded_();
}

float SceneObjectsListDrawer::drawCustomTreeObjectProperties_( Object&, bool )
{
    return 0.f;
}

bool SceneObjectsListDrawer::collapsingHeader_( const std::string& uniqueName, ImGuiTreeNodeFlags flags )
{
    return ImGui::CollapsingHeader( uniqueName.c_str(), flags );
}

std::string SceneObjectsListDrawer::objectLineStrId_( const Object& object, const std::string& uniqueStr )
{
    return object.name() + "##" + uniqueStr;
}

void SceneObjectsListDrawer::changeSelection( bool isDown, bool isShift )
{
    const auto& all = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selectable>();
    const auto& selected = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>();
    if ( isDown )
    {
        if ( downLastSelected_.index != -1 )
        {
            if ( !isShift )
                for ( const auto& data : selected )
                    data->select( false );
            all[downLastSelected_.index]->select( true );
            downLastSelected_.needScroll = true;
            if ( showNewSelectedObjects_ )
                all[downLastSelected_.index]->setGlobalVisibility( true );
        }
    }
    else
    {
        if ( upFirstSelected_.index != -1 )
        {
            if ( !isShift )
                for ( const auto& data : selected )
                    data->select( false );
            all[upFirstSelected_.index]->select( true );
            upFirstSelected_.needScroll = true;
            if ( showNewSelectedObjects_ )
                all[upFirstSelected_.index]->setGlobalVisibility( true );
        }
    }
}

void SceneObjectsListDrawer::changeVisible( bool isDown )
{
    const auto& all = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selectable>();
    if ( all.empty() )
        return;

    const auto& selected = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>();
    std::shared_ptr<Object> nextObj;
    if ( selected.empty() )
    {
        if ( isDown )
        {
            nextObj = all[0];
        }
        else
        {
            const auto& rootChildren = SceneRoot::get().children();
            const auto it = std::find_if( rootChildren.rbegin(), rootChildren.rend(), [] ( auto& obj )
            {
                return !obj->isAncillary();
            } );
            nextObj = *it;
        }
    }
    else
    {
        const auto& children = selected[0]->parent()->children();
        auto itFirst = std::find( children.begin(), children.end(), selected[0] );
        const int itFirstIndex = int( std::distance( children.begin(), itFirst ) );
        int nextIndex = itFirstIndex;
        if ( isDown )
        {
            for ( int i = 1; i < children.size(); ++i )
            {
                nextIndex = int( ( itFirstIndex + i ) % children.size() );
                if ( !children[nextIndex]->isAncillary() )
                    break;
            }
        }
        else
        {
            for ( int i = 1; i < children.size(); ++i )
            {
                nextIndex = int( ( itFirstIndex - i + children.size() ) % children.size() );
                if ( !children[nextIndex]->isAncillary() )
                    break;
            }
        }
        nextObj = children[nextIndex];
    }
    const auto itAll = std::find( all.begin(), all.end(), nextObj );
    nextVisible_.index = int( std::distance( all.begin(), itAll ) );

    for ( const auto& obj : nextObj->parent()->children() )
        obj->setVisible( false );
    for ( const auto& obj : selected )
        obj->select( false );
    nextObj->setVisible( true );
    nextObj->select( true );
}

void SceneObjectsListDrawer::selectAllObjects()
{
    const auto& selectable = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selectable>();
    for ( auto obj : selectable )
    {
        obj->select( true );
        if ( showNewSelectedObjects_ )
            obj->setVisible( true );
    }
}

void SceneObjectsListDrawer::setLeavesVisibility( bool visible )
{
    const auto& selectable = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selectable>();
    for ( auto obj : selectable )
    {
        obj->setVisible( visible );
    }
}

void SceneObjectsListDrawer::setObjectTreeState( const Object* obj, bool open )
{
    if ( obj )
        sceneOpenCommands_[obj] = open;
}

void SceneObjectsListDrawer::allowSceneReorder( bool allow )
{
    allowSceneReorder_ = allow;
}

void SceneObjectsListDrawer::drawObjectsList_()
{
    const auto& all = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selectable>();
    std::vector<int> depths( all.size(), -1 );
    auto getDepth = [&] ( int i )
    {
        if ( depths[i] == -1 )
        {
            const Object* obj = all[i].get();
            int depth = 0;
            while ( obj->parent() && obj->parent() != SceneRoot::getSharedPtr().get() )
            {
                obj = obj->parent();
                if ( obj->isAncillary() )
                    --depth;
                ++depth;
            }
            depths[i] = depth;
        }
        return depths[i];
    };

    int curentDepth = 0;
    std::stack<std::shared_ptr<Object>> objDepthStack;

    SkippableRenderer skippableRenderer;

    int collapsedHeaderDepth = -1;
    const float itemSpacingY = ImGui::GetStyle().ItemSpacing.y;
    const float frameHeight = ImGui::GetFrameHeight();

    bool firstSelectedIsFound = false;
    bool previousWasSelected = false;

    const float borderShift = ImGui::GetFrameHeight();
    const float scrollPosY = ImGui::GetScrollY();
    const float windowHeight = ImGui::GetWindowHeight();
    if ( upFirstSelected_.needScroll && upFirstSelected_.posY < ( scrollPosY + ImGui::GetStyle().WindowPadding.y + borderShift ) )
        ImGui::SetScrollY( upFirstSelected_.posY - ImGui::GetStyle().WindowPadding.y - borderShift );
    if ( downLastSelected_.needScroll && downLastSelected_.posY > ( scrollPosY + windowHeight - borderShift ) )
        ImGui::SetScrollY( downLastSelected_.posY - windowHeight + ImGui::GetStyle().WindowPadding.y + borderShift );
    if ( nextVisible_.needScroll )
    {
        if ( nextVisible_.posY < ( scrollPosY + ImGui::GetStyle().WindowPadding.y + borderShift ) )
            ImGui::SetScrollY( nextVisible_.posY - ImGui::GetStyle().WindowPadding.y - borderShift );
        else if ( ( nextVisible_.posY + frameHeight ) > ( scrollPosY + windowHeight - borderShift ) )
            ImGui::SetScrollY( nextVisible_.posY + frameHeight - windowHeight + ImGui::GetStyle().WindowPadding.y + borderShift );
        nextVisible_ = MoveAndScrollData();
    }

    upFirstSelected_ = MoveAndScrollData();
    downLastSelected_ = MoveAndScrollData();

    for ( int i = 0; i < all.size(); ++i )
    {
        const bool isLast = i == int( all.size() ) - 1;
        const int nextDepth = isLast ? 0 : getDepth( i + 1 );
        // skip child elements after collapsed header
        if ( collapsedHeaderDepth >= 0 )
        {
            if ( getDepth( i ) > collapsedHeaderDepth)
            {
                if ( curentDepth > nextDepth )
                {
                    for ( ; curentDepth > nextDepth; --curentDepth )
                    {
                        if ( needDragDropTarget_() )
                            skippableRenderer.draw( getDrawDropTargetHeight_(), itemSpacingY, [&] { makeDragDropTarget_( *objDepthStack.top(), false, true, "0" ); } );
                        objDepthStack.pop();
                        ImGui::Unindent();
                    }
                }
                continue;
            }
            else
                collapsedHeaderDepth = -1;
        }

        auto& object = *all[i];
        if ( curentDepth < getDepth( i ) )
        {
            ImGui::Indent();
            if ( i > 0 )
                objDepthStack.push( all[i - 1] );
            ++curentDepth;
            assert( curentDepth == getDepth( i ) );
        }

        {
            std::string uniqueStr = std::to_string( intptr_t( &object ) );
            bool isOpen{ false };

            if ( needDragDropTarget_() )
                skippableRenderer.draw( getDrawDropTargetHeight_(), itemSpacingY, [&] { makeDragDropTarget_(object, true, true, uniqueStr); });

            if ( object.isSelected() )
                firstSelectedIsFound = true;
            else if ( !firstSelectedIsFound )
            {
                upFirstSelected_.index = i;
                upFirstSelected_.posY = skippableRenderer.getCursorPosY();
            }

            if ( nextVisible_.index == i )
            {
                nextVisible_.posY = skippableRenderer.getCursorPosY();
                nextVisible_.needScroll = true;
            }

            skippableRenderer.draw( frameHeight, itemSpacingY,
            [&] { isOpen = drawObject_( object, uniqueStr, curentDepth ); },
            [&] { isOpen = drawSkippedObject_( object, uniqueStr, curentDepth ); } );

            if ( object.isSelected() )
                previousWasSelected = true;
            else
            {
                if ( previousWasSelected )
                {
                    downLastSelected_.index = i;
                    downLastSelected_.posY = skippableRenderer.getCursorPosY();
                }
                previousWasSelected = false;
            }

            if ( object.isSelected() )
                drawSceneContextMenu_( SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>(), uniqueStr );

            if ( isOpen )
            {
                const float drawPropertiesHeight = drawCustomTreeObjectProperties_( object, true );
                if ( drawPropertiesHeight > 0.f )
                    skippableRenderer.draw( drawPropertiesHeight, itemSpacingY, [&] { drawCustomTreeObjectProperties_( object, false ); } );
            }
            else
                collapsedHeaderDepth = curentDepth;
        }

        for ( ; curentDepth > nextDepth; --curentDepth )
        {
            if ( needDragDropTarget_() )
                skippableRenderer.draw( getDrawDropTargetHeight_(), itemSpacingY, [&] { makeDragDropTarget_(*objDepthStack.top(), false, true, "0"); });
            objDepthStack.pop();
            ImGui::Unindent();
        }
    }
    if ( needDragDropTarget_() )
        skippableRenderer.draw( getDrawDropTargetHeight_(), itemSpacingY, [&] { makeDragDropTarget_( SceneRoot::get(), false, true, "" ); } );
    skippableRenderer.endDraw();
}

bool SceneObjectsListDrawer::drawObject_( Object& object, const std::string& uniqueStr, int /*depth*/ )
{
    const bool hasRealChildren = objectHasSelectableChildren( object );

    drawObjectVisibilityCheckbox_( object, uniqueStr );
    drawCustomObjectPrefixInScene_( object, false );
    return drawObjectCollapsingHeader_( object, uniqueStr, hasRealChildren );
}

bool SceneObjectsListDrawer::drawSkippedObject_( Object& object, const std::string& uniqueStr, int )
{
    bool hasRealChildren = objectHasSelectableChildren( object );
    return ImGui::TreeNodeUpdateNextOpen( ImGui::GetCurrentWindow()->GetID( objectLineStrId_( object, uniqueStr ).c_str() ),
                    ( hasRealChildren ? ImGuiTreeNodeFlags_DefaultOpen : 0 ) );
}

void SceneObjectsListDrawer::drawObjectVisibilityCheckbox_( Object& object, const std::string& uniqueStr )
{
    const auto& viewerRef = getViewerInstance();
    // Visibility checkbox
    bool isVisible = object.isVisible( viewerRef.viewport().id );
    auto ctx = ImGui::GetCurrentContext();
    assert( ctx );
    auto window = ctx->CurrentWindow;
    assert( window );
    auto diff = ImGui::GetStyle().FramePadding.y - cCheckboxPadding * menuScaling_;
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + diff );
    if ( UI::checkbox( ( "##VisibilityCheckbox" + uniqueStr ).c_str(), &isVisible ) )
    {
        object.setVisible( isVisible, viewerRef.viewport().id );
        if ( deselectNewHiddenObjects_ && !object.isVisible( viewerRef.getPresentViewports() ) )
            object.select( false );
    }
    window->DC.CursorPosPrevLine.y -= diff;
    ImGui::SameLine();
}

bool SceneObjectsListDrawer::drawObjectCollapsingHeader_( Object& object, const std::string& uniqueStr, bool hasRealChildren )
{
    const auto& selected = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>();
    const bool isSelected = object.isSelected();

    auto openCommandIt = sceneOpenCommands_.find( &object );
    if ( openCommandIt != sceneOpenCommands_.end() )
        ImGui::SetNextItemOpen( openCommandIt->second );

    if ( !isSelected )
        ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0, 0, 0, 0 ) );
    else
    {
        ImGui::PushStyleColor( ImGuiCol_Header, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::SelectedObjectFrame ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::SelectedObjectText ).getUInt32() );
    }

    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );

    const ImGuiTreeNodeFlags flags = 
        ImGuiTreeNodeFlags_SpanAvailWidth | 
        ImGuiTreeNodeFlags_Framed | 
        ( hasRealChildren ? ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_Bullet ) |
        ( isSelected ? ImGuiTreeNodeFlags_Selected : 0 );

    const bool isOpen = collapsingHeader_( objectLineStrId_( object, uniqueStr ).c_str(), flags );

    ImGui::PopStyleColor( isSelected ? 2 : 1 );
    ImGui::PopStyleVar();

    makeDragDropSource_( selected );
    makeDragDropTarget_( object, false, false, uniqueStr );

    if ( ImGui::IsItemHovered() )
        processItemClick_( object, selected );    

    return isOpen;
}

void SceneObjectsListDrawer::processItemClick_( Object& object, const std::vector<std::shared_ptr<Object>>& selected )
{
    const auto& all = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selectable>();
    auto isSelected = object.isSelected();

    if ( ImGui::IsMouseDoubleClicked( 0 ) )
    {
        if ( auto menu = getViewerInstance().getMenuPlugin() )
            menu->tryRenameSelectedObject();
    }

    bool pressed = !isSelected && ( ImGui::IsMouseClicked( 0 ) || ImGui::IsMouseClicked( 1 ) );
    bool released = isSelected && !dragTrigger_ && !clickTrigger_ && ImGui::IsMouseReleased( 0 );

    if ( pressed )
        clickTrigger_ = true;
    if ( isSelected && clickTrigger_ && ImGui::IsMouseReleased( 0 ) )
        clickTrigger_ = false;

    if ( pressed || released )
        updateSelection_( &object, selected, all );
}

void SceneObjectsListDrawer::makeDragDropSource_( const std::vector<std::shared_ptr<Object>>& payload )
{
    if ( !allowSceneReorder_ || payload.empty() )
        return;

    if ( std::any_of( payload.begin(), payload.end(), std::mem_fn( &Object::isParentLocked ) ) )
        return; // Those objects don't want their parents to be changed.

    if ( ImGui::BeginDragDropSource( ImGuiDragDropFlags_AcceptNoDrawDefaultRect | ImGuiDragDropFlags_SourceNoDisableHover ) )
    {
        dragTrigger_ = true;

        std::vector<Object*> vectorObjPtr;
        for ( auto& ptr : payload )
            vectorObjPtr.push_back( ptr.get() );

        ImGui::SetDragDropPayload( "_TREENODE", vectorObjPtr.data(), sizeof( Object* ) * vectorObjPtr.size() );
        std::string allNames;
        allNames = payload[0]->name();
        for ( int i = 1; i < payload.size(); ++i )
            allNames += "\n" + payload[i]->name();
        ImGui::Text( "%s", allNames.c_str() );
        ImGui::EndDragDropSource();
    }
}

bool SceneObjectsListDrawer::needDragDropTarget_()
{
    if ( !allowSceneReorder_ )
        return false;
    const ImGuiPayload* payloadCheck = ImGui::GetDragDropPayload();
    return payloadCheck && std::string_view( payloadCheck->DataType ) == "_TREENODE";
}

void SceneObjectsListDrawer::makeDragDropTarget_( Object& target, bool before, bool betweenLine, const std::string& uniqueStr )
{
    if ( !allowSceneReorder_ )
        return;
    const ImGuiPayload* payloadCheck = ImGui::GetDragDropPayload();
    ImVec2 curPos{};
    const bool lineDrawed = payloadCheck && std::string_view( payloadCheck->DataType ) == "_TREENODE" && betweenLine;
    if ( lineDrawed )
    {
        curPos = ImGui::GetCursorPos();
        auto width = ImGui::GetContentRegionAvail().x;
        ImGui::ColorButton( ( "##InternalDragDropArea" + uniqueStr ).c_str(),
            ImVec4( 0, 0, 0, 0 ),
            0, ImVec2( width, 4 * menuScaling_ ) );
    }
    if ( ImGui::BeginDragDropTarget() )
    {
        if ( lineDrawed )
        {
            ImGui::SetCursorPos( curPos );
            auto width = ImGui::GetContentRegionAvail().x;
            ImGui::ColorButton( ( "##ColoredInternalDragDropArea" + uniqueStr ).c_str(),
                ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered],
                0, ImVec2( width, 4 * menuScaling_ ) );
        }
        if ( const ImGuiPayload* payload = ImGui::AcceptDragDropPayload( "_TREENODE" ) )
        {
            assert( payload->DataSize % sizeof( Object* ) == 0 );
            Object** objArray = ( Object** )payload->Data;
            const int size = payload->DataSize / sizeof( Object* );
            std::vector<Object*> vectorObj( size );
            for ( int i = 0; i < size; ++i )
                vectorObj[i] = objArray[i];
            sceneReorderCommand_ = { vectorObj, &target, before };
        }
        ImGui::EndDragDropTarget();
    }
}

void SceneObjectsListDrawer::reorderSceneIfNeeded_()
{
    if ( !allowSceneReorder_ )
        return;

    const bool filledReorderCommand = !sceneReorderCommand_.who.empty() && sceneReorderCommand_.to;
    const bool sourceNotTarget = std::all_of( sceneReorderCommand_.who.begin(), sceneReorderCommand_.who.end(), [target = sceneReorderCommand_.to] ( auto it )
    {
        return it != target;
    } );
    const bool trueTarget = !sceneReorderCommand_.before || sceneReorderCommand_.to->parent();
    const bool trueSource = std::all_of( sceneReorderCommand_.who.begin(), sceneReorderCommand_.who.end(), [] ( auto it )
    {
        return bool( it->parent() );
    } );
    if ( !( filledReorderCommand && sourceNotTarget && trueSource && trueTarget ) )
    {
        sceneReorderCommand_ = {};
        return;
    }

    bool dragOrDropFailed = false;
    std::shared_ptr<Object> childTo = nullptr;
    if ( sceneReorderCommand_.before )
    {
        for ( auto childToItem : sceneReorderCommand_.to->parent()->children() )
            if ( childToItem.get() == sceneReorderCommand_.to )
            {
                childTo = childToItem;
                break;
            }
        assert( childTo );
    }

    struct MoveAction
    {
        std::shared_ptr<ChangeSceneAction> detachAction;
        std::shared_ptr<ChangeSceneAction> attachAction;
        std::shared_ptr<ChangeXfAction> xfAction;
    };
    std::vector<MoveAction> actionList;
    for ( const auto& source : sceneReorderCommand_.who )
    {
        Object * fromParent = source->parent();
        assert( fromParent );
        std::shared_ptr<Object> sourcePtr = source->getSharedPtr();
        assert( sourcePtr );
        const auto worldXf = source->worldXf();

        auto detachAction = std::make_shared<ChangeSceneAction>( "Detach object", sourcePtr, ChangeSceneAction::Type::RemoveObject );
        bool detachSuccess = sourcePtr->detachFromParent();
        if ( !detachSuccess )
        {
            //showModalMessage( "Cannot perform such reorder", NotificationType::Error );
            showModal( "Cannot perform such reorder", NotificationType::Error );
            dragOrDropFailed = true;
            break;
        }

        auto attachAction = std::make_shared<ChangeSceneAction>( "Attach object", sourcePtr, ChangeSceneAction::Type::AddObject );
        bool attachSucess{ false };
        Object * toParent;
        if ( !sceneReorderCommand_.before )
        {
            toParent = sceneReorderCommand_.to;
            attachSucess = toParent->addChild( sourcePtr );
        }
        else
        {
            toParent = sceneReorderCommand_.to->parent();
            attachSucess = toParent->addChildBefore( sourcePtr, childTo );
        }
        if ( !attachSucess )
        {
            detachAction->action( HistoryAction::Type::Undo );
            //showModalMessage( "Cannot perform such reorder", NotificationType::Error );
            showModal( "Cannot perform such reorder", NotificationType::Error );
            dragOrDropFailed = true;
            break;
        }

        // change xf to preserve world location of the object
        std::shared_ptr<ChangeXfAction> xfAction;
        const auto fromParentXf = fromParent->worldXf();
        const auto toParentXf = toParent->worldXf();
        if ( fromParentXf != toParentXf )
        {
            xfAction = std::make_shared<ChangeXfAction>( "Xf", sourcePtr );
            source->setWorldXf( worldXf );
        }

        actionList.push_back( { detachAction, attachAction, xfAction } );
    }

    if ( dragOrDropFailed )
    {
        for ( int i = int( actionList.size() ) - 1; i >= 0; --i )
        {
            actionList[i].attachAction->action( HistoryAction::Type::Undo );
            actionList[i].detachAction->action( HistoryAction::Type::Undo );
            if ( actionList[i].xfAction )
                actionList[i].xfAction->action( HistoryAction::Type::Undo );
        }
    }
    else
    {
        SCOPED_HISTORY( "Reorder scene" );
        for ( const auto& moveAction : actionList )
        {
            AppendHistory( moveAction.detachAction );
            AppendHistory( moveAction.attachAction );
            if ( moveAction.xfAction )
                AppendHistory( moveAction.xfAction );
        }
    }
    sceneReorderCommand_ = {};
    dragTrigger_ = false;
}

void SceneObjectsListDrawer::updateSceneWindowScrollIfNeeded_()
{
    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    if ( !window )
        return;

    ScrollPositionPreservation scrollInfo;
    scrollInfo.relativeMousePos = ImGui::GetMousePos().y - window->Pos.y;
    scrollInfo.absLinePosRatio = window->ContentSize.y == 0.0f ? 0.0f : ( scrollInfo.relativeMousePos + window->Scroll.y ) / window->ContentSize.y;

    if ( nextFrameFixScroll_ )
    {
        nextFrameFixScroll_ = false;
        window->Scroll.y = std::clamp( prevScrollInfo_.absLinePosRatio * window->ContentSize.y - prevScrollInfo_.relativeMousePos, 0.0f, window->ScrollMax.y );
    }
    else if ( dragObjectsMode_ )
    {
        float relativeMousePosRatio = window->Size.y == 0.0f ? 0.0f : scrollInfo.relativeMousePos / window->Size.y;
        float shift = 0.0f;
        if ( relativeMousePosRatio < 0.05f )
            shift = ( relativeMousePosRatio - 0.05f ) * 25.0f - 1.0f;
        else if ( relativeMousePosRatio > 0.95f )
            shift = ( relativeMousePosRatio - 0.95f ) * 25.0f + 1.0f;

        auto newScroll = std::clamp( window->Scroll.y + shift, 0.0f, window->ScrollMax.y );
        if ( newScroll != window->Scroll.y )
        {
            window->Scroll.y = newScroll;
            getViewerInstance().incrementForceRedrawFrames();
        }
    }

    const ImGuiPayload* payloadCheck = ImGui::GetDragDropPayload();
    bool dragModeNow = payloadCheck && std::string_view( payloadCheck->DataType ) == "_TREENODE";
    if ( dragModeNow && !dragObjectsMode_ )
    {
        dragObjectsMode_ = true;
        nextFrameFixScroll_ = true;
        getViewerInstance().incrementForceRedrawFrames( 2, true );
    }
    else if ( !dragModeNow && dragObjectsMode_ )
    {
        dragObjectsMode_ = false;
        nextFrameFixScroll_ = true;
        getViewerInstance().incrementForceRedrawFrames( 2, true );
    }

    if ( !nextFrameFixScroll_ )
        prevScrollInfo_ = scrollInfo;
}

std::vector<Object*> SceneObjectsListDrawer::getPreSelection_( Object* meshclicked, bool isShift, bool isCtrl,
    const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all_objects )
{
    if ( selected.empty() || !isShift )
        return { meshclicked };

    const auto& first = isCtrl ? selected.back().get() : selected.front().get();

    auto firstIt = std::find_if( all_objects.begin(), all_objects.end(), [first] ( const std::shared_ptr<Object>& obj )
    {
        return obj.get() == first;
    } );
    auto clickedIt = std::find_if( all_objects.begin(), all_objects.end(), [meshclicked] ( const std::shared_ptr<Object>& obj )
    {
        return obj.get() == meshclicked;
    } );

    size_t start{ 0 };
    std::vector<Object*> res;
    if ( firstIt < clickedIt )
    {
        start = std::distance( all_objects.begin(), firstIt );
        res.resize( std::distance( firstIt, clickedIt + 1 ) );
    }
    else
    {
        start = std::distance( all_objects.begin(), clickedIt );
        res.resize( std::distance( clickedIt, firstIt + 1 ) );
    }
    for ( int i = 0; i < res.size(); ++i )
    {
        res[i] = all_objects[start + i].get();
    }
    return res;
}

void SceneObjectsListDrawer::updateSelection_( Object* objPtr, const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all )
{
    auto newSelection = getPreSelection_( objPtr, ImGui::GetIO().KeyShift, ImGui::GetIO().KeyCtrl, selected, all );
    if ( ImGui::GetIO().KeyCtrl )
    {
        for ( auto& sel : newSelection )
        {
            const bool select = ImGui::GetIO().KeyShift || !sel->isSelected();
            sel->select( select );
            if ( showNewSelectedObjects_ && select )
                sel->setGlobalVisibility( true );
        }
    }
    else
    {
        for ( const auto& data : selected )
        {
            auto inNewSelList = std::find( newSelection.begin(), newSelection.end(), data.get() );
            if ( inNewSelList == newSelection.end() )
                data->select( false );
        }
        for ( auto& sel : newSelection )
        {
            sel->select( true );
            if ( showNewSelectedObjects_ )
                sel->setGlobalVisibility( true );
        }
    }
}

}
