#pragma once

#include "MRViewer.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRSurfacePointPicker.h"
#include <unordered_map>
#include "MRMesh/MRObjectMeshHolder.h"


namespace MR
{

class SurfaceContoursWidget : public MultiListener<
    MouseDownListener,
    MouseMoveListener>
{
public:

    using PickerPointCallBack = std::function<void( std::shared_ptr<MR::ObjectMeshHolder> )>;
    using SurfaceContour = std::vector<std::shared_ptr<SurfacePointWidget>>;
    using SurfaceContours = std::unordered_map <std::shared_ptr<MR::ObjectMeshHolder>, SurfaceContour>;

    MRVIEWER_API bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;

    MRVIEWER_API std::shared_ptr<SurfacePointWidget> createPickWidget_( std::shared_ptr<MR::ObjectMeshHolder> obj_, const MeshTriPoint& pt );

    MRVIEWER_API void enable( bool isEnaled )
    {
        isPickerActive_ = isEnaled;
        if ( !isPickerActive_ )
            pickedPoints_.clear();
    }

    MRVIEWER_API void create( PickerPointCallBack onPointAdd, PickerPointCallBack onPointMove, PickerPointCallBack onPointMoveFinish, PickerPointCallBack onPointRemove )
    {
        onPointAdd_ = std::move( onPointAdd );
        onPointMove_ = std::move( onPointMove );
        onPointMoveFinish_ = std::move( onPointMoveFinish );
        onPointRemove_ = std::move( onPointRemove );

        clear();

        // 10 group to imitate plugins behavior
        connect( &getViewerInstance(), 10, boost::signals2::at_front );
    }

    MRVIEWER_API void clear()
    {
        pickedPoints_.clear();
        activeIndex_ = 0;
        activeObject_ = nullptr;
    }

    MRVIEWER_API void reset();

    [[nodiscard]] const SurfaceContour& getSurfaceContour( const std::shared_ptr<MR::ObjectMeshHolder> obj ) 
    {
        return pickedPoints_[obj];
    }

    [[nodiscard]] const SurfaceContours& getSurfaceContours() const
    {
        return pickedPoints_;
    }


    int activeIndex_{ 0 };
    std::shared_ptr<MR::ObjectMeshHolder> activeObject_ = nullptr;

    // TODO move ti to private !!!!! 
    bool isPickerActive_ = false;
    PickerPointCallBack onPointAdd_;
    PickerPointCallBack onPointMove_;
    PickerPointCallBack onPointMoveFinish_;
    PickerPointCallBack onPointRemove_;
    // data storage
    SurfaceContours pickedPoints_;

private:

    // SurfaceContoursWidget interlal variables 

    bool moveClosedPoint_ = false;
    bool activeChange_ = false;




    friend class AddPointActionBestPickerPoint;
    friend class RemovePointActionPickerPoint;
    friend class ChangePointActionPickerPoint;
};




}