#pragma once
#include "MRMeshViewerPlugin.h"
#include "MRMeshViewer.h"
#include "MRViewerEventsListener.h"
#include "exports.h"

namespace MR
{

class Open : public MR::ViewerPlugin, public MultiListener<LoadListener, DragDropListener>
{
public:
    MRVIEWER_API virtual void init( Viewer* _viewer ) override;
    MRVIEWER_API virtual void shutdown() override;

protected:
    MRVIEWER_API virtual bool load_( const std::filesystem::path& filename ) override;
    MRVIEWER_API virtual bool dragDrop_( const std::vector<std::filesystem::path>& paths ) override;
};

MRVIEWER_API extern Open OpenInstance;

}
