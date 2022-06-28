#pragma once
#include "MRMeshViewerPlugin.h"
#include "MRMeshViewer.h"
#include "MRViewerEventsListener.h"
#include "exports.h"

namespace MR
{
class Save : public MR::ViewerPlugin, public MultiListener<SaveListener>
{
public:
    MRVIEWER_API virtual void init( Viewer* _viewer ) override;
    MRVIEWER_API virtual void shutdown() override;
protected:
    MRVIEWER_API virtual bool save_( const std::filesystem::path & filename ) override;
};

}
