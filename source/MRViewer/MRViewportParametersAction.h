#pragma once
#include <MRMesh/MRHistoryAction.h>
#include <MRMesh/MRHeapBytes.h>
#include "MRViewport.h"
#include "MRViewer.h"

namespace MR
{

class ViewportParametersAction : public HistoryAction
{
public:
    ViewportParametersAction( std::string name, ViewportMask viewports = ViewportMask::all() ) : 
        name_(name)
    {
        Viewer* viewer = &getViewerInstance();
        for ( ViewportId id : viewports & viewer->getPresentViewports() )
            saveParameters_.push_back( { id, viewer->viewport( id ).getParameters() } );
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        Viewer* viewer = &getViewerInstance();
        for ( auto& [id, parameters] : saveParameters_ )
            if ( viewer->getPresentViewports().contains( id ) )
            {
                Viewport::Parameters t = viewer->viewport( id ).getParameters();
                viewer->viewport( id ).setParameters( parameters );
                parameters = t;
            }
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + MR::heapBytes( saveParameters_ );
    }

private:
    std::vector<std::pair<ViewportId, Viewport::Parameters>> saveParameters_;

    std::string name_;
};

} // namespace MR
