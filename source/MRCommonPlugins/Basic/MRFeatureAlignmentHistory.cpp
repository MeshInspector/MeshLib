#include "MRFeatureAlignmentController.h"
#include <MRViewer/MRAppendHistory.h>
#include <MRMesh/MRChangeXfAction.h>

namespace MR::FeatureAlignmentTool
{
Expected<void> ViewerTransformHistory::commit( const ObjectPtr &object, const AffineXf3f &local,
                                               const std::string &label )
{
    if ( !HistoryStore::getViewerInstance() )
        return unexpected( std::string( "Native undo history is unavailable. The part was not moved." ) );
    AppendHistory<ChangeXfAction>( label, object, local );
    return {};
}
} // namespace MR::FeatureAlignmentTool
