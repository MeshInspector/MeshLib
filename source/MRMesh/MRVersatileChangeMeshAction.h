#pragma once
#include "MRHistoryAction.h"

namespace MR
{

/// Undo action for ObjectMesh points only (not topology) change;
/// It starts its life storing all points (uncompressed format),
/// but can be switched to store only modified points (compressed format)
class MRMESH_CLASS VersatileChangeMeshPointsAction : public HistoryAction
{
public:
    using Obj = ObjectMesh;

    /// use this constructor to remember object's mesh points in uncompressed format before making any changes in it
    MRMESH_API VersatileChangeMeshPointsAction( std::string name, const std::shared_ptr<ObjectMesh>& obj );

    MRMESH_API ~VersatileChangeMeshPointsAction();

    [[nodiscard]] MRMESH_API virtual std::string name() const override;

    MRMESH_API virtual void action( HistoryAction::Type t ) override;

    MRMESH_API static void setObjectDirty( const std::shared_ptr<ObjectMesh>& obj );

    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

    /// switch from uncompressed to compressed format to occupy less amount of memory
    MRMESH_API void compress();

private:
    std::unique_ptr<ChangeMeshPointsAction> stdAction_;
    std::unique_ptr<PartialChangeMeshPointsAction> diffAction_;
};

} // namespace MR
