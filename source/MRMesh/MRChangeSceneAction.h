#pragma once
#include "MRHistoryAction.h"
#include "MRObject.h"
#include <memory>


namespace MR
{

class MRMESH_CLASS ChangeSceneAction : public HistoryAction
{
public:
    enum class Type
    {
        AddObject,
        RemoveObject
    };
    // Constructed before removal or addiction
    MRMESH_API ChangeSceneAction( const std::string& name, const std::shared_ptr<Object>& obj, Type type );

    virtual std::string name() const override { return name_; }

    MRMESH_API virtual void action( HistoryAction::Type actionType ) override;

private:
    // updates parent and next child if it was not preset before
    void updateParent_();

    Object* parent_{ nullptr };
    std::shared_ptr<Object> nextObj_; // next child of parent (needed to insert child to correct place of tree)
    std::shared_ptr<Object> obj_;
    std::string name_;
    Type type_;
};

}