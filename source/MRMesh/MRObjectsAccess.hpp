#include <stack>

namespace MR
{

template<typename ObjectT>
std::shared_ptr<ObjectT> asSelectivityType( std::shared_ptr<Object> obj, const ObjectSelectivityType& type )
{
    auto visObj = std::dynamic_pointer_cast<ObjectT>( std::move( obj ) );
    if ( !visObj )
        return visObj;
    switch ( type )
    {
    case ObjectSelectivityType::Selectable:
        if ( visObj->isAncillary() )
            visObj.reset();
        break;
    case ObjectSelectivityType::Selected:
        if ( !visObj->isSelected() )
            visObj.reset();
        break;
    case ObjectSelectivityType::Any:
        break;
    }
    return visObj;
}

template<typename ObjectT>
void appendObjectFromTreeRecursive( std::shared_ptr<Object> obj, std::vector<std::shared_ptr<ObjectT>>& res, const ObjectSelectivityType& type )
{
    if ( !obj )
        return;

    if ( auto visObj = asSelectivityType<ObjectT>( obj, type ) )
        res.push_back( visObj );
    for ( const auto& child : obj->children() )
        appendObjectFromTreeRecursive( child, res, type );
}

template<typename ObjectT>
std::vector<std::shared_ptr<ObjectT>> getAllObjectsInTree( Object* root, const ObjectSelectivityType& type/* = ObjectSelectivityType::Selectable*/ )
{
    std::vector<std::shared_ptr<ObjectT>> res;
    if ( !root )
        return res;

    for ( const auto& child : root->children() )
        appendObjectFromTreeRecursive( child, res, type );

    return res;
}

template<typename ObjectT>
std::vector<std::shared_ptr<ObjectT>> getTopmostObjects( Object* root, const ObjectSelectivityType& type/* = ObjectSelectivityType::Selectable*/, bool visibilityCheck/* = false*/)
{
    std::vector<std::shared_ptr<ObjectT>> res;
    if ( !root )
        return res;

    std::stack<Object*> todo;
    todo.push( root );
    while ( !todo.empty() )
    {
        root = todo.top();
        todo.pop();
        const auto & children = root->children();
        for ( const auto& child : children )
        {
            if ( !child || ( visibilityCheck && !child->isVisible() ) )
                continue;
            if ( auto visObj = asSelectivityType<ObjectT>( child, type ) )
                res.push_back( std::move( visObj ) );
            else
                todo.push( child.get() );
        }
    }

    return res;
}

template<typename ObjectT>
std::shared_ptr<ObjectT> getDepthFirstObject( Object* root, const ObjectSelectivityType& type )
{
    if ( !root )
        return {};
    
    std::stack<Object*> todo;
    todo.push( root );
    while ( !todo.empty() )
    {
        root = todo.top();
        todo.pop();
        const auto & children = root->children();
        for ( const auto& child : children )
            if ( auto visObj = asSelectivityType<ObjectT>( child, type ) )
                return visObj;
        for ( auto it = children.rbegin(); it != children.rend(); ++it )
        {
            if ( auto * child = it->get() )
                todo.push( child );
        }
    }
    return {};
}

}
