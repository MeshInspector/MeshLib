#include "MRIRenderObject.h"
#include "MRphmap.h"

namespace MR
{

class RenderObjectConstructorsHolder
{
public:
    static IRenderObjectConstructorLambda findConstructorLambda( const std::type_index& type )
    {
        const auto& inst = instance_();
        auto it = inst.map_.find( type );
        if ( it == inst.map_.end() )
            return {};
        return it->second;
    }
    static void addConstructorLambda( const std::type_index& type, IRenderObjectConstructorLambda lambda )
    {
        auto& inst = instance_();
        inst.map_[type] = lambda;
    }
    static void removeConstructorLambda( const std::type_index& type )
    {
        auto& inst = instance_();
        inst.map_.erase( type );
    }
private:
    static RenderObjectConstructorsHolder& instance_()
    {
        static RenderObjectConstructorsHolder holder;
        return holder;
    }
    HashMap<std::type_index, IRenderObjectConstructorLambda> map_;
};


RegisterRenderObjectConstructor::RegisterRenderObjectConstructor( const std::type_index& type, IRenderObjectConstructorLambda lambda )
    : type_( type )
{
    RenderObjectConstructorsHolder::addConstructorLambda( type, lambda );
}

RegisterRenderObjectConstructor::~RegisterRenderObjectConstructor()
{
    RenderObjectConstructorsHolder::removeConstructorLambda( type_ );
}

std::unique_ptr<IRenderObject> createRenderObject( const VisualObject& visObj, const std::type_index& type )
{
    auto lambda = RenderObjectConstructorsHolder::findConstructorLambda( type );
    if ( !lambda )
        return {};
    return lambda( visObj );
}

}