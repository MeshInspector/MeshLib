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
private:
    static RenderObjectConstructorsHolder& instance_()
    {
        static RenderObjectConstructorsHolder holder;
        return holder;
    }
    HashMap<std::type_index, IRenderObjectConstructorLambda> map_;
};


RegisterRenderObjectConstructor::RegisterRenderObjectConstructor( const std::type_index& type, IRenderObjectConstructorLambda lambda )
{
    RenderObjectConstructorsHolder::addConstructorLambda( type, lambda );
}

std::unique_ptr<IRenderObject> createRenderObject( const VisualObject& visObj, const std::type_index& type )
{
    auto lambda = RenderObjectConstructorsHolder::findConstructorLambda( type );
    if ( !lambda )
        return {};
    return lambda( visObj );
}

}