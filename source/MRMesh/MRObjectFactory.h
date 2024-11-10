#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"
#include <memory>
#include <string>

namespace MR
{

/// \ingroup DataModelGroup
/// \{

/// the function to create new object instance by registered class name
MRMESH_API std::shared_ptr<Object> createObject( const std::string & className );

/// use this macro to register a class in the factory before calling createObject
#define MR_ADD_CLASS_FACTORY( className ) \
    static MR::ObjectFactory<className> className##_Factory_{ #className };

using ObjectMakerFunc = std::shared_ptr<Object>();

class ObjectFactoryBase
{
public:
    MR_BIND_IGNORE MRMESH_API ObjectFactoryBase( std::string className, ObjectMakerFunc * creator );
    MRMESH_API ~ObjectFactoryBase();

private:
    std::string className_;
};

template<typename T>
class ObjectFactory : public ObjectFactoryBase
{
public:
    static_assert( std::is_base_of_v<Object, T>, "MR::Object is not base of T" );

    ObjectFactory( std::string className )
        : ObjectFactoryBase( std::move( className ),
            []() { return std::static_pointer_cast<Object>( std::make_shared<T>() ); } )
    { }
};

/// \}

} // namespace MR
