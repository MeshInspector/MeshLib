#include "MRObjectFactory.h"
#include "MRphmap.h"

namespace MR
{

class ObjectMakers
{
public:
    static ObjectMakers& instance()
    {
        static ObjectMakers the;
        return the;
    }

    void add( const std::string & className, ObjectMakerFunc * f ) 
    {
        assert ( f );
        if ( !f )
            return;
        std::unique_lock lock( mutex_ );
        map_[ className ] = f;
    }
    void del( const std::string & className ) 
    {
        std::unique_lock lock( mutex_ );
        map_.erase( className );
    }
    std::shared_ptr<Object> createObject( const std::string & className ) 
    {
        std::unique_lock lock( mutex_ );
        auto it = map_.find( className );
        if ( it == map_.end() )
            return {}; // no name registered
        return (*it->second)();
    }

private:
    ObjectMakers() = default;

    std::mutex mutex_;
    HashMap<std::string, ObjectMakerFunc*> map_;
};

std::shared_ptr<Object> createObject( const std::string & className )
{
    return ObjectMakers::instance().createObject( className );
}

ObjectFactoryBase::ObjectFactoryBase( std::string className, ObjectMakerFunc * creator ) 
    : className_( std::move( className ) )
{
    ObjectMakers::instance().add( className_, creator );
}

ObjectFactoryBase::~ObjectFactoryBase()
{
    ObjectMakers::instance().del( className_ );
}

} //namespace MR
