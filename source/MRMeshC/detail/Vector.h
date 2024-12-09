#pragma once

#include "TypeCast.h"

#include <MRMesh/MRVector.h>

/// helper class to represent std::vector data as a pointer+length pair
/// NOTE: changing the source vector might invalidate the data pointer
//        it's your responsibility to update it by calling invalidate() after the vector's change, explicit or implicit
template <typename T>
struct vector_wrapper_base
{
    using value_type = T;

    value_type* data;
    size_t size;

    operator const std::vector<T>&() const
    {
        return *vec_;
    }

    operator std::vector<T>&()
    {
        return *vec_;
    }

    void invalidate()
    {
        data = vec_->data();
        size = vec_->size();
    }

protected:
    explicit vector_wrapper_base( std::vector<T>* vec )
        : vec_( vec )
    {
        invalidate();
    }

    std::vector<T>* vec_;
};

template <typename T>
struct vector_wrapper : vector_wrapper_base<T>
{
    using base = vector_wrapper_base<T>;

    explicit vector_wrapper( std::vector<T>&& vec )
        : base( new std::vector<T>( std::move( vec ) ) )
    { }

    template <typename I>
    explicit vector_wrapper( MR::Vector<T, I>&& vec )
        : vector_wrapper( std::move( vec.vec_ ) )
    { }

    ~vector_wrapper()
    {
        delete base::vec_;
    }
};

template <typename T>
struct vector_ref_wrapper : public vector_wrapper_base<T>
{
    using base = vector_wrapper_base<T>;

    explicit vector_ref_wrapper( const std::vector<T>& vec )
        : base( const_cast<std::vector<T>*>( &vec ) )
    { }

    template <typename I>
    explicit vector_ref_wrapper( const MR::Vector<T, I>& vec )
        : vector_ref_wrapper( vec.vec_ )
    { }
};

#define MR_VECTOR_LIKE_IMPL( ClassName, Type )                                               \
static_assert( sizeof( MR_CONCAT( MR, ClassName ) ) == sizeof( vector_ref_wrapper<Type> ) ); \
void MR_CONCAT( MR_CONCAT( mr, ClassName ), Invalidate )( MR_CONCAT( MR, ClassName )* vec )  \
{                                                                                            \
    reinterpret_cast<vector_ref_wrapper<Type>*>( vec )->invalidate();                        \
}                                                                                            \
void MR_CONCAT( MR_CONCAT( mr, ClassName ), Free )( MR_CONCAT( MR, ClassName )* vec )        \
{                                                                                            \
    delete reinterpret_cast<vector_wrapper<Type>*>( vec );                                   \
}                                                                                            \
MR_CONCAT( MR, ClassName )* MR_CONCAT( MR_CONCAT( mr, ClassName ), New )(void)               \
{                                                                                            \
    return reinterpret_cast<MR_CONCAT( MR, ClassName )*>( new vector_wrapper<Type>( std::vector<Type>() ) ); \
}

#define MR_VECTOR_IMPL( Type ) MR_VECTOR_LIKE_IMPL( MR_CONCAT( Vector, Type ), Type )

#define VECTOR_WRAPPER( Type ) vector_wrapper<typename Type::value_type>
#define VECTOR_REF_WRAPPER( Type ) vector_ref_wrapper<typename Type::value_type>

#define REGISTER_VECTOR_LIKE( ClassName, Type )       \
ADD_AUTO_CAST( ClassName, vector_ref_wrapper<Type> ); \
ADD_AUTO_CAST( vector_ref_wrapper<Type>, ClassName ); \
ADD_AUTO_CAST( vector_wrapper<Type>, ClassName );

#define REGISTER_VECTOR( Type )                                     \
ADD_AUTO_CAST( MR_CONCAT( MR, Type ), VECTOR_REF_WRAPPER( Type ) ); \
ADD_AUTO_CAST( VECTOR_REF_WRAPPER( Type ), MR_CONCAT( MR, Type ) ); \
ADD_AUTO_CAST( VECTOR_WRAPPER( Type ), MR_CONCAT( MR, Type ) );

#define VECTOR( ... ) vector_ref_wrapper( __VA_ARGS__ )

#define NEW_VECTOR( ... ) new vector_wrapper( __VA_ARGS__ )

#define RETURN_VECTOR( ... ) return auto_cast( VECTOR( __VA_ARGS__ ) )

#define RETURN_NEW_VECTOR( ... ) return auto_cast( NEW_VECTOR( __VA_ARGS__ ) )
