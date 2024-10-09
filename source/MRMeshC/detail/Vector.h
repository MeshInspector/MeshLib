#pragma once

#include "TypeCast.h"

#include <MRMesh/MRVector.h>

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
    { }

    std::vector<T>* vec_;
};

template <typename T>
struct vector_wrapper : vector_wrapper_base<T>
{
    using base = vector_wrapper_base<T>;

    explicit vector_wrapper( std::vector<T>&& vec )
        : base( new std::vector<T>( std::move( vec ) ) )
    {
        base::invalidate();
    }

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
    {
        base::invalidate();
    }

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
}

#define MR_VECTOR_IMPL( Type ) MR_VECTOR_LIKE_IMPL( MR_CONCAT( Vector, Type ), Type )

#define VECTOR_WRAPPER( Type ) vector_wrapper<typename Type::value_type>
#define VECTOR_REF_WRAPPER( Type ) vector_ref_wrapper<typename Type::value_type>

#define REGISTER_VECTOR( Type )                                     \
ADD_AUTO_CAST( MR_CONCAT( MR, Type ), VECTOR_REF_WRAPPER( Type ) ); \
ADD_AUTO_CAST( VECTOR_REF_WRAPPER( Type ), MR_CONCAT( MR, Type ) ); \
ADD_AUTO_CAST( VECTOR_WRAPPER( Type ), MR_CONCAT( MR, Type ) );

#define VECTOR( ... ) vector_ref_wrapper( __VA_ARGS__ )

#define NEW_VECTOR( ... ) new vector_wrapper( __VA_ARGS__ )

#define RETURN_VECTOR( ... ) return auto_cast( VECTOR( __VA_ARGS__ ) )

#define RETURN_NEW_VECTOR( ... ) return auto_cast( NEW_VECTOR( __VA_ARGS__ ) )
