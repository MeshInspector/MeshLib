#pragma once

#include "TypeCast.h"

#include <MRMesh/MRVector.h>

template <typename T>
struct vector_wrapper
{
    using value_type = T;

    value_type* data;
    size_t size;

    explicit vector_wrapper( const std::vector<T>& vec )
        : vec_( &vec )
        , own_( false )
    {
        invalidate();
    }

    explicit vector_wrapper( std::vector<T>&& vec )
        : vec_( new std::vector<T>( std::move( vec ) ) )
        , own_( true )
    {
        invalidate();
    }

    template <typename I>
    explicit vector_wrapper( const MR::Vector<T, I>& vec )
        : vec_( &vec.vec_ )
        , own_( false )
    {
        invalidate();
    }

    template <typename I>
    explicit vector_wrapper( MR::Vector<T, I>&& vec )
        : vec_( new std::vector<T>( std::move( vec.vec_ ) ) )
        , own_( true )
    {
        invalidate();
    }

    ~vector_wrapper()
    {
        if ( own_ )
            delete vec_;
    }

    operator const std::vector<T>&() const
    {
        return *vec_;
    }

    void invalidate()
    {
        data = vec_->data();
        size = vec_->size();
    }

private:
    std::vector<T>* vec_;
    bool own_;
};

#define MR_VECTOR_LIKE_IMPL( ClassName, Type )                                              \
void MR_CONCAT( MR_CONCAT( mr, ClassName ), Invalidate )( MR_CONCAT( MR, ClassName )* vec ) \
{                                                                                           \
    reinterpret_cast<vector_wrapper<Type>*>( vec )->invalidate();                           \
}                                                                                           \
void MR_CONCAT( MR_CONCAT( mr, ClassName ), Free )( MR_CONCAT( MR, ClassName )* vec )       \
{                                                                                           \
    delete reinterpret_cast<vector_wrapper<Type>*>( vec );                                  \
}

#define MR_VECTOR_IMPL( Type ) MR_VECTOR_LIKE_IMPL( MR_CONCAT( Vector, Type ), Type )

#define REGISTER_VECTOR( Type ) REGISTER_AUTO_CAST2( MR_CONCAT( MR, Type ), vector_wrapper<typename Type::value_type> )

#define NEW_VECTOR( ... ) new vector_wrapper( __VA_ARGS__ )

#define RETURN_NEW_VECTOR( ... ) return auto_cast( NEW_VECTOR( __VA_ARGS__ ) )
