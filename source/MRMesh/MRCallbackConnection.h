#pragma once

#include <functional>
#include <memory>

namespace MR
{

template <typename Sig>
class WeakCallback;

/// owns a callback that a registry, possibly in another module, calls through WeakCallback only while this object is alive,
/// so a module revokes the callback by destroying this object, e.g. a static one at module unloading (cf. boost::signals2::scoped_connection);
/// construct it in the module keeping the WeakCallback, inside a non-inline function, because releasing the last WeakCallback
/// frees the shared state through code of the module that allocated it, which must not be the module of the callback
template <typename Sig>
class CallbackConnection
{
public:
    CallbackConnection( std::function<Sig> callback ) : callback_( std::make_shared<std::function<Sig>>( std::move( callback ) ) ) {}
    CallbackConnection( CallbackConnection&& ) = default;
    CallbackConnection& operator=( CallbackConnection&& ) = default;

private:
    friend class WeakCallback<Sig>;
    std::shared_ptr<std::function<Sig>> callback_;
};

/// non-owning side of a CallbackConnection, kept by the registry
template <typename Sig>
class WeakCallback
{
public:
    WeakCallback( const CallbackConnection<Sig>& connection ) : callback_( connection.callback_ ) {}

    /// returns the callback while its CallbackConnection is alive, and nullptr afterwards
    std::shared_ptr<const std::function<Sig>> lock() const { return callback_.lock(); }

private:
    std::weak_ptr<std::function<Sig>> callback_;
};

} //namespace MR
