#pragma once

#include "exports.h"

#include "MRMesh/MRExpected.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace MR::Mcp
{

/// This is used to build json schemas.
namespace Schema
{
    /// A common base class for the different schemas. Functions can accept this by value, it's fine to slice it.
    struct Base
    {
      protected:
        nlohmann::json json;
        Base( nlohmann::json json ) : json( std::move( json ) ) {}

      public:
        [[nodiscard]] const nlohmann::json& asJson() const & { return json; }
        [[nodiscard]] nlohmann::json&& asJson() && { return std::move( json ); }
    };

    /// An empty schema.
    struct Empty : Base
    {
        Empty() : Base( {} ) {}
    };

    /// A schema describing a scalar.
    struct Number : Base
    {
        Number()
            : Base( nlohmann::json::object( {
                { "type", "number" },
            } ) )
        {}
    };

    /// A schema describing a string.
    struct String : Base
    {
        String()
            : Base( nlohmann::json::object( {
                { "type", "string" },
            } ) )
        {}
    };

    /// A schema describing a boolean.
    struct Bool : Base
    {
        Bool()
            : Base( nlohmann::json::object( {
                { "type", "boolean" },
            } ) )
        {}
    };

    /// A schema describing an array of whatever is passed to the constructor.
    struct Array : Base
    {
        Array( Base elemSchema )
            : Base( nlohmann::json::object( {
                { "type", "array" },
                { "items", std::move( elemSchema ).asJson() },
            } ) )
        {}
    };

    /// A schema describing an object.
    /// Construct like this: `Object{}.addMember(...).addMember(...)`.
    struct Object : Base
    {
        Object()
            : Base( nlohmann::json::object( {
                { "type", "object" },
                { "properties", nlohmann::json::object() },
                { "required", nlohmann::json::array() },
            } ) )
        {}

        /// Add required member. Returns a reference to `*this`.
        Object &addMember( std::string name, Base schema ) &
        {
            json.at("required").push_back( name );
            addMemberOpt( std::move( name ), std::move( schema ) );
            return *this;
        }
        /// Add optional member. Returns a reference to `*this`.
        Object &addMemberOpt( std::string name, Base schema ) &
        {
            json.at( "properties" ).push_back( nlohmann::json::object_t::value_type( std::move( name ), std::move( schema ).asJson() ) );
            return *this;
        }

        /// Add required member. Returns a reference to `*this`.
        [[nodiscard]] Object&& addMember( std::string name, Base schema ) &&
        {
            addMember( std::move( name ), std::move( schema ) );
            return std::move( *this );
        }
        /// Add optional member. Returns a reference to `*this`.
        [[nodiscard]] Object&& addMemberOpt( std::string name, Base schema ) &&
        {
            addMemberOpt( std::move( name ), std::move( schema ) );
            return std::move( *this );
        }
    };
} // namespace Schema

/// Owns a HTTP MCP server (using the SSE protocol).
class Server
{
public:
    struct Params
    {
        std::string address = "127.0.0.1"; ///< You don't need to change this, unless you want to accept connections from the outside world.
        int port = 7887;
        std::string name; ///< A default string is set in the constructor.
        std::string version; ///< A default string is set in the constructor.

        friend bool operator==( const Params&, const Params& ) = default;

        MRMCP_API Params();
    };

    MRMCP_API Server();
    MRMCP_API Server( Server&& );
    MRMCP_API Server& operator=( Server&& );
    MRMCP_API ~Server();

    /// Those functions are allowed to throw, that's how you report errors to the MCP.
    using ToolFunc = std::function<nlohmann::json( const nlohmann::json& args )>;

    /// Registers a new tool.
    /// @param id An arbitrary function name, e.g. `foo.bar`.
    /// @param name A human/ai-readable name.
    /// @param desc A human/ai-readable explanation of what the tool does.
    /// @param inputSchema Describes the arguments. Normally it should be `Schema::Object{}` with some fields added.
    /// @param outputSchema Describes the returned JSON.
    /// Fails if the tool with this `id` already exists.
    /// Must be called early, before `setRunning(true)` is called for the first time, otherwise fails.
    /// Returns true on success. Asserts when returning false, so you don't have to check the return value.
    /// NOTE: Consult `docs/testing_mcp.md` for how to test your tool.
    MRMCP_API bool addTool( std::string id, std::string name, std::string desc, Schema::Base inputSchema, Schema::Base outputSchema, ToolFunc func );

    [[nodiscard]] MRMCP_API const Params& getParams() const;

    /// This restarts the server if necessary.
    MRMCP_API void setParams( Params params );

    [[nodiscard]] MRMCP_API bool isRunning() const;
    /// Returns true on success, including if the server is already running and you're trying to start it again.
    /// Stopping always returns true.
    MRMCP_API bool setRunning( bool enable );

    /// Tears down the running asio server (if any) and clears all registered tools.
    /// Use before unloading DLLs whose translation units called `addTool` — their captured
    /// std::function deleters dangle once those DLLs are unmapped, so a later `~Server`
    /// would segfault. Idempotent. Safe to call when nothing was registered.
    MRMCP_API void shutdown();

    /// Returns the list of currently-registered tools as a JSON array of MCP `tool` entries
    /// (`name`, optional `title`/`description`, `inputSchema`, `outputSchema`).
    /// Suitable for splicing into a `tools/list` response or persisting to a cache file.
    [[nodiscard]] MRMCP_API nlohmann::json dumpToolsAsJson() const;

    /// Atomically writes `{ "tools": dumpToolsAsJson() }` to @p path, creating parent
    /// directories as needed. Returns an error message on I/O failure.
    MRMCP_API Expected<void> saveToolsCache( const std::filesystem::path& path ) const;

    /// Optional predicate consulted before every tool dispatch, given the tool's id.
    /// Return {} to allow; return `unexpected("reason")` to block — the reason surfaces
    /// to the MCP client as the tool-call error.
    /// Evaluated per call, so changes (e.g. user sign-in) take effect immediately.
    using ToolValidator = std::function<Expected<void>( const std::string& toolId )>;
    MRMCP_API void setToolValidator( ToolValidator validator );

private:
    struct State;

    /// This is null until either `setParams()` or `setRunning(true)` is called for the first time.
    std::unique_ptr<State> state_;

    Params params_;
};

/// The global instance of the MCP server.
[[nodiscard]] MRMCP_API Server& getDefaultServer();

} // namespace MR
