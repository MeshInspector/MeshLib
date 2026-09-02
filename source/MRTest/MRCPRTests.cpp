#include "MRMesh/MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__)
#include <cpr/cpr.h>
#include <cpr/cprver.h>
#include <curl/curl.h>
#include "MRPch/MRSpdlog.h"
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include "MRPch/MRFmt.h"
#ifdef _WIN32
#include "MRPch/MRWinapi.h"
#endif

constexpr int MAX_RETRIES = 10;
constexpr std::chrono::seconds COOLDOWN_PERIOD { 10 };

TEST( MRViewer, CPRSslBackends )
{
    spdlog::info( "cpr version: {}", CPR_VERSION );
    spdlog::info( "libcurl version: {}", curl_version() );

    const curl_version_info_data * info = curl_version_info( CURLVERSION_NOW );
    ASSERT_NE( info, nullptr );
    spdlog::info( "libcurl default SSL backend: {}", info->ssl_version ? info->ssl_version : "<none>" );

#if LIBCURL_VERSION_NUM >= 0x075400 // 7.84.0: cainfo/capath fields added (CURLVERSION_TENTH)
    spdlog::info( "libcurl compiled-in CAINFO: {}", info->cainfo ? info->cainfo : "<none>" );
    spdlog::info( "libcurl compiled-in CAPATH: {}", info->capath ? info->capath : "<none>" );
#endif

    if ( const char * v = std::getenv( "CURL_CA_BUNDLE" ) )
        spdlog::info( "env CURL_CA_BUNDLE: {}", v );
    else
        spdlog::info( "env CURL_CA_BUNDLE: <unset>" );
    if ( const char * v = std::getenv( "SSL_CERT_FILE" ) )
        spdlog::info( "env SSL_CERT_FILE: {}", v );
    else
        spdlog::info( "env SSL_CERT_FILE: <unset>" );

    const curl_ssl_backend ** avail = nullptr;
    // Documented query idiom: CURLSSLBACKEND_NONE returns CURLSSLSET_UNKNOWN_BACKEND and fills `avail`.
    curl_global_sslset( CURLSSLBACKEND_NONE, nullptr, &avail );
    if ( avail )
    {
        for ( int i = 0; avail[i]; ++i )
            spdlog::info( "libcurl SSL backend [{}]: {} (id={})", i, avail[i]->name, (int)avail[i]->id );
    }
    else
    {
        spdlog::info( "libcurl reports a single SSL backend (no list available)" );
    }
}

namespace
{

#ifdef _WIN32
// The default handler logs only the faulting DATA address; the faulting PC (and the
// register that held the bad value) is what identifies the instruction. Unwinding
// through OpenSSL's arm64 assembly fails, so the stacktrace alone is useless here.
LONG NTAPI logFaultPc( EXCEPTION_POINTERS * info )
{
    const auto * er = info->ExceptionRecord;
    if ( er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION )
        return EXCEPTION_CONTINUE_SEARCH;

    const auto pc = (uintptr_t)er->ExceptionAddress;
    HMODULE mod = nullptr;
    char path[MAX_PATH] = {};
    uintptr_t rva = 0;
    if ( GetModuleHandleExA( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
             (LPCSTR)pc, &mod ) && mod )
    {
        GetModuleFileNameA( mod, path, MAX_PATH );
        rva = pc - (uintptr_t)mod;
    }
    spdlog::critical( "FAULT pc={:#x} module={} base={:#x} rva={:#x} access={} addr={:#x}",
        pc, path[0] ? path : "<unknown>", (uintptr_t)mod, rva,
        (size_t)er->ExceptionInformation[0], (size_t)er->ExceptionInformation[1] );
#if defined( _M_ARM64 )
    const auto * c = info->ContextRecord;
    spdlog::critical( "FAULT arm64 pc={:#x} lr={:#x} sp={:#x} fp={:#x}",
        (uintptr_t)c->Pc, (uintptr_t)c->Lr, (uintptr_t)c->Sp, (uintptr_t)c->Fp );
    std::string regs;
    for ( int i = 0; i < 29; ++i )
        regs += fmt::format( "x{}={:#x} ", i, (uintptr_t)c->X[i] );
    spdlog::critical( "FAULT regs {}", regs );
    // Fingerprint the code that actually ran, so the disassembly can be matched offline.
    if ( mod )
    {
        const auto * dos = (const IMAGE_DOS_HEADER *)mod;
        const auto * nt = (const IMAGE_NT_HEADERS64 *)( (const char *)mod + dos->e_lfanew );
        spdlog::critical( "FAULT module stamp={:#x} sizeOfImage={:#x} checksum={:#x}",
            (unsigned)nt->FileHeader.TimeDateStamp, (unsigned)nt->OptionalHeader.SizeOfImage,
            (unsigned)nt->OptionalHeader.CheckSum );
    }
    std::string code;
    const auto * words = (const unsigned int *)( pc - 16 );
    for ( int i = 0; i < 9; ++i )
        code += fmt::format( "{}{:08x} ", i == 4 ? "[pc]" : "", words[i] );
    spdlog::critical( "FAULT code {}", code );
    // Return addresses live on the stack; a raw scan beats a broken unwinder.
    std::string cands;
    const auto * stack = (const uintptr_t *)c->Sp;
    for ( int i = 0; i < 64; ++i )
    {
        const auto v = stack[i];
        HMODULE m = nullptr;
        char p2[MAX_PATH] = {};
        if ( v > 0x10000 && GetModuleHandleExA( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                 (LPCSTR)v, &m ) && m )
        {
            GetModuleFileNameA( m, p2, MAX_PATH );
            const std::string_view sv( p2 );
            const auto slash = sv.find_last_of( "/" );
            cands += fmt::format( "[sp+{:#x}] {}+{:#x}  ", i * 8,
                slash == std::string_view::npos ? sv : sv.substr( slash + 1 ), v - (uintptr_t)m );
        }
    }
    spdlog::critical( "FAULT stack return-address candidates: {}", cands );
#endif
    return EXCEPTION_CONTINUE_SEARCH;
}

const struct FaultPcLoggerInstaller
{
    FaultPcLoggerInstaller()
    {
        AddVectoredExceptionHandler( 1, logFaultPc );
    }
} faultPcLoggerInstaller;
#endif

size_t discardBody( char *, size_t size, size_t nmemb, void * )
{
    return size * nmemb;
}

int curlTrace( CURL *, curl_infotype type, char * data, size_t size, void * )
{
    if ( type == CURLINFO_TEXT )
        spdlog::info( "curl: {}", std::string_view( data, size ) );
    return 0;
}

} // namespace

// Raw libcurl request without cpr: tells a cpr bug from a libcurl one.
// Every step is logged so a crash points at the exact call that faulted.
// nativeCa mirrors what cpr::SslOptions{} does (CURLSSLOPT_NATIVE_CA), which is
// a no-op for Schannel but makes OpenSSL import the Windows certificate store.
static void rawGet( bool nativeCa )
{
    const char * backendEnv = std::getenv( "CURL_SSL_BACKEND" );
    spdlog::info( "raw: env CURL_SSL_BACKEND={}", backendEnv ? backendEnv : "<unset>" );

    spdlog::info( "raw: curl_global_init..." );
    const auto initRes = curl_global_init( CURL_GLOBAL_ALL );
    spdlog::info( "raw: curl_global_init -> {}", (int)initRes );
    ASSERT_EQ( initRes, CURLE_OK );

    spdlog::info( "raw: libcurl now reports: {}", curl_version() );

    spdlog::info( "raw: curl_easy_init..." );
    CURL * h = curl_easy_init();
    spdlog::info( "raw: curl_easy_init -> {}", (const void *)h );
    ASSERT_NE( h, nullptr );

    curl_easy_setopt( h, CURLOPT_URL, "https://postman-echo.com/get" );
    curl_easy_setopt( h, CURLOPT_TIMEOUT_MS, 8000L );
    curl_easy_setopt( h, CURLOPT_WRITEFUNCTION, &discardBody );
    curl_easy_setopt( h, CURLOPT_VERBOSE, 1L );
    curl_easy_setopt( h, CURLOPT_DEBUGFUNCTION, &curlTrace );
    if ( nativeCa )
    {
        spdlog::info( "raw: setting CURLSSLOPT_NATIVE_CA" );
        curl_easy_setopt( h, CURLOPT_SSL_OPTIONS, (long)CURLSSLOPT_NATIVE_CA );
    }
    spdlog::info( "raw: options set (nativeCa={})", nativeCa );

    spdlog::info( "raw: curl_easy_perform..." );
    const auto res = curl_easy_perform( h );
    spdlog::info( "raw: curl_easy_perform -> {} ({})", (int)res, curl_easy_strerror( res ) );

    long code = 0;
    curl_easy_getinfo( h, CURLINFO_RESPONSE_CODE, &code );
    spdlog::info( "raw: response code {}", code );
    curl_easy_cleanup( h );
    EXPECT_EQ( res, CURLE_OK );
}

TEST( MRViewer, CurlRawGet )
{
    rawGet( false );
}

TEST( MRViewer, CurlRawGetNativeCa )
{
    rawGet( true );
}

// CPRTestGet with curl's verbose trace, to see how far the request gets.
TEST( MRViewer, CPRTestGetVerbose )
{
    spdlog::info( "cpr verbose: constructing session" );
    cpr::Session session;
    session.SetUrl( cpr::Url{ "https://postman-echo.com/get" } );
    session.SetTimeout( cpr::Timeout{ 8000 } );
    session.SetVerbose( cpr::Verbose{ true } );
    spdlog::info( "cpr verbose: session ready, calling Get" );
    const auto resp = session.Get();
    spdlog::info( "cpr verbose: status {}, curl error {} ({})", resp.status_code,
        (int32_t)resp.error.code, resp.error.message );
}

TEST( MRViewer, CPRTestGet )
{
    std::string baseUrl = "https://postman-echo.com/get";
    if ( const char * overrideUrl = std::getenv( "MRTEST_OVERRIDE_ECHO_SERVER_URL" ) )
        baseUrl = overrideUrl;

    std::vector<std::pair<std::string, std::string>> params = { {"1","1"} };

    cpr::Parameters parameters;
    for ( const auto& [key, val] : params )
        parameters.Add( cpr::Parameter( key, val ) );

    for ( int i = 0; i < MAX_RETRIES; ++i )
    {
        // cpr::SslOptions{} triggers CURLSSLOPT_NATIVE_CA on Windows OpenSSL.
        const auto resp = cpr::Get( cpr::Url{ baseUrl }, cpr::Timeout{ 3000 }, parameters, cpr::SslOptions{} );
        auto code = resp.status_code;
        if ( code == 200 )
            break;
        spdlog::warn( "status code {} after try #{}", code, i + 1 );
        if ( resp.error )
            spdlog::warn( "curl error {}: {}", (int32_t)resp.error.code, resp.error.message );
        if ( i + 1 == MAX_RETRIES )
        {
            EXPECT_EQ( code, 200 );
        }
        std::this_thread::sleep_for( COOLDOWN_PERIOD );
    }
}

TEST( MRViewer, CPRTestPost )
{
    std::string baseUrl = "https://postman-echo.com/post";
    if ( const char * overrideUrl = std::getenv( "MRTEST_OVERRIDE_ECHO_SERVER_URL" ) )
        baseUrl = overrideUrl;

    std::vector<std::pair<std::string, std::string>> params = { {"1","1"} };

    std::vector<cpr::Pair> pairs;
    pairs.reserve( params.size() );
    for ( const auto& [key, val] : params )
        pairs.push_back( { key,val } );

    cpr::Payload payload( pairs.begin(), pairs.end() );

    for ( int i = 0; i < MAX_RETRIES; ++i )
    {
        // cpr::SslOptions{} triggers CURLSSLOPT_NATIVE_CA on Windows OpenSSL.
        const auto resp = cpr::Post( cpr::Url{ baseUrl }, cpr::Timeout{ 3000 }, payload, cpr::SslOptions{} );
        auto code = resp.status_code;
        if ( code == 200 )
            break;
        spdlog::warn( "status code {} after try #{}", code, i + 1 );
        if ( resp.error )
            spdlog::warn( "curl error {}: {}", (int32_t)resp.error.code, resp.error.message );
        if ( i + 1 == MAX_RETRIES )
        {
            EXPECT_EQ( code, 200 );
        }
        std::this_thread::sleep_for( COOLDOWN_PERIOD );
    }
}
#endif
