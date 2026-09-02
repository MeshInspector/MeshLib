/*
 * Minimal reproducer: OpenSSL 3.6.3 built for arm64-windows with optimization
 * crashes while parsing a TLS 1.3 ServerHello.
 *
 * No curl, no cpr, no C++: only libssl/libcrypto.
 *
 * Build (native ARM64 developer prompt, vcpkg openssl):
 *   cl /O2 /MD arm64_openssl_mre.c /I %VCPKG%\installed\arm64-windows\include ^
 *      /link /LIBPATH:%VCPKG%\installed\arm64-windows\lib ^
 *      libssl.lib libcrypto.lib ws2_32.lib crypt32.lib
 *
 * Run:
 *   arm64_openssl_mre.exe [host] [port]      (default postman-echo.com 443)
 *
 * Expected: "handshake OK".
 * Actual, arm64 + Release OpenSSL DLLs:
 *   access violation reading 0x00000879 in
 *   libssl-3-arm64.dll!tls_parse_all_extensions (ssl/statem/extensions.c:747)
 *   at `ldr x8, [x0, #0x878]`, i.e. `s->cert` where the SSL_CONNECTION *s
 *   argument is 1 instead of a pointer (0x879 == 1 + offsetof(cert)).
 * The very same DLLs built without optimization complete the handshake, so the
 * fault is optimization-dependent. x64 is unaffected; so is Schannel.
 *
 * Certificate verification is deliberately left off: the crash happens while
 * parsing the ServerHello, long before any certificate is checked, so the
 * reproducer needs no CA store.
 */
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <stdint.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>

/* Reports the faulting instruction as module+RVA, so the crash can be located
 * in a disassembly without a debugger. Not part of the reproduction itself. */
static LONG NTAPI reportFault( EXCEPTION_POINTERS * info )
{
    const EXCEPTION_RECORD * er = info->ExceptionRecord;
    HMODULE mod = NULL;
    char path[MAX_PATH] = { 0 };

    if ( er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION )
        return EXCEPTION_CONTINUE_SEARCH;

    if ( GetModuleHandleExA( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                 GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
             (LPCSTR)er->ExceptionAddress, &mod ) &&
        mod != NULL )
        GetModuleFileNameA( mod, path, MAX_PATH );

    printf( "\nACCESS VIOLATION %s address 0x%llx\n"
            "  faulting pc 0x%llx = %s + 0x%llx\n",
        er->ExceptionInformation[0] == 0 ? "reading" : "writing",
        (unsigned long long)er->ExceptionInformation[1],
        (unsigned long long)(uintptr_t)er->ExceptionAddress,
        path[0] != '\0' ? path : "<unknown module>",
        (unsigned long long)( (uintptr_t)er->ExceptionAddress - (uintptr_t)mod ) );
#if defined( _M_ARM64 )
    printf( "  x0 = 0x%llx (the SSL_CONNECTION* argument; expected a pointer)\n",
        (unsigned long long)info->ContextRecord->X0 );
#endif
    fflush( stdout );
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

int main( int argc, char ** argv )
{
    const char * host = argc > 1 ? argv[1] : "postman-echo.com";
    const char * port = argc > 2 ? argv[2] : "443";
    SSL_CTX * ctx = NULL;
    BIO * bio = NULL;
    SSL * ssl = NULL;

#ifdef _WIN32
    AddVectoredExceptionHandler( 1, reportFault );
#endif

    printf( "OpenSSL: %s\n", OpenSSL_version( OPENSSL_VERSION_STRING ) );
    printf( "target : %s:%s (TLS 1.3 only)\n", host, port );

    ctx = SSL_CTX_new( TLS_client_method() );
    if ( ctx == NULL )
    {
        ERR_print_errors_fp( stderr );
        return 1;
    }
    /* The crash is on the TLS 1.3 ServerHello path (SSL_EXT_TLS1_3_SERVER_HELLO). */
    SSL_CTX_set_min_proto_version( ctx, TLS1_3_VERSION );
    SSL_CTX_set_verify( ctx, SSL_VERIFY_NONE, NULL );

    bio = BIO_new_ssl_connect( ctx );
    if ( bio == NULL )
    {
        ERR_print_errors_fp( stderr );
        return 1;
    }
    BIO_get_ssl( bio, &ssl );
    SSL_set_tlsext_host_name( ssl, host ); /* SNI */
    BIO_set_conn_hostname( bio, host );
    BIO_set_conn_port( bio, port );

    printf( "connecting and handshaking...\n" );
    fflush( stdout );

    if ( BIO_do_connect( bio ) <= 0 )
    {
        printf( "BIO_do_connect failed (no crash)\n" );
        ERR_print_errors_fp( stderr );
        BIO_free_all( bio );
        SSL_CTX_free( ctx );
        return 2;
    }

    printf( "handshake OK: %s / %s\n", SSL_get_version( ssl ), SSL_get_cipher( ssl ) );
    BIO_free_all( bio );
    SSL_CTX_free( ctx );
    return 0;
}
