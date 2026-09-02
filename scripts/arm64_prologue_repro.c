/*
 * Reducing the MSVC ARM64 /O2 /Gs0 prologue defect: `bl __chkstk` is emitted in
 * the leading part of the prologue while the callee-saved spills - including lr -
 * are deferred past an early-exit test, so the call destroys the return address
 * before it is saved.
 *
 *   cl /c /O2 /Gs0 arm64_prologue_repro.c
 *   dumpbin /disasm:nobytes arm64_prologue_repro.obj
 *   python arm64_prologue_report.py <listing>
 *
 * v7 is the known-bad control (mirrors OpenSSL's tls_parse_all_extensions).
 * v11..v18 strip one thing at a time to find the smallest form that reproduces.
 */

/* ---- scaffolding used by the v7 control only ------------------------------- */

typedef struct Cert
{
    char pad[0x88];
    unsigned long long meths_count;
} Cert;

typedef struct Conn
{
    char pad[0x878];
    Cert *cert;
} Conn;

typedef struct Def
{
    int context;
    int ( *final )( Conn *, int, int );
} Def;

extern const Def defs[29];
extern int parse_one( Conn *s, unsigned long long i, int ctx, void *exts, void *x, unsigned long long idx );

/* v7: control - known BAD at /O2 /Gs0. */
int v7( Conn *s, int context, void *exts, void *x, unsigned long long chainidx, int fin )
{
    unsigned long long i, numexts = 29 + s->cert->meths_count;
    unsigned long long a = ( unsigned long long )exts, b = ( unsigned long long )x;
    unsigned long long c = chainidx, d = ( unsigned long long )context;
    unsigned long long e = ( unsigned long long )fin, f = numexts, g = 0, h = 0;
    const Def *dd;

    for ( i = 0; i < numexts; i++ )
    {
        if ( !parse_one( s, i, context, exts, x, chainidx ) )
            return 0;
        g += a + b + c + d + e + f;
        h ^= g + i;
    }
    if ( fin )
    {
        for ( i = 0, dd = defs; i < 29; i++, dd++ )
            if ( ( dd->context & context ) != 0 && dd->final != 0 && !dd->final( s, context, 1 ) )
                return 0;
    }
    return ( int )( ( g ^ h ) & 1 );
}

/* ---- reduction candidates -------------------------------------------------- */

extern int call1( unsigned long long i );
extern void sink( unsigned long long *p );

typedef struct Count
{
    unsigned long long n;
} Count;

/* v11: v7 with a plain one-field struct instead of the padded double dereference,
 * and a one-argument callee. */
int v11( Count *s, int context, void *exts, void *x, unsigned long long chainidx, int fin )
{
    unsigned long long i, numexts = 29 + s->n;
    unsigned long long a = ( unsigned long long )exts, b = ( unsigned long long )x;
    unsigned long long c = chainidx, d = ( unsigned long long )context;
    unsigned long long e = ( unsigned long long )fin, f = numexts, g = 0, h = 0;

    for ( i = 0; i < numexts; i++ )
    {
        if ( !call1( i ) )
            return 0;
        g += a + b + c + d + e + f;
        h ^= g + i;
    }
    return ( int )( ( g ^ h ) & 1 );
}

/* v12: loop bound straight from a parameter - no struct, no dereference. */
int v12( unsigned long long n, unsigned long long a, unsigned long long b, unsigned long long c,
    unsigned long long d, unsigned long long e, unsigned long long f, unsigned long long h )
{
    unsigned long long i, g = 0, k = 0;

    for ( i = 0; i < n; i++ )
    {
        if ( !call1( i ) )
            return 0;
        g += a + b + c + d + e + f + h;
        k ^= g + i;
    }
    return ( int )( ( g ^ k ) & 1 );
}

/* v13: v12 without the second accumulator. */
int v13( unsigned long long n, unsigned long long a, unsigned long long b, unsigned long long c,
    unsigned long long d, unsigned long long e, unsigned long long f, unsigned long long h )
{
    unsigned long long i, g = 0;

    for ( i = 0; i < n; i++ )
    {
        if ( !call1( i ) )
            return 0;
        g += a + b + c + d + e + f + h + i;
    }
    return ( int )g;
}

/* v14: v13 with six parameters instead of eight. */
int v14( unsigned long long n, unsigned long long a, unsigned long long b, unsigned long long c,
    unsigned long long d, unsigned long long e )
{
    unsigned long long i, g = 0;

    for ( i = 0; i < n; i++ )
    {
        if ( !call1( i ) )
            return 0;
        g += a + b + c + d + e + i;
    }
    return ( int )g;
}

/* v15: v13 with the sum hoisted out of the loop, so fewer values stay live. */
int v15( unsigned long long n, unsigned long long a, unsigned long long b, unsigned long long c,
    unsigned long long d, unsigned long long e, unsigned long long f, unsigned long long h )
{
    unsigned long long i, g = 0, s = a + b + c + d + e + f + h;

    for ( i = 0; i < n; i++ )
    {
        if ( !call1( i ) )
            return 0;
        g += s + i;
    }
    return ( int )g;
}

/* v16: v13 with the early exit written explicitly rather than left to the loop. */
int v16( unsigned long long n, unsigned long long a, unsigned long long b, unsigned long long c,
    unsigned long long d, unsigned long long e, unsigned long long f, unsigned long long h )
{
    unsigned long long i, g = 0;

    if ( n == 0 )
        return 1;

    for ( i = 0; i < n; i++ )
    {
        if ( !call1( i ) )
            return 0;
        g += a + b + c + d + e + f + h + i;
    }
    return ( int )g;
}

/* v17: v13 with no early exit at all - the loop body always runs once. */
int v17( unsigned long long n, unsigned long long a, unsigned long long b, unsigned long long c,
    unsigned long long d, unsigned long long e, unsigned long long f, unsigned long long h )
{
    unsigned long long i, g = 0;

    for ( i = 0; i <= n; i++ )
    {
        if ( !call1( i ) )
            return 0;
        g += a + b + c + d + e + f + h + i;
    }
    return ( int )g;
}

/* v18: v13 plus an address-taken local, which made the prologue correct at v7's
 * size - included to confirm that still holds in the reduced form. */
int v18( unsigned long long n, unsigned long long a, unsigned long long b, unsigned long long c,
    unsigned long long d, unsigned long long e, unsigned long long f, unsigned long long h )
{
    unsigned long long i, g = 0, local;

    for ( i = 0; i < n; i++ )
    {
        if ( !call1( i ) )
            return 0;
        g += a + b + c + d + e + f + h + i;
    }
    local = g;
    sink( &local );
    return ( int )local;
}
