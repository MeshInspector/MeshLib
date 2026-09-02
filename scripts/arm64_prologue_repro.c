/*
 * Candidate shapes for the MSVC ARM64 /O2 /Gs0 prologue defect:
 * `bl __chkstk` is emitted before `str x30`, so the return address is destroyed
 * before it is spilled.
 *
 *   cl /c /O2 /Gs0 arm64_prologue_repro.c
 *   dumpbin /disasm:nobytes arm64_prologue_repro.obj
 *
 * A function is BAD when its prologue shows `bl __chkstk` ahead of the
 * instruction that stores x30. v1 mirrors OpenSSL's tls_parse_all_extensions;
 * the others strip one ingredient each, to find which are required.
 */

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

/* v1: same shape as tls_parse_all_extensions - six parameters, a count read
 * through a two-level dereference of the first one, a loop of calls, then a
 * second loop calling through a table of function pointers. */
int v1( Conn *s, int context, void *exts, void *x, unsigned long long chainidx, int fin )
{
    unsigned long long i, numexts = 29;
    const Def *d;

    numexts += s->cert->meths_count;

    for ( i = 0; i < numexts; i++ )
        if ( !parse_one( s, i, context, exts, x, chainidx ) )
            return 0;

    if ( fin )
    {
        for ( i = 0, d = defs; i < 29; i++, d++ )
            if ( ( d->context & context ) != 0 && d->final != 0 && !d->final( s, context, 1 ) )
                return 0;
    }
    return 1;
}

/* v2: v1 without the function-pointer loop. */
int v2( Conn *s, int context, void *exts, void *x, unsigned long long chainidx, int fin )
{
    unsigned long long i, numexts = 29;

    numexts += s->cert->meths_count;
    ( void )fin;

    for ( i = 0; i < numexts; i++ )
        if ( !parse_one( s, i, context, exts, x, chainidx ) )
            return 0;
    return 1;
}

/* v3: v1 without the dereference that produces the loop bound, so no early-exit
 * test can be hoisted ahead of the register saves. */
int v3( Conn *s, int context, void *exts, void *x, unsigned long long chainidx, int fin )
{
    unsigned long long i;
    const Def *d;

    for ( i = 0; i < 29; i++ )
        if ( !parse_one( s, i, context, exts, x, chainidx ) )
            return 0;

    if ( fin )
    {
        for ( i = 0, d = defs; i < 29; i++, d++ )
            if ( ( d->context & context ) != 0 && d->final != 0 && !d->final( s, context, 1 ) )
                return 0;
    }
    return 1;
}

/* v4: v1 with fewer parameters. */
int v4( Conn *s, int context, int fin )
{
    unsigned long long i, numexts = 29;
    const Def *d;

    numexts += s->cert->meths_count;

    for ( i = 0; i < numexts; i++ )
        if ( !parse_one( s, i, context, 0, 0, 0 ) )
            return 0;

    if ( fin )
    {
        for ( i = 0, d = defs; i < 29; i++, d++ )
            if ( ( d->context & context ) != 0 && d->final != 0 && !d->final( s, context, 1 ) )
                return 0;
    }
    return 1;
}

/* v5: the early-exit alone - a count of zero skips everything, which is the test
 * `cbz x7` that /O2 hoists above the register saves in the real function. */
int v5( Conn *s, int context, void *exts, void *x, unsigned long long chainidx, int fin )
{
    unsigned long long i, numexts = s->cert->meths_count;
    const Def *d;

    if ( numexts == 0 )
        return 1;

    for ( i = 0; i < numexts; i++ )
        if ( !parse_one( s, i, context, exts, x, chainidx ) )
            return 0;

    if ( fin )
    {
        for ( i = 0, d = defs; i < 29; i++, d++ )
            if ( ( d->context & context ) != 0 && d->final != 0 && !d->final( s, context, 1 ) )
                return 0;
    }
    return 1;
}

/* v6: no calls at all, so nothing needs x30 saved. Control shape only. */
int v6( Conn *s, int context )
{
    unsigned long long i, numexts = 29, acc = 0;

    numexts += s->cert->meths_count;
    for ( i = 0; i < numexts; i++ )
        acc += ( unsigned long long )( defs[i % 29].context & context );
    return ( int )acc;
}
