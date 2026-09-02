typedef struct Cert { char pad[0x88]; unsigned long long meths_count; } Cert;
typedef struct Conn { char pad[0x878]; Cert *cert; } Conn;
typedef struct Def { int context; int ( *final )( Conn *, int, int ); } Def;
extern const Def defs[29];
extern int parse_one( Conn *s, unsigned long long i, int ctx, void *exts, void *x, unsigned long long idx );
int repro( Conn *s, int context, void *exts, void *x, unsigned long long chainidx, int fin )
{
    unsigned long long i;
    unsigned long long numexts = 29 + s->cert->meths_count;
    unsigned long long a = ( unsigned long long )exts;
    unsigned long long b = ( unsigned long long )x;
    unsigned long long c = chainidx;
    unsigned long long d = ( unsigned long long )context;
    unsigned long long e = ( unsigned long long )fin;
    unsigned long long f = numexts;
    unsigned long long g = 0;
    unsigned long long h = 0;
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
