#include "TestFunctions.h"
#include <MRCMesh/MRCube.h>
#include <MRCMesh/MRMesh.h>

#define NSECS ( 1000 * 1000 * 1000 )

struct timespec timespec_now( void )
{
    struct timespec result;
    timespec_get( &result, TIME_UTC );
    return result;
}

struct timespec timespec_get_duration( const struct timespec* before, const struct timespec* after )
{
    struct timespec result;
    result.tv_sec = after->tv_sec - before->tv_sec;
    if ( before->tv_nsec <= after->tv_nsec )
    {
        result.tv_nsec = after->tv_nsec - before->tv_nsec;
    }
    else
    {
        result.tv_sec -= 1;
        result.tv_nsec = NSECS + after->tv_nsec - before->tv_nsec;
    }
    return result;
}

double timespec_to_seconds( const struct timespec* ts )
{
    return (double)ts->tv_sec + (double)ts->tv_nsec / NSECS;
}

MR_Mesh* createCube(void)
{
    MR_Vector3f size = MR_Vector3f_diagonal(1.0f);
    MR_Vector3f base = MR_Vector3f_diagonal(-0.5f);
    return MR_makeCube(&size, &base);
}
