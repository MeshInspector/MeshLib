#pragma once

#include <time.h>

// returns a timespec object holding the current time
struct timespec timespec_now( void );

// returns a timespec object holding a duration between two time values
struct timespec timespec_get_duration( const struct timespec* before, const struct timespec* after );

// converts a timespec object to a timestamp in seconds
double timespec_to_seconds( const struct timespec* ts );
