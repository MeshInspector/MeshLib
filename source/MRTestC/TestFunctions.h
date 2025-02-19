#pragma once

#include <time.h>

struct timespec timespec_now( void );

struct timespec timespec_get_duration( const struct timespec* before, const struct timespec* after );

double timespec_to_seconds( const struct timespec* ts );
