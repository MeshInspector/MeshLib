#pragma once

#include "config.h"
#ifndef MRVIEWER_NO_LOCALE
#include "exports.h"

#include <MRMesh/MRMeshFwd.h>

#include <string>

namespace MR::Locale
{

/// \brief Translates a message using the active locale.
MRVIEWER_API std::string translate( const char* msg, int domainId = 0 );

/// \brief Translates a message in context using the active locale.
MRVIEWER_API std::string translate( const char* context, const char* msg, int domainId = 0 );

/// \brief Translates a plural message form using the active locale.
MRVIEWER_API std::string translate( const char* single, const char* plural, Int64 n, int domainId = 0 );

/// \brief Translates a plural message form in context using the active locale.
MRVIEWER_API std::string translate( const char* context, const char* single, const char* plural, Int64 n, int domainId = 0 );

} // namespace MR::Locale

#ifndef MR_NO_I18N_MACROS
#define _tr( ... ) MR::Locale::translate( __VA_ARGS__ ).c_str()
#define f_tr( ... ) fmt::runtime( MR::Locale::translate( __VA_ARGS__ ) )
#endif // MR_NO_I18N_MACROS

#else // MRVIEWER_NO_LOCALE

#ifndef MR_NO_I18N_MACROS
#define _tr( ... ) MR::Locale::translate_noop( __VA_ARGS__ )
#define f_tr( ... ) fmt::runtime( MR::Locale::translate_noop( __VA_ARGS__ ) )
#endif // MR_NO_I18N_MACROS

#endif // MRVIEWER_NO_LOCALE
