//! [custom-domain-header]
#pragma once

#include <MRViewer/MRI18n.h>

// forward declaration from MRLocale.h
namespace MR::Locale { MRVIEWER_API int findDomain( const char* domainName ); }

// replace "MyPlugin" with your actual domain name
// (must match the .pot/.po filename stem)
inline constexpr const char* MY_PLUGIN_I18N_DOMAIN = "MyPlugin";

// redefine the i18n macros to use your domain by default
#undef _tr
#undef s_tr
#undef f_tr
#define _tr( ... ) MR::Locale::translate( __VA_ARGS__, MR::Locale::Domain{ MR::Locale::findDomain( MY_PLUGIN_I18N_DOMAIN ) } ).c_str()
#define s_tr( ... ) MR::Locale::translate( __VA_ARGS__, MR::Locale::Domain{ MR::Locale::findDomain( MY_PLUGIN_I18N_DOMAIN ) } )
#define f_tr( ... ) fmt::runtime( MR::Locale::translate( __VA_ARGS__, MR::Locale::Domain{ MR::Locale::findDomain( MY_PLUGIN_I18N_DOMAIN ) } ) )
//! [custom-domain-header]
