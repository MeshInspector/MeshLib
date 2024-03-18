find_path(HPDF_INCLUDE_DIR
	NAMES hpdf_version.h
)
IF(NOT HPDF_INCLUDE_DIR)
	set(HPDF_FOUND FALSE)
	return()
ENDIF()

add_library(hpdf::hpdf IMPORTED INTERFACE)
set_target_properties(hpdf::hpdf PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${HPDF_INCLUDE_DIR}"
)
set(HPDF_FOUND TRUE)
