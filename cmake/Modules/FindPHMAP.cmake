find_path(PHMAP_INCLUDE_DIR
	NAMES phmap.h
	PATH_SUFFIXES parallel_hashmap
)
IF(NOT PHMAP_INCLUDE_DIR)
	set(PHMAP_FOUND FALSE)
	return()
ENDIF()

add_library(phmap::phmap IMPORTED INTERFACE)
set_target_properties(phmap::phmap PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${PHMAP_INCLUDE_DIR}/.."
)
set(PHMAP_FOUND TRUE)
