#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "utf8proc::utf8proc" for configuration "Release"
set_property(TARGET utf8proc::utf8proc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(utf8proc::utf8proc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libutf8proc.so.3.2.3"
  IMPORTED_SONAME_RELEASE "libutf8proc.so.3"
  )

list(APPEND _cmake_import_check_targets utf8proc::utf8proc )
list(APPEND _cmake_import_check_files_for_utf8proc::utf8proc "${_IMPORT_PREFIX}/lib/libutf8proc.so.3.2.3" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
