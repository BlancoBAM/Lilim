#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "thriftnb::thriftnb" for configuration "Release"
set_property(TARGET thriftnb::thriftnb APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(thriftnb::thriftnb PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libthriftnb.so.0.25.0"
  IMPORTED_SONAME_RELEASE "libthriftnb.so.0.25.0"
  )

list(APPEND _cmake_import_check_targets thriftnb::thriftnb )
list(APPEND _cmake_import_check_files_for_thriftnb::thriftnb "${_IMPORT_PREFIX}/lib/libthriftnb.so.0.25.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
