#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "AWS::aws-c-s3" for configuration "Release"
set_property(TARGET AWS::aws-c-s3 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(AWS::aws-c-s3 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaws-c-s3.so.1.2.0"
  IMPORTED_SONAME_RELEASE "libaws-c-s3.so.1.2"
  )

list(APPEND _cmake_import_check_targets AWS::aws-c-s3 )
list(APPEND _cmake_import_check_files_for_AWS::aws-c-s3 "${_IMPORT_PREFIX}/lib/libaws-c-s3.so.1.2.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
