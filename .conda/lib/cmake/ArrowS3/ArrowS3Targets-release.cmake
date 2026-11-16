#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ArrowS3::arrow_s3_shared" for configuration "Release"
set_property(TARGET ArrowS3::arrow_s3_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ArrowS3::arrow_s3_shared PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "aws-cpp-sdk-config;aws-cpp-sdk-transfer;aws-cpp-sdk-identity-management;aws-cpp-sdk-cognito-identity;aws-cpp-sdk-sts;aws-cpp-sdk-s3;aws-cpp-sdk-core"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libarrow_s3.so.2600.0.0"
  IMPORTED_SONAME_RELEASE "libarrow_s3.so.2600"
  )

list(APPEND _cmake_import_check_targets ArrowS3::arrow_s3_shared )
list(APPEND _cmake_import_check_files_for_ArrowS3::arrow_s3_shared "${_IMPORT_PREFIX}/lib/libarrow_s3.so.2600.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
