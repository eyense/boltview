# Try to find Eyen's BoltView library

# Once done this will define
#  BOLT_FOUND - system has found the ecip library with correct include directory
#  BOLT_INCLUDE_DIR - the ecip include directory

# When searching for BoltView, this module respects the following variables (both CMake and environment)
#  BOLT_ROOT - the library is searched for as ${BOLT_ROOT}/boltview/device_image.h
#  EYEN_ROOT - If search in BOLT_ROOT is not successfull, EYEN_ROOT is tried out and BoltView is searched as ${BOLT_ROOT}/boltview/device_image.h

if (NOT BOLT_FOUND)
	find_path(BOLT_INCLUDE_DIR NAMES boltview/device_image.h
		PATHS
		${BOLT_ROOT}
		$ENV{BOLT_ROOT}
		${EYEN_ROOT}
		$ENV{EYEN_ROOT}
		../
		../../
		../../../
		PATH_SUFFIXES include
		)

	include(FindPackageHandleStandardArgs)
	find_package_handle_standard_args(BoltView DEFAULT_MSG BOLT_INCLUDE_DIR)

    set(BOLT_FOUND TRUE)
	mark_as_advanced(BOLT_INCLUDE_DIR)

	add_library(bolt INTERFACE)
	target_include_directories(bolt INTERFACE ${BOLT_INCLUDE_DIR})
	target_compile_features(bolt INTERFACE cxx_std_14)

endif()

