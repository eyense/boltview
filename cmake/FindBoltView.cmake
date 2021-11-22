MESSAGE(WARNING "FindBolt is out of date, it might not work")
# Try to find Eyen's ECIP library

# Once done this will define
#  ECIP_FOUND - system has found the ecip library with correct include directory
#  ECIP_INCLUDE_DIR - the ecip include directory
#  ECIP_LIBRARIES - ecip static library

# When searching for ECIP, this module respects the following variables (both CMake and environment)
#  ECIP_ROOT - the library is searched for as ${ECIP_ROOT}/ecip/image.h
#  EYEN_ROOT - If search in ECIP_ROOT is not successfull, EYEN_ROOT is tried out and ECIP is searched as ${EYEN_ROOT}/lib/ecip/ecip/image.h


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
		PATH_SUFFIXES boltview include/boltview
		)

	include(FindPackageHandleStandardArgs)
	find_package_handle_standard_args(BoltView DEFAULT_MSG BOLT_INCLUDE_DIR)

	mark_as_advanced(BOLT_INCLUDE_DIR)

	add_library(bolt INTERFACE)
	target_include_directories(bolt INTERFACE ${BOLT_INCLUDE_DIR})
	target_compile_features(bolt INTERFACE cxx_std_14)

endif()

