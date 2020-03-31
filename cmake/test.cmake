macro(bolt_test_suite TEST_ROOTDIR TEST_LIST LIB_LIST)
	foreach(TEST_TARGET ${${TEST_LIST}})
		message(STATUS "Adding test: ${TEST_TARGET}")
		if (USE_CLANG_GPU)
			# ".cu" files will be also compiled by clang
			set_source_files_properties(${TEST_ROOTDIR}/${TEST_TARGET}.cu PROPERTIES LANGUAGE CXX)
			add_executable(
				${TEST_TARGET}
				"${TEST_ROOTDIR}/${TEST_TARGET}.cu"
				)
			target_link_libraries(${TEST_TARGET} ${CUDA_LIBRARIES} bolt)
			set_target_properties(${TEST_TARGET} PROPERTIES LINKER_LANGUAGE CXX)
			set_target_properties(${TEST_TARGET} PROPERTIES COMPILE_FLAGS ${CLANG_GPU_CXX_FLAGS})
		else()
			add_executable(
				${TEST_TARGET}
				"${TEST_ROOTDIR}/${TEST_TARGET}.cu"
				${bolt_HEADERS}
				${bolt_TCC}
				)
		endif()
		target_link_libraries(${TEST_TARGET} ${Boost_LIBRARIES} ${${LIB_LIST}} bolt)
		add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})
	endforeach(TEST_TARGET ${TEST_LIST})
endmacro()


#TODO(johny) - turn on/off on demand
macro(target_clang_check check_target header_filter)
set_target_properties(
	${check_target} PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON
	CXX_CLANG_TIDY "clang-tidy;-header-filter=${header_filter};"
)
endmacro(target_clang_check)

