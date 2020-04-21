if(BOLT_USE_THREAD_BUILDING_BLOCKS)
	set(TBB tbb)
endif()

# scheduler test
add_executable(mgt_test
	scheduler_test_problems.cu
	scheduler_test.cu
	)

target_link_libraries(mgt_test
	boost_system
	bolt
	${CUFFT_LIB}
	${TBB}
	)

# memory cache test

add_executable(memory_cache_test
	# device_code.cu
	memory_cache_test.cu
	)

target_link_libraries(memory_cache_test
	boost_system
	bolt
	${CUFFT_LIB}
	${TBB}
	)

add_test(mgt_test mgt_test)
add_test(memory_cache_test memory_cache_test)

