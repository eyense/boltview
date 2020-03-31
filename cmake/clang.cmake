macro(detect_clang_gpu)
  if (NOT DEFINED USE_CLANG_GPU)
    # use regular expression to match Clang and AppleClang
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(USE_CLANG_GPU ON CACHE BOOL "Compile CUDA with clang.")
    endif()
  endif()
endmacro()
