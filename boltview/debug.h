#pragma once

#include <iostream>
#include <boost/format.hpp>
#include <signal.h>

// NOTE(fidli): add this line to breakpoints.gdb, works both for gdb and cuda-gdb
#define BOLT_BREAK_DEF
// NOTE(fidli): following use only during active debugging, for host
// NOTE(fidli): break during runtime - every time
#define BOLT_BREAK_HOST_RT raise(SIGINT)

namespace detail {
/// ends recursion
inline void formatHelper(boost::format &a_format) {}

/// Static recursion for format filling
template <typename TT, typename... TArgs>
void formatHelper(boost::format &a_format, TT &&a_value, TArgs &&...a_args) {
	a_format % a_value;
	formatHelper(a_format, std::forward<TArgs>(a_args)...);
}

}  // namespace detail

	#define D_FORMAT(format_string, ...) /* NOLINT */ \
	do { \
                boost::format format(format_string); \
                ::detail::formatHelper(format, ##__VA_ARGS__); \
                std::cout << __FILE__ << ":" << __LINE__ << ":" \
                        << format << std::endl; \
        } while (0)


