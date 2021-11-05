#pragma once

#include <iostream>
#include <boost/format.hpp>
#include <signal.h>

// NOTE(fidli): add this line to breakpoints.gdb
#define BOLT_BREAK
// NOTE(fidli): following use only during active debugging
// NOTE(fidli): break during runtime - every time
#define BOLT_BREAK_RT raise(SIGINT)
// NOTE(fidli): following can be problematic with multiple DLLs, would need extern then and exact definition in one DLL that is loaded by all
// NOTE(fidli): break during runtime - only once
#define BOLT_BREAK_RT_ONCE \
{ \
    static bool __debug_hit = false; \
    if(__debug_hit) \
    { \
        BOLT_BREAK_RT; \
    } \
    __debug_hit = true; \
}
// NOTE(fidli): break always during runtime after nth hit
#define BOLT_BREAK_RT_AFTER(n) \
{ \
    static int __debug_hit_counter = (n); \
    if(__debug_hit_counter == 0) \
    { \
        BOLT_BREAK_RT; \
    }else \
    { \
        __debug_hit_counter--; \
    } \
}
// NOTE(fidli): break once during runtime at nth hit
#define BOLT_BREAK_RT_AT(n) \
{ \
    static int __debug_hit_counter = (n); \
    if(__debug_hit_counter == 0) \
    { \
        BOLT_BREAK_RT; \
        __debug_hit_counter--; \
    }else if(__debug_hit_counter > 0)\
    { \
        __debug_hit_counter--; \
    } \
}

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


