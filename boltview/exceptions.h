// Copyright 2016 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <exception>
#include <string>
#include <boost/exception/all.hpp>
#include <boost/filesystem.hpp>

namespace bolt {

/// Base class for all eyen exceptions.
class ExceptionBase: public virtual boost::exception, public virtual std::exception {
public:
	const char* what() const noexcept override {
		return boost::diagnostic_information(*this).c_str();
	}
};

/// \addtogroup Utilities
/// @{

/// \addtogroup ExceptionErrorInfo
/// @{
//
/// Error info containing string. Can be used for passing messages together with thrown exceptions (see examples).
using MessageErrorInfo = boost::error_info<struct tag_message, std::string>;

/// Error info containing file path.
using FilenameErrorInfo = boost::error_info<struct tag_filename, boost::filesystem::path>;


// TODO(johny): the exceptions specific to given class should be moved to the respective class header


/// @}

/// \addtogroup Exceptions
/// @{
struct BoltError: ExceptionBase {};

struct CudaError: BoltError {};

struct IncompatibleViewSizes: CudaError {};

struct ContiguousMemoryNeeded: CudaError {};

struct SliceOutOfRange: BoltError {};

struct InvalidNDRange: BoltError {};

struct NotYetImplemented: BoltError {};

#define BOLT_THROW(x) /*NOLINT*/ BOOST_THROW_EXCEPTION(x)

/// @}
/// @}

}  // namespace bolt
