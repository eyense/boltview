#include <boost/optional.hpp>
#include <boost/program_options.hpp>
#include <boost/spirit/home/x3.hpp>
#include <boost/format.hpp>
#include <boost/fusion/adapted/std_tuple.hpp>
#include <boltview/math/vector.h>

// put the validate() function into bolt namespace so ADL finds it when validating Int2 and Float2 in boost::program_options
namespace bolt {

inline void parse(bolt::Float2& out, const std::string &s) {
	using namespace boost::program_options;
	namespace ascii = boost::spirit::x3::ascii;
	namespace x3 = boost::spirit::x3;
	using x3::int_;
	using x3::lit;
	using x3::char_;
	using x3::float_;
	using ascii::blank;

	std::cout << "Parse " << s << "\n";
	std::tuple<float, float> result;
	auto rule = lit('[') >> float_ >> lit(',') >> float_  >> lit(']');
	bool const res = x3::phrase_parse(s.begin(), s.end(), rule, blank, result);
	if (res) {
		out = bolt::Float2{ std::get<0>(result), std::get<1>(result) };
	} else {
		throw validation_error(validation_error::invalid_option_value);
	}
}

inline void parse(bolt::Int2& out, const std::string &s) {
	using namespace boost::program_options;
	namespace ascii = boost::spirit::x3::ascii;
	namespace x3 = boost::spirit::x3;
	using x3::int_;
	using x3::lit;
	using x3::char_;
	using x3::float_;
	using ascii::blank;

	std::tuple<int, int> result;
	auto rule = lit('[') >> int_ >> lit(',') >> int_  >> lit(']');
	bool const res = x3::phrase_parse(s.begin(), s.end(), rule, blank, result);
	if (res) {
		out = bolt::Int2{ std::get<0>(result), std::get<1>(result) };
	} else {
		throw validation_error(validation_error::invalid_option_value);
	}
}

template<typename TType>
void validate(boost::any& v, const std::vector<std::string>& values, TType* target_type, int)
// void validate(boost::any& v, const std::vector<std::string>& values, bolt::Int2* target_type, int)
{
	using namespace boost::program_options;
	namespace ascii = boost::spirit::x3::ascii;
	namespace x3 = boost::spirit::x3;

	// Make sure no previous assignment to 'a' was made.
	validators::check_first_occurrence(v);
	// Extract the first string from 'values'. If there is more than
	// one string, it's an error, and exception will be thrown.
	const std::string& s = validators::get_single_string(values);

	TType result;
	parse(result, s);
	v = boost::any(result);
}

}  // namespace bolt
