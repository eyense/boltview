// Copyright 2019 Eyen SE
// Author: Adam Kubišta adam.kubista@eyen.se
#pragma once

#define BOLT_TEST_SKIP BOLT_TEST_SKIP

#if BOOST_VERSION<=103200
#define BOLT_AUTO_TEST_CASE_BOLT_TEST_RUN(testName) BOOST_AUTO_UNIT_TEST(testName)
#define BOLT_FIXTURE_TEST_CASE_BOLT_TEST_RUN(testName, fixture) BOOST_FIXTURE_UNIT_TEST(testName, fixture)
#else
#define BOLT_AUTO_TEST_CASE_BOLT_TEST_RUN(testName) BOOST_AUTO_TEST_CASE(testName)
#define BOLT_FIXTURE_TEST_CASE_BOLT_TEST_RUN(testName, fixture) BOOST_FIXTURE_TEST_CASE(testName, fixture)
#endif

#define DO_PRAGMA(x) _Pragma (#x)
#define BOLT_CONCAT(A, B) A ## B

// TODO(fidli): Runtime print? to make it even more noticeable
// NOTE(fidli): if all tests are disabled, it produces a failure of empty test tree, therefore insert dummy test to avoid such failures
#define BOLT_AUTO_TEST_CASE_ECIT_TEST_SKIP(testName, ...) \
	BOLT_AUTO_TEST_CASE_BOLT_TEST_RUN(BOLT_CONCAT(testName, _Disabled)) \
	{ \
		DO_PRAGMA(message ("Skipping test - " #testName)) \
	} \
	 \
	void testName() \


// NOTE(fidli): if all tests are disabled, it produces a failure of empty test tree, therefore insert dummy test to avoid such failures
#define BOLT_FIXTURE_TEST_CASE_ECIT_TEST_SKIP(testName, fixture, ...) \
	BOLT_AUTO_TEST_CASE_BOLT_TEST_RUN(BOLT_CONCAT(testName, _Disabled)) \
	{ \
		DO_PRAGMA(message ("Skipping test - " #testName)) \
	} \
	 \
	struct BOLT_CONCAT(fixture, testName) : public fixture \
	{ \
		void testName(); \
	}; \
	void BOLT_CONCAT(fixture, testName)::testName() \


#define BOLT_GET_1ST_ARG(arg1, ...) arg1
#define BOLT_GET_2ND_ARG(arg1, arg2, ...) arg2
#define BOLT_GET_3RD_ARG(arg1, arg2, arg3, ...) arg3
#define BOLT_GET_4TH_ARG(arg1, arg2, arg3, arg4, ...) arg4

#define BOLT_AUTO_TEST_CASE_CHOOSER(...) BOLT_GET_3RD_ARG(__VA_ARGS__, BOLT_AUTO_TEST_CASE_ECIT_TEST_SKIP, BOLT_AUTO_TEST_CASE_BOLT_TEST_RUN, )
#define BOLT_FIXTURE_TEST_CASE_CHOOSER(...) BOLT_GET_4TH_ARG(__VA_ARGS__, BOLT_FIXTURE_TEST_CASE_ECIT_TEST_SKIP, BOLT_FIXTURE_TEST_CASE_BOLT_TEST_RUN, )

#define BOLT_AUTO_TEST_CASE(...) BOLT_AUTO_TEST_CASE_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define BOLT_FIXTURE_TEST_CASE(...) BOLT_FIXTURE_TEST_CASE_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
