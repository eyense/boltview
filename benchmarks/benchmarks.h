// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.eu

#pragma once

#include <boost/timer/timer.hpp>

#include <string>
#include <map>
#include <vector>
#include <utility>
#include <functional>

/// Benchmark manager and its helper classes

namespace bolt {

/// Measure time and hold intervals' info
class Timer {
public:
	class Interval;

	/// \param name Timer name
	explicit Timer(const std::string &name);

	/// Start interval, return interval object which add interval info
	/// back to Timer when stopped or destroyed
	/// \param interval_name Interval name
	Timer::Interval start(const std::string &interval_name);

	/// Print all intervals
	void printAll();
	void printCSV(const std::string &group_name);

	class Interval {
	public:
		/// Stop interval and save result to the Timer
		void stop();
		~Interval();

	private:
		friend class Timer;

		/// \param timer Parent Timer
		/// \param name Interval name
		/// \param id If of the interval
		Interval(Timer *timer, const std::string &name, int id);

		Timer *timer_;
		std::string name_;
		bool running_;
		boost::timer::cpu_timer boost_timer_;
		int id_;
	};

private:
	/// Stop interval
	/// \param id Interval to stop
	/// \paramn elapsed Elapsed time
	void stop(
		const int interval_id,
		const boost::timer::cpu_times &elapsed);

	std::string name_;
	using StringTime = std::pair<std::string, boost::timer::cpu_times>;
	std::vector<StringTime> intervals_;
};

/// Manages benchmarks
class BenchmarkManager{
public:
	BenchmarkManager()
	 {}

	 /// Add benchmark
	 /// \param name Benchmark name
	 /// \param func void<Timer&> function to be run, eg. [](Timer &t){...}
	void add(const std::string &name, std::function<void(Timer&)> func);
	void runAll();

	/// Print results of all benchmarks
	void printAll();
	void printCSV(const std::string &group_name);

 private:
	typedef std::pair<std::string, std::function<void(Timer&)>> str_func;
	std::vector<str_func> benchs_;
	std::vector<Timer> timers_;
};


}  // namespace bolt
