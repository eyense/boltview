// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.eu

#include "benchmarks.h"

#include <boltview/cuda_utils.h>

#include <utility>
#include <iostream>
#include <iomanip>
#include <string>

namespace bolt {

void Timer::Interval::stop(){
	if(running_){
		timer_->stop(id_, boost_timer_.elapsed());
		running_ = false;
	}
}


Timer::Interval::Interval(Timer *timer, const std::string &name, int id):
	timer_(timer), name_(name), id_(id), running_(true)
{}


Timer::Interval::~Interval(){
	stop();
}


Timer::Timer(const std::string &name):
	name_(name)
{}


Timer::Interval Timer::start(const std::string &name){
	boost::timer::cpu_times t;
	t.clear();
	intervals_.push_back(std::make_pair(name, t));
	return Timer::Interval(this, name, intervals_.size()-1);
}


void Timer::stop(
		const int id,
		const boost::timer::cpu_times &elapsed)
{
	intervals_[id].second = elapsed;
}


void Timer::printAll() {
	for(const auto &i : intervals_){
		std::cout << std::left << std::setw(30) << ('[' + name_ + ']');
		std::cout << std::setw(30) << i.first;
		std::cout << boost::timer::format(i.second);
	}
}

void Timer::printCSV(const std::string &group_name) {
	for(const auto &i : intervals_){
		std::cout << group_name << ";";
		std::cout << name_ << ";";
		std::cout << i.first << ";";
		std::cout << i.second.wall * 0.000000001 << "\n";
	}
}


void BenchmarkManager::add(const std::string &name, std::function<void(Timer&)> func){
	benchs_.push_back(std::make_pair(name, func));
}


void BenchmarkManager::runAll(){
	bolt::getDeviceMemoryInfo();
	for(auto b : benchs_){
		Timer t(b.first);
		b.second(t);
		timers_.push_back(t);
	}
}


void BenchmarkManager::printAll(){
	for(auto t : timers_){
		t.printAll();
		std::cout << '\n';
	}
}

void BenchmarkManager::printCSV(const std::string &group_name){
	for(auto t : timers_){
		t.printCSV(group_name);
		std::cout << '\n';
	}
}

}  // namespace bolt
