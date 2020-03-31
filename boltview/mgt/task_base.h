// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <boltview/mgt/task_interface.h>

namespace bolt {

namespace mgt {

/// \addtogroup mgt Multi GPU Scheduling
/// @{

class BaseTask : public ITask {
public:
	BaseTask(GroupId task_group_id, int priority) :
		task_group_id_(task_group_id),
		priority_(priority) {}

	virtual ~BaseTask() = default;

protected:
	GroupId getGroupId() const {
		return task_group_id_;
	}

	int getPriority() const {
		return priority_;
	}

private:
	/// Tasks with the same task_group_id should ideally run on the same gpu stream
	/// presumably beacause of shared input or output.
	GroupId task_group_id_;
	int priority_;
};

class BaseProblem : public IProblem {
public:
	explicit BaseProblem(ProblemId problem_id) : problem_id_(problem_id) {}
	virtual ~BaseProblem() = default;

protected:
	ProblemId getProblemId() const {
		return problem_id_;
	}

private:
	ProblemId problem_id_;
};

/// @}

}  // namespace mgt

}  // namespace bolt
