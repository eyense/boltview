// Copyright 2016 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once


namespace bolt {

struct MovedFromFlag {
	MovedFromFlag() :
		moved_from_(false)
	{}

	MovedFromFlag(MovedFromFlag &&other) :
		moved_from_(other.moved_from_)
	{
		other.moved_from_ = true;
	}

	MovedFromFlag &operator=(MovedFromFlag &&other) {
		moved_from_ = other.moved_from_;
		other.moved_from_ = true;
		return *this;
	}

	MovedFromFlag(const MovedFromFlag &other) = delete;
	MovedFromFlag &operator=(const MovedFromFlag &other) = delete;

	bool isMoved() const {
		return moved_from_;
	}

	bool isValid() const {
		return !moved_from_;
	}

	bool moved_from_;
};


}  // namespace bolt
