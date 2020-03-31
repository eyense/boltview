// Copyright 2019 Eyen SE
// Author: Martin Hora martin.hora@eyen.se

#pragma once

namespace bolt {

/// Standard policy using 32-bit integers to index images/views.
class DefaultViewPolicy {
public:
  using IndexType = int;
};

/// Policy using 64-bit integers to index images/views.
class LongIndexViewPolicy {
public:
  using IndexType = int64_t;
};

}   // namespace bolt
