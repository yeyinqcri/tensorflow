/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_CLIQUE_KEY_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_CLIQUE_KEY_H_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/service/global_device_id.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

TSL_LIB_GTL_DEFINE_INT_TYPE(NcclStreamId, uint64_t);

// A standalone library without any dependencies on NCCL that allows us to
// include this header in all of XLA without worrying about NCCL availability.

//===----------------------------------------------------------------------===//
// AsyncStreamKind
//===----------------------------------------------------------------------===//

// We include a stream kind into the NCCL clique key because in XLA we do not
// share communicators for collective operations of different kind (CUDA-graph
// launched, async collectives, sync collectives) as it can lead to dead locks.
//
// We carefully isolate different kinds of collectives using separate
// communicators and guarantee that all collective operations have a total order
// that will not create a deadlock.
//
// See more details in `nccl_clique` library.

enum class AsyncStreamKind : int64_t {
  kCollective = 0,  // Stream for asynchronous collective ops.
  kP2P0 = 1,        // One Stream for P2P Send and Recv ops.
  kP2P1 = 2,        // Another Stream for P2P Send and Recv ops.
  kMemCpyP2P = 3,   // Stream for MemCpyP2P
};

constexpr static int64_t kAsyncStreamTotal =
    static_cast<int64_t>(AsyncStreamKind::kMemCpyP2P) + 1;

// Assigns a unique ID to a stream for asynchronous or synchronous execution.
// These IDs can be used, for example, to look up the NCCL communicator.
inline NcclStreamId GetStreamId(
    bool is_async, AsyncStreamKind stream_kind = AsyncStreamKind::kCollective) {
  return NcclStreamId(is_async ? static_cast<uint64_t>(stream_kind) + 1 : 0);
}

//===----------------------------------------------------------------------===//
// NcclCliqueKey
//===----------------------------------------------------------------------===//

// Key for naming up a particular NCCL clique. This is just a set of unique
// device IDs (i.e. GPU IDs) and a stream_id. The device IDs must be global
// within a cluster. The stream_id is used to create different NCCL clique and
// communicators for collectives executed on different streams within an
// executable.
class NcclCliqueKey : public CliqueKey {
 public:
  explicit NcclCliqueKey(
      std::vector<GlobalDeviceId> devices,
      NcclStreamId stream_id = NcclStreamId(0),
      AsyncStreamKind stream_kind = AsyncStreamKind::kCollective,
      std::vector<std::vector<GlobalDeviceId>> participant_groups = {});

  NcclStreamId stream_id() const;

  // Returns true if this clique is a subset of `other`: both cliques have the
  // same `stream_id` and all clique devices are part of `other` clique.
  bool IsSubsetOf(const CliqueKey& other) const final;

  // Returns the stream kind for this clique key,
  // stream kind will be used to specify what configuration
  // to pass for each type of operation.
  AsyncStreamKind stream_kind() const { return stream_kind_; }

  std::string ToString() const final;

  friend bool operator==(const NcclCliqueKey& a, const NcclCliqueKey& b);
  friend bool operator<(const NcclCliqueKey& a, const NcclCliqueKey& b);
  friend bool operator>(const NcclCliqueKey& a, const NcclCliqueKey& b);

 private:
  void HashValue(absl::HashState state) const final;

  NcclStreamId stream_id_;
  AsyncStreamKind stream_kind_;

  // The full list of groups across all devices which this clique is a part of.
  // When enable_nccl_comm_splitting is enabled, this is used to distinguish
  // which cliques can be reused from the cache or must be split in order to
  // prevent a deadlock situation.
  // For example, imagine we have a communicator with devices = [0,1] and groups
  // = [0, 1] Later on, we may want to create communicators [0, 1] and [2, 3] by
  // splitting [0, 1, 2, 3] If ranks 0 and 1 reuse the exisiting [0, 1] clique
  // but ranks 2 and 3 initiate a split, there will be a deadlock since ranks 2,
  // 3 and will be waiting forever for 0, 1 to join the split. Having the
  // particating groups as part of the cache key will prevent such situations
  std::vector<std::vector<GlobalDeviceId>> participant_groups_;
};

bool operator==(const NcclCliqueKey& a, const NcclCliqueKey& b);
bool operator<(const NcclCliqueKey& a, const NcclCliqueKey& b);

//===----------------------------------------------------------------------===//
// NcclCliqueId
//===----------------------------------------------------------------------===//

// A callback to get a unique clique id (see `ncclUniqueId` documentation).
using NcclCliqueIdCallback =  // NOLINT
    std::function<absl::StatusOr<CliqueId>(const NcclCliqueKey&)>;

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_CLIQUE_KEY_H_
