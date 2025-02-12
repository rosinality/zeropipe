import dataclasses
from dataclasses import dataclass
from enum import Enum
import functools


class FnType(Enum):
    F = "F"
    B = "B"
    W = "W"
    BW = "BW"

    SEND_FORWARD = "SEND_FORWARD"
    SEND_BACKWARD = "SEND_BACKWARD"
    RECV_FORWARD = "RECV_FORWARD"
    RECV_BACKWARD = "RECV_BACKWARD"

    POST_VALIDATION = "POST_VALIDATION"
    SEND_POST_VALIDATION = "SEND_POST_VALIDATION"
    RECV_POST_VALIDATION = "RECV_POST_VALIDATION"

    def __repr__(self):
        return self.value

    def is_compute(self):
        return self in {FnType.F, FnType.B, FnType.W, FnType.BW}

    def is_comm(self):
        return self in {
            FnType.SEND_FORWARD,
            FnType.SEND_BACKWARD,
            FnType.RECV_FORWARD,
            FnType.RECV_BACKWARD,
        }

    def is_backward_comm(self):
        return self in {
            FnType.SEND_BACKWARD,
            FnType.RECV_BACKWARD,
        }

    def is_post_validation(self):
        return self in {
            FnType.POST_VALIDATION,
            FnType.SEND_POST_VALIDATION,
            FnType.RECV_POST_VALIDATION,
        }

    def is_send(self):
        return self in {
            FnType.SEND_FORWARD,
            FnType.SEND_BACKWARD,
            FnType.SEND_POST_VALIDATION,
        }

    def is_recv(self):
        return self in {
            FnType.RECV_FORWARD,
            FnType.RECV_BACKWARD,
            FnType.RECV_POST_VALIDATION,
        }

    def peer_type(self):
        pairs = dict(
            [
                (FnType.SEND_FORWARD, FnType.RECV_FORWARD),
                (FnType.SEND_BACKWARD, FnType.RECV_BACKWARD),
                (FnType.SEND_POST_VALIDATION, FnType.RECV_POST_VALIDATION),
            ]
        )

        pairs.update({v: k for k, v in pairs.items()})

        return pairs[self]


class Direction(Enum):
    NEXT = 0
    PREV = 1


@dataclass(eq=True, frozen=True)
class NodeKey:
    type: FnType
    layer_group_index: int
    microbatch: int
    seq_split_index: int

    def __post_init__(self):
        assert isinstance(self.type, FnType)

    def __hash__(self):
        return hash(
            (self.type, self.layer_group_index, self.microbatch, self.seq_split_index)
        )


@dataclass(eq=True)
class ScheduleNode:
    type: FnType
    stage: int
    microbatch: int
    chunk: int = 0
    seq_split_index: int = 0

    layer_group_index: int | None = None
    start_time: int | None = None
    complete_time: int | None = None

    recv_peer_stage: int | None = None
    send_peer_stage: int | None = None

    comm_direction: Direction | None = None
    comm_peer_stage: int | None = None
    comm_pair_id: int | None = None

    rollback: bool = False

    def __post_init__(self):
        assert isinstance(self.type, FnType)

    def __hash__(self):
        return hash(self.key)

    @property
    def key(self):
        return NodeKey(
            type=self.type,
            layer_group_index=self.layer_group_index,
            microbatch=self.microbatch,
            seq_split_index=self.seq_split_index,
        )

    def get_prev_key(self, n_layer_groups: int):
        assert self.layer_group_index is not None

        if self.type == FnType.F:
            if self.layer_group_index == 0:
                return None

            prev_layer_group_index = self.layer_group_index - 1

            return NodeKey(
                self.type, prev_layer_group_index, self.microbatch, self.seq_split_index
            )

        if self.type in {FnType.B, FnType.BW}:
            prev_layer_group_index = self.layer_group_index + 1

            assert prev_layer_group_index <= n_layer_groups

            if prev_layer_group_index == n_layer_groups:
                return NodeKey(
                    FnType.F,
                    self.layer_group_index,
                    self.microbatch,
                    self.seq_split_index,
                )

            return NodeKey(
                self.type, prev_layer_group_index, self.microbatch, self.seq_split_index
            )

        assert self.type == FnType.W

        return NodeKey(
            FnType.B, self.layer_group_index, self.microbatch, self.seq_split_index
        )

    def get_activation_key(self):
        return self.microbatch, self.chunk, self.seq_split_index


@dataclass(eq=True, frozen=True)
class CommunicationPair:
    send_node: ScheduleNode
    recv_node: ScheduleNode
    recv_deadline: ScheduleNode


@dataclass
class ScheduleConfig:
    cost_forward: list[float] | None = None
    cost_backward: list[float] | None = None
    cost_weight: list[float] | None = None
    cost_comm: float = 0.0

    mem_forward: list[int] | None = None
    mem_backward: list[int] | None = None
    mem_weight: list[int] | None = None

    max_mem: list[float] | None = None

    max_chunks: int = 1
    n_stages: int | None = None
    n_microbatches: int | None = None

    @property
    def n_layer_groups(self):
        return self.n_stages * self.max_chunks


def finalize_schedule(
    config: ScheduleConfig, schedule: list[list[ScheduleNode]], validate=True
):
    schedule = _add_send_recv(config, schedule)
    schedule = _add_time(config, schedule)
    schedule = _add_communication_nodes(config, schedule)

    if validate:
        _validate_communication(schedule)

    return schedule


def _add_send_recv(config: ScheduleConfig, schedule: list[list[ScheduleNode]]):
    nodes = sum(schedule, [])
    node_map = {node.key: node for node in nodes}

    for node in nodes:
        prev_key = node.get_prev_key(config.n_layer_groups)

        if prev_key is None:
            continue

        if prev_key not in node_map:
            raise ValueError(f"Previous node {prev_key} for {node.key} not found")

        prev_node = node_map[prev_key]

        if prev_node.stage == node.stage:
            continue

        prev_node.send_peer_stage = node.stage
        node.recv_peer_stage = prev_node.stage

    return schedule


def _add_time(config: ScheduleConfig, schedule: list[list[ScheduleNode]]):
    if schedule[0][0].start_time is not None:
        return schedule

    nodes = sum(schedule, [])
    node_map = {node.key: node for node in nodes}

    type_cost = {
        FnType.F: config.cost_forward,
        FnType.B: config.cost_backward,
        FnType.W: config.cost_weight,
        FnType.BW: config.cost_backward + config.cost_weight,
    }

    new_schedule = [[] for _ in schedule]
    completion_time = {}
    stage_current_t = [0.0 for _ in schedule]
    stage_current_index = [0 for _ in schedule]

    index = 0

    while index < len(nodes):
        found = False
        pending = []

        for stage in range(len(schedule)):
            if stage_current_index[stage] >= len(schedule[stage]):
                continue

            node = schedule[stage][stage_current_index[stage]]
            prev_compute_node = node.get_prev_key(config.n_layer_groups)

            if (
                prev_compute_node is not None
                and prev_compute_node not in completion_time
            ):
                pending.append((node.key, prev_compute_node))

                continue

            stage_current_index[stage] += 1
            t = stage_current_t[stage]

            if prev_compute_node is not None:
                prev_t = completion_time[prev_compute_node]

                if node_map[prev_compute_node].stage != node.stage:
                    prev_t += config.cost_comm

                t = max(prev_t, t)

            compute_t = type_cost[node.type][node.stage]
            end_t = t + compute_t
            completion_time[node.key] = end_t
            stage_current_t[node.stage] = end_t

            new_schedule[node.stage].append(
                dataclasses.replace(
                    dataclasses.replace(node, complete_time=end_t), start_time=t
                )
            )

            found = True
            index += 1

        if not found:
            error_msg = [
                f"stage_current_index: {stage_current_index} completed {completion_time}"
            ]

            for node, prev in pending:
                error_msg.append(f"pending {node} {prev}")

            error_msg = "\n".join(error_msg)

            raise RuntimeError(f"Cannot find next runnable node\n{error_msg}")

    assert len(new_schedule) == len(schedule)

    for new, prev in zip(new_schedule, schedule):
        assert len(new) == len(prev)

    return new_schedule


def _add_communication_nodes(
    config: ScheduleConfig, schedule: list[list[ScheduleNode]]
):
    schedule, comm_pairs = _insert_send_nodes(config, schedule)
    schedule = _insert_recv_nodes(config, schedule, comm_pairs)

    return schedule


def _create_communication_node(
    node,
    category,
    send_recv,
    compute_node,
    comm_peer_stage,
    t,
    comm_direction,
    comm_pair_id,
):
    return ScheduleNode(
        type=FnType(send_recv + "_" + category),
        chunk=compute_node.chunk,
        stage=compute_node.stage,
        microbatch=node.microbatch,
        seq_split_index=node.seq_split_index,
        layer_group_index=compute_node.layer_group_index,
        start_time=t,
        complete_time=t,
        comm_direction=comm_direction,
        comm_peer_stage=comm_peer_stage,
        comm_pair_id=comm_pair_id,
    )


def _insert_send_nodes(config: ScheduleConfig, schedule: list[list[ScheduleNode]]):
    assert len(schedule) == config.n_stages

    node_map = {node.key: node for node in sum(schedule, [])}
    comm_pair_id = 0
    new_schedule = [[node for node in nodes] for nodes in schedule]

    comm_pairs = []

    for stage in range(config.n_stages):
        for node in schedule[stage]:
            assert stage == node.stage, f"invalid node stage {stage} {node}"

            if node.type not in {FnType.F, FnType.B, FnType.BW}:
                continue

            category_str = "FORWARD" if node.type == FnType.F else "BACKWARD"

            if node.recv_peer_stage is None or node.recv_peer_stage == node.stage:
                pass

            else:
                if node.recv_peer_stage + 1 == node.stage or (
                    node.stage == 0 and node.recv_peer_stage == config.n_stages - 1
                ):
                    send_direction = Direction.NEXT
                    recv_direction = Direction.PREV

                else:
                    assert node.recv_peer_stage == node.stage + 1 or (
                        node.recv_peer_stage == 0 and node.stage == config.n_stages - 1
                    ), f"invalid send-recv stages {node.recv_peer_stage} {node.stage}"

                    send_direction = Direction.PREV
                    recv_direction = Direction.NEXT

                peer = node_map[node.get_prev_key(config.n_layer_groups)]
                assert peer.stage == node.recv_peer_stage
                send_node = _create_communication_node(
                    node,
                    category_str,
                    "SEND",
                    peer,
                    stage,
                    peer.complete_time,
                    send_direction,
                    comm_pair_id,
                )
                recv_node = _create_communication_node(
                    node,
                    category_str,
                    "RECV",
                    node,
                    peer.stage,
                    peer.complete_time,
                    recv_direction,
                    comm_pair_id,
                )
                comm_pairs.append(
                    CommunicationPair(send_node, recv_node, recv_deadline=node)
                )
                comm_pair_id += 1

                send_stage_nodes = new_schedule[send_node.stage]
                send_compute_pos = next(
                    (i for i, n in enumerate(send_stage_nodes) if n.key == peer.key)
                )
                insert_pos = send_compute_pos + 1

                insert_pos = next(
                    (
                        i + insert_pos
                        for i, n in enumerate(send_stage_nodes[insert_pos:])
                        if n.start_time >= send_node.start_time
                    ),
                    len(send_stage_nodes),
                )
                send_stage_nodes.insert(insert_pos, send_node)

    return new_schedule, comm_pairs


def _cmp_pair(send_index_map, a: CommunicationPair, b: CommunicationPair):
    sa, sb = a.send_node, b.send_node

    if sa.stage == sb.stage:
        return send_index_map[sa.key] - send_index_map[sb.key]

    elif sa.start_time == sb.start_time:
        a_type_rank = int(sa.type != FnType.SEND_POST_VALIDATION)
        b_type_rank = int(sb.type != FnType.SEND_POST_VALIDATION)

        return a_type_rank - b_type_rank

    return sa.start_time - sb.start_time


def _insert_recv_nodes(
    config: ScheduleConfig,
    schedule: list[list[ScheduleNode]],
    comm_pairs: list[CommunicationPair],
):
    send_index_map = {
        p.send_node.key: schedule[p.send_node.stage].index(p.send_node)
        for p in comm_pairs
    }

    comm_pairs.sort(
        key=functools.cmp_to_key(functools.partial(_cmp_pair, send_index_map))
    )
    start_indices = [0 for _ in schedule]
    new_schedule = schedule

    for pair in comm_pairs:
        send_node = pair.send_node
        recv_node = pair.recv_node
        recv_deadline = pair.recv_deadline

        send_stage_nodes = new_schedule[send_node.stage]
        start = start_indices[send_node.stage]
        send_pos = next(
            (
                i + start
                for i, n in enumerate(send_stage_nodes[start:])
                if n.key == send_node.key
            )
        )
        assert send_pos >= start
        start_indices[send_node.stage] = send_pos + 1

        recv_stage_nodes = new_schedule[recv_node.stage]
        start = start_indices[recv_node.stage]
        recv_pos = next(
            (
                i + start
                for i, n in enumerate(recv_stage_nodes[start:])
                if (
                    recv_node.start_time <= n.complete_time
                    or n.key == recv_deadline.key
                    or n.type.is_send()
                )
            )
        )
        recv_stage_nodes.insert(recv_pos, recv_node)
        start_indices[recv_node.stage] = recv_pos + 1

    assert len(new_schedule) == len(schedule)

    return new_schedule


def _validate_communication(schedule: list[list[ScheduleNode]]):
    fused_comm = []
    n_comm = 0

    for nodes in schedule:
        comms = []
        curr_comm = set()

        for node in nodes:
            if node.type.is_send() or node.type.is_recv():
                assert node not in curr_comm
                curr_comm.add(node)

                continue

            if curr_comm:
                comms.append(curr_comm)
                n_comm += len(curr_comm)
                curr_comm = set()

        if curr_comm:
            comms.append(curr_comm)
            n_comm += len(curr_comm)

        fused_comm.append(comms)

    assert len(fused_comm) == len(schedule)

    stage_current_index = [0 for _ in fused_comm]
    index = 0
    last_found = 0

    while index < n_comm:
        found = False
        pending_comm = {}

        for stage in range(len(fused_comm)):
            if stage_current_index[stage] >= len(fused_comm[stage]):
                continue

            curr_fused_nodes = list(fused_comm[stage][stage_current_index[stage]])

            for node in curr_fused_nodes:
                assert node.stage == stage, f"stage: {stage} node: {node}"
                assert node.comm_peer_stage is not None

                peer_key = node.comm_pair_id
                if peer_key not in pending_comm:
                    node_key = node.comm_pair_id
                    pending_comm[node_key] = node
                    last_found = False

                    continue

                found = True
                last_found = True
                index += 2
                peer = pending_comm.pop(peer_key)

                for n in (node, peer):
                    fused = fused_comm[n.stage][stage_current_index[n.stage]]
                    fused.remove(n)

                    if not fused:
                        stage_current_index[n.stage] += 1

        if not found:
            raise RuntimeError(f"Cannot find next node, pending: {pending_comm}")
