from collections import deque

from zeropipe.schedules.schedule import FnType, ScheduleConfig, ScheduleNode, NodeChunk


class ZeroBubbleVSchedule:
    def __init__(self, config: ScheduleConfig):
        self.n_nodes = 6 * config.n_stages * config.n_microbatches
        self.n_stages = config.n_stages
        self.n_microbatches = config.n_microbatches

        self.cost_forward = config.cost_forward[0]
        self.cost_backward = config.cost_backward[0]
        self.cost_weight = config.cost_weight[0]
        self.cost_comm = config.cost_comm

        self.mem_forward = config.mem_forward[0]
        self.mem_backward = config.mem_backward[0]
        self.mem_weight = config.mem_weight[0]

        self.max_mem = config.max_mem
        if self.max_mem is None:
            self.max_mem = self.mem_forward * self.n_stages * 2

        self.max_chunks = config.max_chunks

    def _category_to_id(self, category: FnType):
        return {FnType.F: 0, FnType.B: 1, FnType.W: 2}.get(category)

    def get_id(
        self,
        category: FnType | str,
        chunk: NodeChunk | int,
        stage: int,
        microbatch: int,
    ):
        if isinstance(category, FnType):
            category = self._category_to_id(category)

        if isinstance(chunk, NodeChunk):
            chunk = chunk.value

        return (
            category * 2 * self.n_stages * self.n_microbatches
            + chunk * self.n_stages * self.n_microbatches
            + stage * self.n_microbatches
            + microbatch
        )

    def get_cost(self, category: FnType):
        if category == FnType.F:
            return self.cost_forward

        elif category == FnType.B:
            return self.cost_backward

        elif category == FnType.W:
            return self.cost_weight

    def get_mem(self, category: FnType):
        if category == FnType.F:
            return self.mem_forward

        elif category == FnType.B:
            return self.mem_backward

        elif category == FnType.W:
            return self.mem_weight

    def build_schedule(self):
        schedule, end_time, max_bubble = None, None, None
        cost_forward = self.cost_forward
        cost_backward = self.cost_backward
        cost_weight = self.cost_weight

        expected_time = (
            (cost_forward + cost_backward + cost_weight) * self.n_microbatches * 2
        )

        for fill_b in (True, False):
            for fill_f in (True, False):
                try_schedule, try_end_time, try_max_bubble = self.try_v_schedule(
                    fill_f=fill_f,
                    fill_b=fill_b,
                )

                if max_bubble is None or try_max_bubble < max_bubble:
                    max_bubble = try_max_bubble
                    schedule = try_schedule
                    end_time = try_end_time

        bubble_rate = max_bubble / (expected_time + max_bubble)

        local_order = [[] for _ in range(self.n_stages)]

        for stage in range(self.n_stages):
            for category, chunk, microbatch in schedule[stage]:
                chunk = (
                    chunk.value
                    if category == FnType.F
                    else self.max_chunks - 1 - chunk.value
                )

                if category in {FnType.F, FnType.B}:
                    assert self.max_chunks == 2

                else:
                    assert category == FnType.W

                layer_group_index = self.n_stages * chunk

                if chunk % 2 == 0:
                    layer_group_index += stage

                else:
                    layer_group_index += self.n_stages - 1 - stage

                local_order[stage].append(
                    ScheduleNode(
                        type=category,
                        chunk=chunk,
                        stage=stage,
                        microbatch=microbatch,
                        layer_group_index=layer_group_index,
                    )
                )

        return local_order, {"bubble_rate": bubble_rate, "end_time": end_time}

    def put_w(
        self, schedule, count, mem, cur_time, end_time, pending_w, stage_bubbles, stage
    ):
        assert len(pending_w[stage]) > 0

        _, chunk, _ = pending_w[stage].popleft()

        self.put(
            schedule,
            count,
            mem,
            cur_time,
            end_time,
            pending_w,
            stage_bubbles,
            FnType.W,
            chunk,
            stage,
        )

    def put(
        self,
        schedule,
        count,
        mem,
        cur_time,
        end_time,
        pending_w,
        stage_bubbles,
        category,
        chunk,
        stage,
        assert_count=True,
    ):
        tmp = no_bubble = cur_time[stage] + self.get_cost(category)
        cat_val = self._category_to_id(category)
        cnt = count[stage][(category, chunk)]

        if cnt >= self.n_microbatches:
            if not assert_count:
                cur_time[stage] = tmp

                return

            assert False

        assert mem[stage] + self.get_mem(category) <= self.max_mem

        if category != FnType.F or chunk == NodeChunk.SECOND:
            last_id = cat_val * 2 + chunk.value - 1

            if category != FnType.W:
                assert end_time[self.get_id(last_id // 2, last_id % 2, stage, cnt)] >= 0

            else:
                assert end_time[self.get_id(FnType.B, 0, stage, cnt)] >= 0

        if chunk == NodeChunk.SECOND and category != FnType.W:
            if stage < self.n_stages - 1:
                fa_id = self.get_id(category, chunk, stage + 1, cnt)
                assert end_time[fa_id] >= 0
                tmp = max(
                    tmp, end_time[fa_id] + self.cost_comm + self.get_cost(category)
                )

        if chunk == NodeChunk.FIRST and category != FnType.W:
            if stage > 0:
                fa_id = self.get_id(category, chunk, stage - 1, cnt)
                assert end_time[fa_id] >= 0, f"{category} {chunk} {stage} {cnt}"
                tmp = max(
                    tmp, end_time[fa_id] + self.cost_comm + self.get_cost(category)
                )

        id = self.get_id(category, chunk, stage, cnt)

        if count[stage][(FnType.F, NodeChunk.FIRST)] > 0:
            stage_bubbles[stage] += tmp - no_bubble

        end_time[id] = tmp
        cur_time[stage] = tmp
        mem[stage] += self.get_mem(category)
        schedule[stage].append((category, chunk, cnt))

        if category == FnType.B:
            pending_w[stage].append((FnType.W, chunk, cnt))

        count[stage][(category, chunk)] += 1

    def get_max_stage_bubble(
        self,
        stage_bubbles,
        approved_bubbles,
        max_approved_bubble,
        stage: int = -1,
    ):
        max_stage_bubble = 0

        for bubble in stage_bubbles:
            max_stage_bubble = max(max_stage_bubble, bubble)

        if stage >= 0:
            max_stage_bubble = max(
                max_stage_bubble, max_approved_bubble - approved_bubbles[stage]
            )

        return max_stage_bubble

    def try_v_schedule(
        self,
        fill_f: bool,
        fill_b: bool,
        approved_bubbles=None,
    ):
        count = []

        for _ in range(self.n_stages):
            count_dict = {}
            for category in (FnType.F, FnType.B, FnType.W):
                for chunk in (NodeChunk.FIRST, NodeChunk.SECOND):
                    count_dict[(category, chunk)] = 0

            count.append(count_dict)

        end_time = [-1] * self.n_nodes
        cur_time = [0] * self.n_stages
        mem = [0] * self.n_stages
        stage_bubbles = [0] * self.n_stages
        pending_w = [deque() for _ in range(self.n_stages)]
        schedule = [[] for _ in range(self.n_stages)]

        if approved_bubbles is None:
            approved_bubbles = [-1] * self.n_stages

        max_approved_bubble = max(approved_bubbles)

        def put_w(stage):
            self.put_w(
                schedule,
                count,
                mem,
                cur_time,
                end_time,
                pending_w,
                stage_bubbles,
                stage,
            )

        def put(
            category,
            chunk,
            stage,
            assert_count=True,
        ):
            self.put(
                schedule,
                count,
                mem,
                cur_time,
                end_time,
                pending_w,
                stage_bubbles,
                category,
                chunk,
                stage,
                assert_count,
            )

        for i in range(self.n_stages):
            put(
                FnType.F,
                NodeChunk.FIRST,
                i,
            )

        for i in range(self.n_stages - 1, -1, -1):
            if i == self.n_stages - 1:
                put(FnType.F, NodeChunk.SECOND, i)

                continue

            tmp = (
                end_time[self.get_id(FnType.F, NodeChunk.SECOND, i + 1, 0)]
                + self.cost_comm
            )

            while (
                mem[i] + self.get_mem(FnType.F) * (2 + i * 2) <= self.max_mem
                and cur_time[i] + self.get_cost(FnType.F) <= tmp
                and count[i][(FnType.F, NodeChunk.FIRST)] < self.n_microbatches
            ):
                for j in range(i + 1):
                    put(FnType.F, NodeChunk.FIRST, j)

            put(FnType.F, NodeChunk.SECOND, i)

        iter_chunk = NodeChunk.FIRST
        end_tmp = 0

        for i in range(self.n_stages):
            if i == 0:
                end_tmp = cur_time[0] + self.get_cost(FnType.B)

                continue

            tmp = end_tmp + self.cost_comm

            while (
                count[i][(FnType.F, NodeChunk.FIRST)]
                + count[i][(FnType.F, NodeChunk.SECOND)]
                < count[i - 1][(FnType.F, NodeChunk.FIRST)]
                + count[i - 1][(FnType.F, NodeChunk.SECOND)]
                or count[i][(FnType.F, NodeChunk.SECOND)]
                <= count[i - 1][(FnType.F, NodeChunk.SECOND)]
                < self.n_microbatches
            ):
                for j in range(self.n_stages - 1, i - 1, -1):
                    if count[j][(FnType.F, iter_chunk)] < self.n_microbatches:
                        put(FnType.F, iter_chunk, j)

                iter_chunk = (
                    NodeChunk.SECOND
                    if iter_chunk == NodeChunk.FIRST
                    else NodeChunk.FIRST
                )

        for _ in range(2 * self.n_microbatches):
            for i in range(self.n_stages):
                while mem[i] + self.get_mem(FnType.B) > self.max_mem:
                    assert len(pending_w[i]) > 0

                    put_w(i)

            b0_ranks, b1_ranks = [], []

            for i in range(self.n_stages):
                if (
                    count[i][(FnType.B, NodeChunk.SECOND)]
                    >= count[i][(FnType.B, NodeChunk.FIRST)]
                ):
                    b0_ranks.append(i)

                elif i == self.n_stages - 1:
                    b1_ranks.append(i)

                else:
                    fa_id = self.get_id(
                        FnType.B,
                        NodeChunk.SECOND,
                        i + 1,
                        count[i][(FnType.B, NodeChunk.SECOND)],
                    )

                    if (
                        end_time[fa_id] >= 0
                        or count[i][(FnType.B, NodeChunk.FIRST)] >= self.n_microbatches
                    ):
                        b1_ranks.append(i)

                    else:
                        b0_ranks.append(i)

            b_ranks = []

            for i in reversed(b1_ranks):
                b_ranks.append((i, NodeChunk.SECOND))

            for i in b0_ranks:
                b_ranks.append((i, NodeChunk.FIRST))

            for i, chunk in b_ranks:
                fa_id = -1

                if chunk == NodeChunk.SECOND and i < self.n_stages - 1:
                    fa_id = self.get_id(
                        FnType.B,
                        NodeChunk.SECOND,
                        i + 1,
                        count[i][(FnType.B, NodeChunk.SECOND)],
                    )

                elif chunk == NodeChunk.FIRST and i > 0:
                    fa_id = self.get_id(
                        FnType.B,
                        NodeChunk.FIRST,
                        i - 1,
                        count[i][(FnType.B, NodeChunk.FIRST)],
                    )

                while (
                    len(pending_w[i]) > 0
                    and fa_id >= 0
                    and end_time[fa_id] + self.cost_comm
                    >= cur_time[i] + self.get_cost(FnType.W)
                ):
                    put_w(i)

                if (
                    len(pending_w[i]) > 0
                    and end_time[fa_id] + self.cost_comm - cur_time[i]
                    > self.get_max_stage_bubble(
                        stage_bubbles, approved_bubbles, max_approved_bubble, i
                    )
                    - stage_bubbles[i]
                ):
                    if chunk == NodeChunk.SECOND or fill_b:
                        put_w(i)

                put(FnType.B, chunk, i)

            for i in range(self.n_stages):
                if count[i][(FnType.F, NodeChunk.SECOND)] >= self.n_microbatches:
                    continue

                put_item = None

                if (
                    count[i][(FnType.F, NodeChunk.SECOND)]
                    >= count[i][(FnType.F, NodeChunk.FIRST)]
                ):
                    put_item = NodeChunk.FIRST

                elif i == self.n_stages - 1:
                    put_item = NodeChunk.SECOND

                else:
                    if (
                        end_time[
                            self.get_id(
                                FnType.F,
                                NodeChunk.SECOND,
                                i + 1,
                                count[i][(FnType.F, NodeChunk.SECOND)],
                            )
                        ]
                        >= 0
                    ):
                        put_item = NodeChunk.SECOND

                    elif count[i][(FnType.F, NodeChunk.FIRST)] < self.n_microbatches:
                        if i == 0:
                            put_item = NodeChunk.FIRST

                        elif (
                            end_time[
                                self.get_id(
                                    FnType.F,
                                    NodeChunk.FIRST,
                                    i - 1,
                                    count[i][(FnType.F, NodeChunk.FIRST)],
                                )
                            ]
                            >= 0
                        ):
                            put_item = NodeChunk.FIRST

                if put_item is None:
                    continue

                while mem[i] + self.get_mem(FnType.F) > self.max_mem:
                    assert len(pending_w[i]) > 0

                    put_w(i)

                fa_id = -1

                if put_item == NodeChunk.FIRST and i > 0:
                    fa_id = self.get_id(
                        FnType.F,
                        NodeChunk.FIRST,
                        i - 1,
                        count[i][(FnType.F, NodeChunk.FIRST)],
                    )

                if put_item == NodeChunk.SECOND and i < self.n_stages - 1:
                    fa_id = self.get_id(
                        FnType.F,
                        NodeChunk.SECOND,
                        i + 1,
                        count[i][(FnType.F, NodeChunk.SECOND)],
                    )

                while (
                    len(pending_w[i]) > 0
                    and fa_id >= 0
                    and end_time[fa_id] + self.cost_comm
                    >= cur_time[i] + self.get_cost(FnType.W)
                ):
                    put_w(i)

                if (
                    len(pending_w[i]) > 0
                    and end_time[fa_id] + self.cost_comm - cur_time[i]
                    > self.get_max_stage_bubble(
                        stage_bubbles, approved_bubbles, max_approved_bubble, i
                    )
                    - stage_bubbles[i]
                ):
                    if fill_f:
                        put_w(i)

                put(FnType.F, put_item, i)

        for i in range(self.n_stages):
            while len(pending_w[i]) > 0:
                put_w(i)

        max_bubble = self.get_max_stage_bubble(
            stage_bubbles, approved_bubbles, max_approved_bubble
        )
        expected_time = (
            (self.cost_forward + self.cost_backward + self.cost_weight)
            * self.n_microbatches
            * 2
        )
        bubble_rate = max_bubble / expected_time

        if max_approved_bubble < 0 or max_bubble < max_approved_bubble:
            try_schedule, try_end_time, try_max_bubble = self.try_v_schedule(
                fill_f=fill_f,
                fill_b=fill_b,
                approved_bubbles=stage_bubbles,
            )

            if try_max_bubble < max_bubble:
                return try_schedule, try_end_time, try_max_bubble

        return schedule, end_time, max_bubble
