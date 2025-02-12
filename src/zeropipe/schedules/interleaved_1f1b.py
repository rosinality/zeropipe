from collections import defaultdict

from zeropipe.schedules.schedule import FnType, ScheduleConfig, ScheduleNode, NodeChunk


class Interleaved1F1BSchedule:
    def __init__(self, config: ScheduleConfig):
        self.n_stages = config.n_stages
        self.n_microbatches = config.n_microbatches
        self.n_rounds = max(1, self.n_microbatches // self.n_stages)
        self.microbatches_per_round = self.n_microbatches // self.n_rounds

        self.max_chunks = config.max_chunks

    def build_schedule(self):
        schedule = []

        for stage in range(self.n_stages):
            schedule.append(self._build_schedule(stage))

        return schedule, {}

    def _build_schedule(self, stage):
        warmup = (self.max_chunks - 1) * self.microbatches_per_round
        warmup = warmup + 2 * ((self.n_stages - 1) - stage)
        warmup_ops = min(warmup, self.n_microbatches * self.max_chunks)

        microbatch_ops = self.max_chunks * self.n_microbatches
        fwd_bwd_ops = microbatch_ops - warmup_ops
        cooldown_ops = microbatch_ops - fwd_bwd_ops

        return self._build_1f1b_schedule(stage, warmup_ops, fwd_bwd_ops, cooldown_ops)

    def _build_1f1b_schedule(self, stage, warmup_ops, fwd_bwd_ops, cooldown_ops):
        fwd_stage_mb_index = defaultdict(int)
        bwd_stage_mb_index = defaultdict(int)

        rank_ops = []

        total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops

        backward_op_ids = []

        for op in range(total_ops):
            if op < warmup_ops:
                chunk, fwd_group_index = self._forward_group_index(stage, op)
                mb_index = fwd_stage_mb_index[fwd_group_index]
                fwd_stage_mb_index[fwd_group_index] += 1
                rank_ops.append(
                    ScheduleNode(
                        type=FnType.F,
                        chunk=chunk,
                        stage=stage,
                        microbatch=mb_index,
                        layer_group_index=fwd_group_index,
                    )
                )

            elif warmup_ops <= op < warmup_ops + fwd_bwd_ops:
                chunk, fwd_group_index = self._forward_group_index(stage, op)
                mb_index = fwd_stage_mb_index[fwd_group_index]
                fwd_stage_mb_index[fwd_group_index] += 1
                rank_ops.append(
                    ScheduleNode(
                        type=FnType.F,
                        chunk=chunk,
                        stage=stage,
                        microbatch=mb_index,
                        layer_group_index=fwd_group_index,
                    )
                )

                chunk, bwd_group_index = self._backward_group_index(
                    stage, warmup_ops, op
                )
                mb_index = bwd_stage_mb_index[bwd_group_index]
                bwd_stage_mb_index[bwd_group_index] += 1
                rank_ops.append(
                    ScheduleNode(
                        type=FnType.BW,
                        chunk=chunk,
                        stage=stage,
                        microbatch=mb_index,
                        layer_group_index=bwd_group_index,
                    )
                )
                backward_op_ids.append(op)

            else:
                chunk, bwd_group_index = self._backward_group_index(
                    stage, warmup_ops, op
                )
                mb_index = bwd_stage_mb_index[bwd_group_index]
                bwd_stage_mb_index[bwd_group_index] += 1
                rank_ops.append(
                    ScheduleNode(
                        type=FnType.BW,
                        chunk=chunk,
                        stage=stage,
                        microbatch=mb_index,
                        layer_group_index=bwd_group_index,
                    )
                )
                backward_op_ids.append(op)

        return rank_ops

    def _forward_group_index(self, stage, step):
        local_index = (step // self.microbatches_per_round) % self.max_chunks

        return local_index, (local_index * self.n_stages) + stage

    def _backward_group_index(self, stage, warmup_ops, step):
        chunk = ((step - warmup_ops) // self.microbatches_per_round) % self.max_chunks
        local_index = self.max_chunks - 1 - chunk

        return chunk, (local_index * self.n_stages) + stage
