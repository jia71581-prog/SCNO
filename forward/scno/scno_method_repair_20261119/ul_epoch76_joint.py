"""Phase C after update 8475, bound to the archived epoch-75 weights."""

from copy import deepcopy
from time import perf_counter

import torch

from repair_stage_schedule import _fingerprint, stage_b_calibration_spec
from ul_backbone_lr_repair import BackboneRateController


FIRST_JOINT_EPOCH = 76


class Epoch76JointController(BackboneRateController):
    """Use the existing C optimizer and a one-time frozen-base calibration."""

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        updates_per_epoch = self.plan['updates_per_epoch']
        if (self.plan['protocol'] != 'staged' or self.plan['epochs'] != 200
                or updates_per_epoch != 113 or not self.config.get('affine_calibration')):
            raise ValueError('epoch-76 transition requires the original UL stages and batch budget')
        original_a, _, original_c = self.plan['stages']
        self.joint_start = (FIRST_JOINT_EPOCH - 1) * updates_per_epoch
        remaining = self.plan['total_updates'] - self.joint_start
        c_stage = dict(deepcopy(original_c), begin=self.joint_start, updates=remaining,
                       start_epoch=float(FIRST_JOINT_EPOCH - 1))
        self.plan.update(
            stages=[c_stage],
            phase_fractions=[self.joint_start / self.plan['total_updates'], 0.0,
                             remaining / self.plan['total_updates']],
            transition={
                'kind': 'isolated_epoch_76_joint_branch',
                'original_A_schedule': deepcopy(original_a),
                'inherited_A_updates': self.joint_start,
                'first_C_epoch_one_based': FIRST_JOINT_EPOCH,
                'first_C_update_one_based': self.joint_start + 1,
                'B_optimizer_updates': 0,
                'B_calibration_before_first_C_update': True,
            },
        )
        self.binding = _fingerprint({'source_controller_binding': self.binding, 'plan': self.plan})

    def prepare_joint(self, calibration_callback):
        if self._resume_pending or self.completed_updates != 0 or self.current_phase is not None:
            raise RuntimeError('run calibration once when bootstrapping an empty branch')
        spec = stage_b_calibration_spec(self.config, self.train_ids, self.native_times)
        self._set_trainability('B')
        self.model.parent.encoder_trainable = False
        start = perf_counter()
        with torch.no_grad():
            receipt = calibration_callback(self.model, deepcopy(spec))
        detail_names = {name for name, _ in self.roles['DETAIL']}
        if (receipt.get('accepted') is not True or not receipt.get('rewritten_parameters')
                or not set(receipt['rewritten_parameters']) <= detail_names):
            raise RuntimeError('the TRAIN-only calibration did not commit a valid DETAIL readout')
        self.calibration = {
            'state': 'complete', 'spec': spec, 'receipt': deepcopy(receipt),
            'pressure_frames': spec['pressure_frames'], 'elapsed_s': perf_counter() - start,
            'completed_updates_at_entry': self.joint_start,
        }
        return self._enter_joint()

    def prepare_joint_after_rejection(self, candidate):
        """Start C on the archived weights after evaluating a harmful fit."""
        if self._resume_pending or self.completed_updates or self.current_phase is not None:
            raise RuntimeError('calibration rejection can only prepare an unused controller')
        spec = stage_b_calibration_spec(self.config, self.train_ids, self.native_times)
        if (candidate['state'] != 'complete' or candidate['spec'] != spec
                or candidate['completed_updates_at_entry'] != self.joint_start):
            raise ValueError('calibration candidate was not fitted at the original epoch-75 boundary')
        self.calibration = {
            **deepcopy(candidate), 'outcome': 'candidate_rejected_on_full_401_validation',
            'applied_to_training_weights': False,
        }
        return self._enter_joint()

    def _enter_joint(self):
        self.completed_updates = self.joint_start
        self.model.parent.encoder_trainable = True
        entry = self.begin_update()
        if self.current_phase != 'C' or self.scheduler.last_epoch != 0 or self.optimizer.state:
            raise RuntimeError('the first C step must start with a fresh phase-local AdamW schedule')
        return entry
