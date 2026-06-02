"""Scripted "grab-the-endpoints-and-stretch" policy for SoftGym RopeFlatten.

State machine (advances when the phase's target is reached or its budget runs out):

    APPROACH_XY  -> position pickers directly above the two rope endpoints (xy only).
    DESCEND      -> lower both pickers to the endpoint heights (z down).
    GRIP         -> hold position, switch grip flag to 1.0 to attach to endpoints.
    LIFT         -> lift the gripped endpoints a small amount (z up).
    STRETCH      -> pull the two pickers apart along the rope's principal axis
                    until they reach the rope's natural length.
    HOLD         -> stay in place to let the rope settle; grip stays on.

Designed for the Picker action space (per picker: (dx, dy, dz, grip)) with action
clipping per env step ~0.01m. Gaussian noise sigma is configurable so that 5000
episodes do not produce 5000 near-identical trajectories.

Compatible with Python 3.6 (used inside the softgym Docker container).
"""

import numpy as np
import pyflex


# Phases — small ints so the rendered metadata is compact
APPROACH_XY = 0
DESCEND = 1
GRIP = 2
LIFT = 3
STRETCH = 4
HOLD = 5


def _picker_xyz(num_picker):
    """Return picker xyz positions as (num_picker, 3) float32."""
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    return shape_states[:num_picker, :3].astype(np.float32)


def _rope_endpoints():
    """Return (xyz_first, xyz_last, natural_length) for the rope.

    SoftGym's rope is modeled as a chain of particles indexed 0..N-1, where 0
    and N-1 are the geometric endpoints (Lin et al., 2020, App A — "10 evenly
    spaced keypoints on the rope, including the two end points").

    natural_length is the sum of segment-to-segment distances along the chain
    — this stays approximately constant regardless of curling, unlike the
    straight-line endpoint distance.
    """
    pos = np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3].astype(np.float32)
    segment_lengths = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    natural_length = float(segment_lengths.sum())
    return pos[0].copy(), pos[-1].copy(), natural_length


class GeometricRopeFlattenPolicy(object):
    """State-machine scripted policy for the RopeFlatten task.

    Parameters
    ----------
    action_space : gym.spaces.Box
        Used only for action low/high (clip bounds). Picker delta is typically
        bounded at +/- 0.01 m per env step.
    num_picker : int
        Expected 2 for RopeFlatten.
    noise_scale : float
        Std-dev of Gaussian noise added to the (dx,dy,dz) deltas as a fraction
        of the action high bound. 0 = deterministic; 0.15 ~ matches colleague's
        prior dataset metadata.
    lateral_scale : float
        Final stretched-apart distance as a fraction of the rope's measured
        length. 1.0 = pull to natural length; >1 will stall against the rope's
        stretch constraint. 0.2 matches the prior dataset metadata.
    lift_height : float
        How high to lift the gripped endpoints above the rope plane (meters).
    rng : np.random.RandomState | None
        Optional RNG for reproducibility. If None, uses np.random.
    """

    def __init__(self, action_space, num_picker=2, noise_scale=0.02,
                 lateral_scale=1.0, lift_height=0.05, action_gain=0.3,
                 max_speed_scale=0.35, deadband=5e-4, rng=None, action_repeat=8):
        assert num_picker == 2, "GeometricRopeFlattenPolicy assumes 2 pickers"
        assert 0.0 < max_speed_scale <= 1.0
        self.action_space = action_space
        self.num_picker = num_picker
        self.noise_scale = float(noise_scale)
        self.lateral_scale = float(lateral_scale)
        self.lift_height = float(lift_height)
        self.action_gain = float(action_gain)
        self.action_repeat = int(action_repeat)  # env applies this many substeps/step
        self.max_speed_scale = float(max_speed_scale)
        self.deadband = float(deadband)
        self.rng = rng if rng is not None else np.random

        # Per-picker delta clip — first 3 dims of (dx,dy,dz,grip). We shrink the
        # env's max action magnitude by max_speed_scale to slow the picker
        # uniformly (the env still applies action_repeat=8 substeps so absolute
        # picker motion is 8 × this cap per env step).
        self.delta_high = (max_speed_scale * action_space.high[:3]).astype(np.float32)
        self.delta_low = (max_speed_scale * action_space.low[:3]).astype(np.float32)

        # Filled in during reset()
        self.phase = APPROACH_XY
        self.phase_step = 0
        self.endpoint0 = None
        self.endpoint1 = None
        self.rope_length = None
        self.targets = None  # per-picker (3,) for the current phase
        self.grip = np.zeros(num_picker, dtype=np.float32)

    # ------------------------------------------------------------------ public

    def reset(self):
        """Read fresh rope endpoints and start the state machine."""
        ep0, ep1, natural_length = _rope_endpoints()
        self.endpoint0 = ep0
        self.endpoint1 = ep1
        self.rope_length = natural_length  # uncurled length, ~constant across episode

        # APPROACH_XY target: directly above each endpoint, with lift_height clearance
        self.targets = np.stack([
            np.array([ep0[0], max(ep0[1] + self.lift_height, self.lift_height), ep0[2]], dtype=np.float32),
            np.array([ep1[0], max(ep1[1] + self.lift_height, self.lift_height), ep1[2]], dtype=np.float32),
        ], axis=0)
        self.phase = APPROACH_XY
        self.phase_step = 0
        self.grip[:] = 0.0

    def get_action(self):
        """Return the action vector for the current step, then advance phase if needed."""
        picker_xyz = _picker_xyz(self.num_picker)

        # Re-read rope endpoints while we're still chasing them (rope is settling
        # during the first ~5 env steps; the targets we cached at reset() are stale).
        if self.phase in (APPROACH_XY, DESCEND):
            ep0, ep1, natural_length = _rope_endpoints()
            self.endpoint0 = ep0
            self.endpoint1 = ep1
            self.rope_length = natural_length
            if self.phase == APPROACH_XY:
                self.targets = np.stack([
                    np.array([ep0[0], ep0[1] + self.lift_height, ep0[2]], dtype=np.float32),
                    np.array([ep1[0], ep1[1] + self.lift_height, ep1[2]], dtype=np.float32),
                ], axis=0)
            else:  # DESCEND
                self.targets = np.stack([
                    np.array([ep0[0], ep0[1] + 0.005, ep0[2]], dtype=np.float32),
                    np.array([ep1[0], ep1[1] + 0.005, ep1[2]], dtype=np.float32),
                ], axis=0)

        # Proportional control with gain < 1.0 → smooth deceleration near the
        # target instead of bang-bang. When err >> delta_high/gain, the clip
        # still saturates at max speed, so far-from-target moves stay fast.
        deltas = np.zeros((self.num_picker, 3), dtype=np.float32)
        for i in range(self.num_picker):
            err = self.targets[i] - picker_xyz[i]
            deltas[i] = np.clip(self.action_gain * err, self.delta_low, self.delta_high)

        # Phase advance: when all pickers are within delta_high of their targets,
        # OR when phase has run too long, transition to the next phase.
        max_err = float(np.max(np.abs(self.targets - picker_xyz)))
        target_reached = max_err < 0.5 * float(np.max(self.delta_high))

        # Per-phase max step budget (defensive — most phases finish much sooner)
        phase_budget = {
            APPROACH_XY: 20,
            DESCEND:     10,
            GRIP:         3,
            LIFT:        10,
            STRETCH:     40,
            HOLD:        20,
        }[self.phase]
        if target_reached or self.phase_step >= phase_budget:
            self._advance_phase(picker_xyz)
            # Recompute deltas for the new phase so we don't waste this step
            for i in range(self.num_picker):
                err = self.targets[i] - picker_xyz[i]
                deltas[i] = np.clip(self.action_gain * err, self.delta_low, self.delta_high)

        # Add Gaussian noise on the delta component (not on grip), scaled by
        # the actual commanded magnitude with no floor. When the policy
        # commands ~0 (HOLD), noise is also ~0, so the picker doesn't jitter.
        if self.noise_scale > 0.0:
            noise = self.rng.randn(*deltas.shape).astype(np.float32)
            noise *= (self.noise_scale * np.abs(deltas))
            deltas = np.clip(deltas + noise, self.delta_low, self.delta_high)

        # Anti-overshoot: the env applies `action_repeat` substeps, so the real
        # per-step motion is ~action_repeat * delta. Cap each delta so the picker
        # never travels PAST the target in one recorded step (the root cause of
        # the bang-bang oscillation: effective gain action_repeat*action_gain>>1).
        cap = np.abs(self.targets[:, :3] - picker_xyz) / float(self.action_repeat)
        deltas = np.clip(deltas, -cap, cap)

        # Deadband: any per-axis command below this threshold is zeroed. Kills
        # proportional-control oscillation during HOLD/GRIP where the rope
        # pulls the picker by sub-mm and the controller would otherwise
        # overshoot 8× via action_repeat.
        deltas = np.where(np.abs(deltas) < self.deadband, 0.0, deltas)

        # Pack into 4*num_picker action vector: [dx,dy,dz,grip]*num_picker
        action = np.zeros(4 * self.num_picker, dtype=np.float32)
        for i in range(self.num_picker):
            action[4 * i:4 * i + 3] = deltas[i]
            action[4 * i + 3] = self.grip[i]
        self.phase_step += 1
        return action

    # ------------------------------------------------------------------ private

    def _advance_phase(self, picker_xyz):
        """Compute next-phase targets and grip state from the current picker positions."""
        if self.phase == APPROACH_XY:
            # Descend to endpoint heights (slightly above the particle so contact is reliable).
            self.targets = np.stack([
                np.array([self.endpoint0[0], self.endpoint0[1] + 0.005, self.endpoint0[2]], dtype=np.float32),
                np.array([self.endpoint1[0], self.endpoint1[1] + 0.005, self.endpoint1[2]], dtype=np.float32),
            ], axis=0)
            self.phase = DESCEND

        elif self.phase == DESCEND:
            # Hold position; turn grip on. Targets stay where the pickers currently are.
            self.targets = picker_xyz.copy()
            self.grip[:] = 1.0
            self.phase = GRIP

        elif self.phase == GRIP:
            # Lift both endpoints slightly.
            self.targets = picker_xyz.copy()
            self.targets[:, 1] = picker_xyz[:, 1] + self.lift_height
            self.phase = LIFT

        elif self.phase == LIFT:
            # Move pickers apart along the current axis to a target separation
            # equal to lateral_scale * rope_natural_length. lateral_scale=1.0
            # pulls to the full natural length (rope straightens).
            midpoint = 0.5 * (picker_xyz[0] + picker_xyz[1])
            axis = picker_xyz[1] - picker_xyz[0]
            n = float(np.linalg.norm(axis) + 1e-8)
            unit = axis / n
            target_separation = self.lateral_scale * self.rope_length
            self.targets = np.stack([
                midpoint - 0.5 * target_separation * unit,
                midpoint + 0.5 * target_separation * unit,
            ], axis=0)
            # keep y at the lifted height
            self.targets[:, 1] = picker_xyz[:, 1]
            self.phase = STRETCH

        elif self.phase == STRETCH:
            # Hold in place; grip stays on so rope is held taut while it settles.
            self.targets = picker_xyz.copy()
            self.phase = HOLD

        else:  # HOLD
            # Stay; do nothing more.
            self.targets = picker_xyz.copy()

        self.phase_step = 0


# ---------------------------------------------------------------------------
# Free-form rope manipulation policy — grabs random non-endpoint particles
# and walks each picker through a smooth waypoint sequence.
# ---------------------------------------------------------------------------

class SmoothRopeManipulationPolicy(object):
    """Manipulation policy that grabs random *middle* rope particles and moves
    them along a smooth waypoint sequence. Produces diverse rope configurations
    rather than always-flattened final states.

    Reuses the same low-level controller (proportional + speed cap + deadband
    + noise) as GeometricRopeFlattenPolicy, but the high-level plan is:

        SETTLE       -> wait a few steps for the rope to fall and stop wiggling.
        PICK_TARGETS -> assign each picker a random non-endpoint particle.
        APPROACH_XY  -> position above the assigned particles.
        DESCEND      -> lower to particle height.
        GRIP         -> attach.
        LIFT         -> raise to a chosen lift height.
        WAYPOINTS    -> walk each picker through K random waypoints in turn.
        RELEASE      -> drop and end.

    Parameters mirror GeometricRopeFlattenPolicy where they overlap.
    """

    SETTLE = 0
    APPROACH_XY = 1
    DESCEND = 2
    GRIP = 3
    LIFT = 4
    WAYPOINTS = 5
    RELEASE = 6

    def __init__(self, action_space, num_picker=2, noise_scale=0.02,
                 action_gain=0.3, max_speed_scale=0.35, deadband=5e-4,
                 lift_height=0.08, settle_steps=3, num_waypoints=3,
                 grip_margin=0.05, rng=None, action_repeat=8):
        assert num_picker == 2
        assert 0.0 < max_speed_scale <= 1.0
        self.action_space = action_space
        self.num_picker = num_picker
        self.noise_scale = float(noise_scale)
        self.action_gain = float(action_gain)
        self.action_repeat = int(action_repeat)  # env applies this many substeps/step
        self.max_speed_scale = float(max_speed_scale)
        self.deadband = float(deadband)
        self.lift_height = float(lift_height)
        self.settle_steps = int(settle_steps)
        self.num_waypoints = int(num_waypoints)
        self.grip_margin = float(grip_margin)  # fraction of rope at each end to avoid
        self.rng = rng if rng is not None else np.random

        self.delta_high = (max_speed_scale * action_space.high[:3]).astype(np.float32)
        self.delta_low = (max_speed_scale * action_space.low[:3]).astype(np.float32)
        # Picker workspace bounds (rope env defaults; see softgym/envs/rope_env.py)
        self.picker_low_xyz = np.array([-0.30, 0.02, -0.30], dtype=np.float32)
        self.picker_high_xyz = np.array([+0.30, 0.28, +0.30], dtype=np.float32)

        # Episode state
        self.phase = self.SETTLE
        self.phase_step = 0
        self.particle_indices = None  # picked particle index per picker
        self.targets = None
        self.grip = np.zeros(num_picker, dtype=np.float32)
        self.waypoint_seqs = None  # (num_picker, num_waypoints, 3)
        self.waypoint_idx = 0

    def reset(self):
        self.phase = self.SETTLE
        self.phase_step = 0
        self.grip[:] = 0.0
        self.particle_indices = None
        self.waypoint_seqs = None
        self.waypoint_idx = 0
        # SETTLE has no target — controllers commanded to current picker pos.
        self.targets = _picker_xyz(self.num_picker).copy()

    def get_action(self):
        picker_xyz = _picker_xyz(self.num_picker)

        # First, deal with phase-specific target / state updates that happen
        # at the START of each call (when the phase has run for >= 1 step).
        if self.phase == self.SETTLE and self.phase_step >= self.settle_steps:
            self._enter_approach_xy(picker_xyz)
        elif self.phase == self.APPROACH_XY:
            # Track the assigned particles in case the rope is still settling.
            self._refresh_targets_to_particles(above=True)

        # Compute deltas (proportional control + clip).
        deltas = np.zeros((self.num_picker, 3), dtype=np.float32)
        for i in range(self.num_picker):
            err = self.targets[i] - picker_xyz[i]
            deltas[i] = np.clip(self.action_gain * err, self.delta_low, self.delta_high)

        # Phase advance check (after computing deltas for the current phase).
        max_err = float(np.max(np.abs(self.targets - picker_xyz)))
        target_reached = max_err < 0.5 * float(np.max(self.delta_high))
        phase_budget = {
            self.SETTLE:      30,
            self.APPROACH_XY: 25,
            self.DESCEND:     15,
            self.GRIP:         2,
            self.LIFT:        15,
            self.WAYPOINTS:   60,  # checked per-waypoint via target_reached
            self.RELEASE:    100,
        }[self.phase]
        if target_reached or self.phase_step >= phase_budget:
            self._advance_phase(picker_xyz)
            # Recompute deltas for the new phase so we don't waste this step.
            for i in range(self.num_picker):
                err = self.targets[i] - picker_xyz[i]
                deltas[i] = np.clip(self.action_gain * err, self.delta_low, self.delta_high)

        # Noise: zero when commanded zero (no floor).
        if self.noise_scale > 0.0:
            noise = self.rng.randn(*deltas.shape).astype(np.float32)
            noise *= (self.noise_scale * np.abs(deltas))
            deltas = np.clip(deltas + noise, self.delta_low, self.delta_high)

        # Anti-overshoot: env applies action_repeat substeps, so cap each delta
        # so the picker never travels past the target in one recorded step
        # (root cause of the bang-bang oscillation; see GeometricRopeFlattenPolicy).
        cap = np.abs(self.targets[:, :3] - picker_xyz) / float(self.action_repeat)
        deltas = np.clip(deltas, -cap, cap)

        # Deadband.
        deltas = np.where(np.abs(deltas) < self.deadband, 0.0, deltas)

        action = np.zeros(4 * self.num_picker, dtype=np.float32)
        for i in range(self.num_picker):
            action[4 * i:4 * i + 3] = deltas[i]
            action[4 * i + 3] = self.grip[i]
        self.phase_step += 1
        return action

    # -------------------------------------------------------------- helpers

    def _particle_positions(self):
        return np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3].astype(np.float32)

    def _enter_approach_xy(self, picker_xyz):
        """Pick two distinct middle particles and prepare APPROACH targets."""
        pos = self._particle_positions()
        n = pos.shape[0]
        margin = max(1, int(self.grip_margin * n))
        lo, hi = margin, n - margin  # avoid endpoints
        candidates = list(range(lo, hi))
        self.rng.shuffle(candidates)
        # Pick two indices that are well-separated along the rope.
        i0 = candidates[0]
        i1 = next((c for c in candidates[1:] if abs(c - i0) > margin * 2), candidates[-1])
        self.particle_indices = [i0, i1]
        self.phase = self.APPROACH_XY
        self.phase_step = 0
        self._refresh_targets_to_particles(above=True)
        self._plan_waypoints()

    def _refresh_targets_to_particles(self, above):
        """Set targets to the assigned particles' xyz, optionally lifted."""
        pos = self._particle_positions()
        new_targets = np.zeros((self.num_picker, 3), dtype=np.float32)
        for i, idx in enumerate(self.particle_indices):
            new_targets[i, 0] = pos[idx, 0]
            new_targets[i, 2] = pos[idx, 2]
            if above:
                new_targets[i, 1] = pos[idx, 1] + self.lift_height
            else:
                new_targets[i, 1] = pos[idx, 1] + 0.005
        self.targets = new_targets

    def _plan_waypoints(self):
        """Pick `num_waypoints` random xyz targets per picker, within bounds."""
        seqs = np.zeros((self.num_picker, self.num_waypoints, 3), dtype=np.float32)
        for i in range(self.num_picker):
            for k in range(self.num_waypoints):
                seqs[i, k, 0] = self.rng.uniform(self.picker_low_xyz[0] + 0.05,
                                                 self.picker_high_xyz[0] - 0.05)
                seqs[i, k, 1] = self.rng.uniform(0.05, self.lift_height + 0.05)
                seqs[i, k, 2] = self.rng.uniform(self.picker_low_xyz[2] + 0.05,
                                                 self.picker_high_xyz[2] - 0.05)
        self.waypoint_seqs = seqs

    def _advance_phase(self, picker_xyz):
        if self.phase == self.SETTLE:
            self._enter_approach_xy(picker_xyz)
        elif self.phase == self.APPROACH_XY:
            self._refresh_targets_to_particles(above=False)
            self.phase = self.DESCEND
        elif self.phase == self.DESCEND:
            self.targets = picker_xyz.copy()
            self.grip[:] = 1.0
            self.phase = self.GRIP
        elif self.phase == self.GRIP:
            self.targets = picker_xyz.copy()
            self.targets[:, 1] = picker_xyz[:, 1] + self.lift_height
            self.phase = self.LIFT
        elif self.phase == self.LIFT:
            self.targets = self.waypoint_seqs[:, 0, :].copy()
            self.waypoint_idx = 0
            self.phase = self.WAYPOINTS
        elif self.phase == self.WAYPOINTS:
            self.waypoint_idx += 1
            if self.waypoint_idx >= self.num_waypoints:
                # Move down to release height, then drop grip.
                self.targets = picker_xyz.copy()
                self.targets[:, 1] = 0.04
                self.phase = self.RELEASE
            else:
                self.targets = self.waypoint_seqs[:, self.waypoint_idx, :].copy()
        else:  # RELEASE
            self.grip[:] = 0.0
            self.targets = picker_xyz.copy()
        self.phase_step = 0


class PushRopePolicy(object):
    """Push the rope around WITHOUT grabbing (grip stays 0). The pickers descend
    to the rope plane, then sweep through smooth random waypoints near the rope,
    shoving it by collision. Smooth (anti-overshoot) like the grab policies — it
    gives the world model clean push/collision dynamics, not just grasp dynamics.
    """

    DESCEND, SWEEP = 0, 1

    def __init__(self, action_space, num_picker=2, noise_scale=0.02,
                 action_gain=0.3, max_speed_scale=0.35, deadband=5e-4,
                 num_waypoints=4, plane_y=0.04, spread=0.18, rng=None, action_repeat=8):
        assert num_picker == 2
        self.action_space = action_space
        self.num_picker = num_picker
        self.noise_scale = float(noise_scale)
        self.action_gain = float(action_gain)
        self.action_repeat = int(action_repeat)
        self.deadband = float(deadband)
        self.num_waypoints = int(num_waypoints)
        self.plane_y = float(plane_y)
        self.spread = float(spread)
        self.rng = rng if rng is not None else np.random
        self.delta_high = (max_speed_scale * action_space.high[:3]).astype(np.float32)
        self.delta_low = (max_speed_scale * action_space.low[:3]).astype(np.float32)
        self.grip = np.zeros(num_picker, dtype=np.float32)
        self.phase = self.DESCEND
        self.phase_step = 0
        self.targets = None
        self.waypoint_seqs = None
        self.waypoint_idx = 0

    def _sample_waypoints(self, center):
        wps = np.zeros((self.num_picker, self.num_waypoints, 3), np.float32)
        for i in range(self.num_picker):
            for k in range(self.num_waypoints):
                xz = center[[0, 2]] + self.rng.uniform(-self.spread, self.spread, size=2).astype(np.float32)
                wps[i, k] = [xz[0], self.plane_y, xz[1]]
        return wps

    def reset(self):
        ep0, ep1, _ = _rope_endpoints()
        ep0 = np.asarray(ep0, np.float32); ep1 = np.asarray(ep1, np.float32)
        center = 0.5 * (ep0 + ep1)
        # Descend to the rope's MEASURED height (like the grab policies' DESCEND
        # target) so the picker sphere is in the rope's plane and actually
        # collides with it — a hardcoded plane height misses the rope.
        self.plane_y = float(0.5 * (ep0[1] + ep1[1]))
        picker_xyz = _picker_xyz(self.num_picker)
        self.targets = picker_xyz.copy()
        self.targets[:, 1] = self.plane_y                 # descend to rope plane
        self.waypoint_seqs = self._sample_waypoints(center)
        self.waypoint_idx = 0
        self.phase = self.DESCEND
        self.phase_step = 0
        self.grip[:] = 0.0

    def get_action(self):
        picker_xyz = _picker_xyz(self.num_picker)
        max_err = float(np.max(np.abs(self.targets - picker_xyz)))
        reached = max_err < 0.5 * float(np.max(self.delta_high))
        budget = 12 if self.phase == self.DESCEND else 25
        if reached or self.phase_step >= budget:
            if self.phase == self.DESCEND:
                self.phase = self.SWEEP
                self.targets = self.waypoint_seqs[:, 0, :].copy()
                self.waypoint_idx = 0
            else:  # cycle through sweep waypoints
                self.waypoint_idx = (self.waypoint_idx + 1) % self.num_waypoints
                self.targets = self.waypoint_seqs[:, self.waypoint_idx, :].copy()
            self.phase_step = 0

        deltas = np.zeros((self.num_picker, 3), np.float32)
        for i in range(self.num_picker):
            err = self.targets[i] - picker_xyz[i]
            deltas[i] = np.clip(self.action_gain * err, self.delta_low, self.delta_high)
        if self.noise_scale > 0.0:
            noise = self.rng.randn(*deltas.shape).astype(np.float32) * (self.noise_scale * np.abs(deltas))
            deltas = np.clip(deltas + noise, self.delta_low, self.delta_high)
        cap = np.abs(self.targets[:, :3] - picker_xyz) / float(self.action_repeat)  # anti-overshoot
        deltas = np.clip(deltas, -cap, cap)
        deltas = np.where(np.abs(deltas) < self.deadband, 0.0, deltas)

        action = np.zeros(4 * self.num_picker, dtype=np.float32)
        for i in range(self.num_picker):
            action[4 * i:4 * i + 3] = deltas[i]
            action[4 * i + 3] = 0.0                        # grip OFF: push, never grab
        self.phase_step += 1
        return action


class GoalRopeConfigurationPolicy(object):
    """Deterministic goal-conditioned RopeConfiguration policy.

    The target character is stored as an ordered rope path. This policy
    repeatedly picks the two currently worst ordered keypoints and drags them
    to their corresponding target locations. It is an oracle data-collection
    policy, not a deployable vision policy.
    """

    APPROACH = 0
    DESCEND = 1
    GRIP = 2
    LIFT = 3
    DRAG = 4
    PLACE = 5
    RELEASE = 6
    SETTLE = 7

    def __init__(self, env, action_space, num_picker=2, action_gain=1.0,
                 max_speed_scale=1.0, deadband=1e-5, lift_height=0.08,
                 place_height=0.055, settle_steps=1, rng=None,
                 action_repeat=8):
        assert num_picker == 2
        self.env = env
        self.action_space = action_space
        self.num_picker = num_picker
        self.action_gain = float(action_gain)
        self.action_repeat = int(action_repeat)
        self.deadband = float(deadband)
        self.lift_height = float(lift_height)
        self.place_height = float(place_height)
        self.settle_steps = int(settle_steps)
        self.rng = rng if rng is not None else np.random

        self.delta_high = (max_speed_scale * action_space.high[:3]).astype(np.float32)
        self.delta_low = (max_speed_scale * action_space.low[:3]).astype(np.float32)

        self.phase = self.SETTLE
        self.phase_step = 0
        self.targets = None
        self.grip = np.zeros(num_picker, dtype=np.float32)
        self.pick_particle_indices = None
        self.goal_targets = None
        self.last_keypoint_indices = []
        self.greedy_top_k = 6
        self.solve_threshold = 0.95
        self.solved = False

    def reset(self):
        self.phase = self.SETTLE
        self.phase_step = 0
        self.grip[:] = 0.0
        self.targets = _picker_xyz(self.num_picker).copy()
        self.pick_particle_indices = None
        self.goal_targets = None
        self.last_keypoint_indices = []
        self.solved = False

    def get_action(self):
        picker_xyz = _picker_xyz(self.num_picker)
        if not self.solved:
            info = self.env._get_info()
            perf = float(info.get("normalized_performance", info.get("performance", -1e9)))
            if perf >= self.solve_threshold:
                self.solved = True
                self.targets = picker_xyz.copy()
                picked = getattr(self.env.action_tool, "picked_particles", [None] * self.num_picker)
                self.grip = np.array(
                    [0.0 if picked[i] is None else 1.0 for i in range(self.num_picker)],
                    dtype=np.float32,
                )
        if self.solved:
            action = np.zeros(4 * self.num_picker, dtype=np.float32)
            for i in range(self.num_picker):
                action[4 * i + 3] = self.grip[i]
            return action

        if self.phase == self.SETTLE and self.phase_step >= self.settle_steps:
            self._start_new_pick(picker_xyz)
        elif self.phase in (self.APPROACH, self.DESCEND):
            self._refresh_pick_targets(above=(self.phase == self.APPROACH))

        deltas = np.zeros((self.num_picker, 3), dtype=np.float32)
        for i in range(self.num_picker):
            err = self.targets[i] - picker_xyz[i]
            deltas[i] = np.clip(self.action_gain * err, self.delta_low, self.delta_high)

        max_err = float(np.max(np.abs(self.targets - picker_xyz)))
        target_reached = max_err < 0.006
        phase_budget = {
            self.SETTLE:   max(1, self.settle_steps),
            self.APPROACH: 4,
            self.DESCEND:  3,
            self.GRIP:     1,
            self.LIFT:     2,
            self.DRAG:     5,
            self.PLACE:    2,
            self.RELEASE:  1,
        }[self.phase]

        if target_reached or self.phase_step >= phase_budget:
            self._advance_phase(picker_xyz)
            for i in range(self.num_picker):
                err = self.targets[i] - picker_xyz[i]
                deltas[i] = np.clip(self.action_gain * err, self.delta_low, self.delta_high)

        cap = np.abs(self.targets[:, :3] - picker_xyz) / float(self.action_repeat)
        deltas = np.clip(deltas, -cap, cap)
        deltas = np.where(np.abs(deltas) < self.deadband, 0.0, deltas)

        action = np.zeros(4 * self.num_picker, dtype=np.float32)
        for i in range(self.num_picker):
            action[4 * i:4 * i + 3] = deltas[i]
            action[4 * i + 3] = self.grip[i]
        self.phase_step += 1
        return action

    def _particle_positions(self):
        return np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3].astype(np.float32)

    def _keypoint_indices(self):
        if hasattr(self.env, "key_point_indices"):
            return list(self.env.key_point_indices)
        n = self._particle_positions().shape[0]
        indices = [0]
        interval = (n - 2) // 8
        for i in range(1, 9):
            indices.append(i * interval)
        indices.append(n - 1)
        return indices

    def _assignment(self):
        pos = self._particle_positions()
        kp = self._keypoint_indices()
        cur = pos[kp]
        goal = np.asarray(self.env.current_config["goal_character_pos"], dtype=np.float32)[:, :3]
        goal_kp = goal[kp]
        pairs = []
        for i in range(len(kp)):
            dist = float(np.linalg.norm(cur[i] - goal_kp[i]))
            pairs.append((dist, int(i), int(i), int(kp[i]), goal_kp[i].copy()))
        pairs.sort(reverse=True, key=lambda x: x[0])
        return pairs

    def _start_new_pick(self, picker_xyz):
        del picker_xyz
        pairs = self._assignment()
        selected = self._choose_greedy_pair(pairs)

        self.pick_particle_indices = [item[3] for item in selected[:self.num_picker]]
        self.goal_targets = np.stack([item[4] for item in selected[:self.num_picker]], axis=0)
        self.last_keypoint_indices = [item[1] for item in selected[:self.num_picker]]
        self._refresh_pick_targets(above=True)
        self.grip[:] = 0.0
        self.phase = self.APPROACH
        self.phase_step = 0

    def _choose_greedy_pair(self, pairs):
        pool = []
        for item in pairs:
            if item[1] in self.last_keypoint_indices and len(pairs) > self.num_picker:
                continue
            pool.append(item)
            if len(pool) >= self.greedy_top_k:
                break
        if len(pool) < self.num_picker:
            pool = pairs[:max(self.num_picker, min(self.greedy_top_k, len(pairs)))]

        candidates = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                if abs(pool[i][1] - pool[j][1]) <= 1 and len(pool) > 3:
                    continue
                candidates.append([pool[i], pool[j]])
        if not candidates and len(pool) >= self.num_picker:
            candidates.append(pool[:self.num_picker])
        if not candidates:
            return pairs[:self.num_picker]

        best = None
        best_score = -1e9
        for cand in candidates:
            score = self._score_candidate(cand)
            if score > best_score:
                best_score = score
                best = cand
        return best if best is not None else pairs[:self.num_picker]

    def _action_toward(self, targets, grip):
        picker_xyz = _picker_xyz(self.num_picker)
        deltas = np.zeros((self.num_picker, 3), dtype=np.float32)
        for i in range(self.num_picker):
            err = targets[i] - picker_xyz[i]
            deltas[i] = np.clip(self.action_gain * err, self.delta_low, self.delta_high)
        cap = np.abs(targets[:, :3] - picker_xyz) / float(self.action_repeat)
        deltas = np.clip(deltas, -cap, cap)
        deltas = np.where(np.abs(deltas) < self.deadband, 0.0, deltas)
        action = np.zeros(4 * self.num_picker, dtype=np.float32)
        for i in range(self.num_picker):
            action[4 * i:4 * i + 3] = deltas[i]
            action[4 * i + 3] = grip[i]
        return action

    def _step_no_render(self, action):
        for _ in range(self.action_repeat):
            self.env._step(action)

    def _score_candidate(self, cand):
        state = self.env.get_state()
        picked = list(getattr(self.env.action_tool, "picked_particles", [None] * self.num_picker))
        time_step = getattr(self.env, "time_step", 0)
        try:
            particle_indices = [item[3] for item in cand[:self.num_picker]]
            goals = np.stack([item[4] for item in cand[:self.num_picker]], axis=0)
            grip0 = np.zeros(self.num_picker, dtype=np.float32)
            grip1 = np.ones(self.num_picker, dtype=np.float32)

            pos = self._particle_positions()
            targets = np.zeros((self.num_picker, 3), dtype=np.float32)
            for i, particle_idx in enumerate(particle_indices):
                targets[i] = pos[particle_idx]
                targets[i, 1] += self.lift_height
            self._step_no_render(self._action_toward(targets, grip0))

            pos = self._particle_positions()
            for i, particle_idx in enumerate(particle_indices):
                targets[i] = pos[particle_idx]
                targets[i, 1] += 0.005
            self._step_no_render(self._action_toward(targets, grip0))
            self._step_no_render(self._action_toward(_picker_xyz(self.num_picker), grip1))

            targets = _picker_xyz(self.num_picker).copy()
            targets[:, 1] += self.lift_height
            self._step_no_render(self._action_toward(targets, grip1))

            targets = goals.copy()
            targets[:, 1] = np.maximum(targets[:, 1], self.place_height + self.lift_height)
            self._step_no_render(self._action_toward(targets, grip1))

            targets = goals.copy()
            targets[:, 1] = self.place_height
            self._step_no_render(self._action_toward(targets, grip1))
            self._step_no_render(self._action_toward(_picker_xyz(self.num_picker), grip0))

            info = self.env._get_info()
            return float(info.get("normalized_performance", info.get("performance", -1e9)))
        finally:
            self.env.set_state(state)
            self.env.action_tool.picked_particles = picked
            self.env.time_step = time_step

    def _refresh_pick_targets(self, above):
        pos = self._particle_positions()
        targets = np.zeros((self.num_picker, 3), dtype=np.float32)
        for i, particle_idx in enumerate(self.pick_particle_indices):
            targets[i] = pos[particle_idx]
            targets[i, 1] += self.lift_height if above else 0.005
        self.targets = targets

    def _advance_phase(self, picker_xyz):
        if self.phase == self.SETTLE:
            self._start_new_pick(picker_xyz)
        elif self.phase == self.APPROACH:
            self._refresh_pick_targets(above=False)
            self.phase = self.DESCEND
        elif self.phase == self.DESCEND:
            self.targets = picker_xyz.copy()
            self.grip[:] = 1.0
            self.phase = self.GRIP
        elif self.phase == self.GRIP:
            self.targets = picker_xyz.copy()
            self.targets[:, 1] += self.lift_height
            self.phase = self.LIFT
        elif self.phase == self.LIFT:
            self.targets = self.goal_targets.copy()
            self.targets[:, 1] = np.maximum(self.targets[:, 1], self.place_height + self.lift_height)
            self.phase = self.DRAG
        elif self.phase == self.DRAG:
            self.targets = self.goal_targets.copy()
            self.targets[:, 1] = self.place_height
            self.phase = self.PLACE
        elif self.phase == self.PLACE:
            self.targets = picker_xyz.copy()
            self.grip[:] = 0.0
            self.phase = self.RELEASE
        elif self.phase == self.RELEASE:
            self.targets = picker_xyz.copy()
            self.targets[:, 1] = np.maximum(self.targets[:, 1], self.lift_height)
            self.phase = self.SETTLE
        self.phase_step = 0


class TraceRopeConfigurationPolicy(object):
    """Endpoint-tracing RopeConfiguration oracle.

    Grips the two rope endpoints and traces the ordered target-character path
    from both ends toward the middle. This uses simulator state for data
    collection, but the recorded actions are ordinary picker deltas.
    """

    SETTLE, APPROACH, DESCEND, GRIP, TRACE, HOLD = range(6)

    def __init__(self, env, action_space, num_picker=2, action_gain=1.0,
                 max_speed_scale=1.0, deadband=1e-5, lift_height=0.08,
                 trace_height=0.055, rng=None, action_repeat=8):
        assert num_picker == 2
        self.env = env
        self.action_space = action_space
        self.num_picker = num_picker
        self.action_gain = float(action_gain)
        self.action_repeat = int(action_repeat)
        self.deadband = float(deadband)
        self.lift_height = float(lift_height)
        self.trace_height = float(trace_height)
        self.rng = rng if rng is not None else np.random
        self.delta_high = (max_speed_scale * action_space.high[:3]).astype(np.float32)
        self.delta_low = (max_speed_scale * action_space.low[:3]).astype(np.float32)

        self.phase = self.SETTLE
        self.phase_step = 0
        self.targets = None
        self.grip = np.zeros(num_picker, dtype=np.float32)
        self.path0 = None
        self.path1 = None
        self.path_idx = 0
        self.solved = False
        self.solve_threshold = 0.95

    def reset(self):
        self.phase = self.SETTLE
        self.phase_step = 0
        self.grip[:] = 0.0
        self.targets = _picker_xyz(self.num_picker).copy()
        self.path_idx = 0
        self.solved = False
        self._plan_paths()

    def get_action(self):
        picker_xyz = _picker_xyz(self.num_picker)
        if not self.solved:
            info = self.env._get_info()
            perf = float(info.get("normalized_performance", info.get("performance", -1e9)))
            if perf >= self.solve_threshold:
                self.solved = True
                self.phase = self.HOLD
                self.targets = picker_xyz.copy()
                picked = getattr(self.env.action_tool, "picked_particles", [None] * self.num_picker)
                self.grip = np.array(
                    [0.0 if picked[i] is None else 1.0 for i in range(self.num_picker)],
                    dtype=np.float32,
                )

        if self.phase == self.SETTLE and self.phase_step >= 1:
            self._enter_approach()
        elif self.phase == self.APPROACH:
            self._refresh_endpoint_targets(above=True)
        elif self.phase == self.DESCEND:
            self._refresh_endpoint_targets(above=False)
        elif self.phase == self.TRACE:
            self._set_trace_targets()

        deltas = np.zeros((self.num_picker, 3), dtype=np.float32)
        for i in range(self.num_picker):
            err = self.targets[i] - picker_xyz[i]
            deltas[i] = np.clip(self.action_gain * err, self.delta_low, self.delta_high)

        max_err = float(np.max(np.abs(self.targets - picker_xyz)))
        target_reached = max_err < 0.006
        budget = {
            self.SETTLE: 1,
            self.APPROACH: 4,
            self.DESCEND: 3,
            self.GRIP: 1,
            self.TRACE: 2,
            self.HOLD: 1000,
        }[self.phase]
        if target_reached or self.phase_step >= budget:
            self._advance_phase(picker_xyz)
            for i in range(self.num_picker):
                err = self.targets[i] - picker_xyz[i]
                deltas[i] = np.clip(self.action_gain * err, self.delta_low, self.delta_high)

        cap = np.abs(self.targets[:, :3] - picker_xyz) / float(self.action_repeat)
        deltas = np.clip(deltas, -cap, cap)
        deltas = np.where(np.abs(deltas) < self.deadband, 0.0, deltas)

        action = np.zeros(4 * self.num_picker, dtype=np.float32)
        for i in range(self.num_picker):
            action[4 * i:4 * i + 3] = deltas[i]
            action[4 * i + 3] = self.grip[i]
        self.phase_step += 1
        return action

    def _particle_positions(self):
        return np.array(pyflex.get_positions()).reshape(-1, 4)[:, :3].astype(np.float32)

    def _plan_paths(self):
        pos = self._particle_positions()
        n = pos.shape[0]
        goal = np.asarray(self.env.current_config["goal_character_pos"], dtype=np.float32)[:, :3]
        idx = np.linspace(0, goal.shape[0] - 1, n).round().astype(np.int32)
        goal = goal[idx].copy()
        goal[:, 1] = self.trace_height

        forward_cost = np.linalg.norm(pos[0] - goal[0]) + np.linalg.norm(pos[-1] - goal[-1])
        reverse_cost = np.linalg.norm(pos[0] - goal[-1]) + np.linalg.norm(pos[-1] - goal[0])
        if reverse_cost < forward_cost:
            goal = goal[::-1].copy()

        goal_char = str(self.env.current_config.get("goal_character", ""))
        if goal_char == "C":
            self.path0 = goal
            self.path1 = np.repeat(goal[0][None], goal.shape[0], axis=0)
            return

        mid = n // 2
        self.path0 = goal[:mid + 1]
        self.path1 = goal[:mid - 1:-1]
        steps = max(len(self.path0), len(self.path1))
        if len(self.path0) < steps:
            self.path0 = np.vstack(
                [self.path0, np.repeat(self.path0[-1][None], steps - len(self.path0), axis=0)]
            )
        if len(self.path1) < steps:
            self.path1 = np.vstack(
                [self.path1, np.repeat(self.path1[-1][None], steps - len(self.path1), axis=0)]
            )

    def _enter_approach(self):
        self._refresh_endpoint_targets(above=True)
        self.phase = self.APPROACH
        self.phase_step = 0
        self.grip[:] = 0.0

    def _refresh_endpoint_targets(self, above):
        pos = self._particle_positions()
        lift = self.lift_height if above else 0.005
        self.targets = np.stack([pos[0].copy(), pos[-1].copy()], axis=0)
        self.targets[:, 1] += lift

    def _set_trace_targets(self):
        idx = min(self.path_idx, len(self.path0) - 1)
        self.targets = np.stack([self.path0[idx], self.path1[idx]], axis=0).astype(np.float32)

    def _advance_phase(self, picker_xyz):
        if self.phase == self.SETTLE:
            self._enter_approach()
        elif self.phase == self.APPROACH:
            self._refresh_endpoint_targets(above=False)
            self.phase = self.DESCEND
        elif self.phase == self.DESCEND:
            self.targets = picker_xyz.copy()
            self.grip[:] = 1.0
            self.phase = self.GRIP
        elif self.phase == self.GRIP:
            self.path_idx = 0
            self._set_trace_targets()
            self.phase = self.TRACE
        elif self.phase == self.TRACE:
            self.path_idx += 1
            if self.path_idx >= len(self.path0):
                self.targets = picker_xyz.copy()
                self.phase = self.HOLD
            else:
                self._set_trace_targets()
        else:
            self.targets = picker_xyz.copy()
        self.phase_step = 0


class HybridRopeConfigurationPolicy(object):
    """Choose the strongest available RopeConfiguration oracle by goal letter."""

    def __init__(self, env, action_space, num_picker=2, action_gain=1.0,
                 max_speed_scale=1.0, deadband=1e-5, lift_height=0.08,
                 rng=None, action_repeat=8):
        self.env = env
        self.goal_policy = GoalRopeConfigurationPolicy(
            env=env,
            action_space=action_space,
            num_picker=num_picker,
            action_gain=action_gain,
            max_speed_scale=max_speed_scale,
            deadband=deadband,
            lift_height=lift_height,
            rng=rng,
            action_repeat=action_repeat,
        )
        self.trace_policy = TraceRopeConfigurationPolicy(
            env=env,
            action_space=action_space,
            num_picker=num_picker,
            action_gain=action_gain,
            max_speed_scale=max_speed_scale,
            deadband=deadband,
            lift_height=lift_height,
            rng=rng,
            action_repeat=action_repeat,
        )
        self.active = self.goal_policy
        self.goal_char = ""
        self.refine_after_trace = False

    def reset(self):
        self.goal_char = str(self.env.current_config.get("goal_character", ""))
        self.refine_after_trace = False
        if self.goal_char in ("C", "O"):
            self.active = self.trace_policy
            self.refine_after_trace = True
        else:
            self.active = self.goal_policy
        self.active.reset()

    def get_action(self):
        if self.refine_after_trace and self.active is self.trace_policy:
            if getattr(self.trace_policy, "phase", None) == self.trace_policy.HOLD:
                info = self.env._get_info()
                perf = float(info.get("normalized_performance", info.get("performance", -1e9)))
                if perf < 0.60:
                    self.goal_policy.reset()
                    self.active = self.goal_policy
                self.refine_after_trace = False
        return self.active.get_action()


# Factory used by collect_trajectories.py
def make_policy(env_name, env, num_picker, noise_scale=0.02, lateral_scale=1.0,
                lift_height=0.05, action_gain=0.3, max_speed_scale=0.35,
                deadband=5e-4, kind="geometric", num_waypoints=3, rng=None):
    """Return a policy object for the given env_name + kind.

    kind=
      "geometric"   -> GeometricRopeFlattenPolicy (grab endpoints, stretch).
      "manipulate"  -> SmoothRopeManipulationPolicy (random middle particles,
                       smooth waypoint walk; creates diverse rope configs).
      "configure"   -> goal-conditioned RopeConfiguration oracle.
    """
    if env_name not in ("RopeFlatten", "RopeConfiguration"):
        raise NotImplementedError(
            "Scripted policy not implemented for env_name=%r (yet)" % env_name
        )
    if env_name != "RopeFlatten" and kind == "geometric":
        raise NotImplementedError(
            "geometric endpoint-stretch policy is only implemented for RopeFlatten"
        )
    if kind == "geometric":
        return GeometricRopeFlattenPolicy(
            action_space=env.action_space,
            num_picker=num_picker,
            noise_scale=noise_scale,
            lateral_scale=lateral_scale,
            lift_height=lift_height,
            action_gain=action_gain,
            max_speed_scale=max_speed_scale,
            deadband=deadband,
            rng=rng,
            action_repeat=getattr(env, "action_repeat", 8),
        )
    if kind == "manipulate":
        return SmoothRopeManipulationPolicy(
            action_space=env.action_space,
            num_picker=num_picker,
            noise_scale=noise_scale,
            action_gain=action_gain,
            max_speed_scale=max_speed_scale,
            deadband=deadband,
            lift_height=max(lift_height, 0.08),
            num_waypoints=num_waypoints,
            rng=rng,
            action_repeat=getattr(env, "action_repeat", 8),
        )
    if kind == "configure":
        if env_name != "RopeConfiguration":
            raise NotImplementedError("configure policy is only for RopeConfiguration")
        return HybridRopeConfigurationPolicy(
            env=env,
            action_space=env.action_space,
            num_picker=num_picker,
            action_gain=max(action_gain, 1.0),
            max_speed_scale=max(max_speed_scale, 1.0),
            deadband=min(deadband, 1e-5),
            lift_height=max(lift_height, 0.08),
            rng=rng,
            action_repeat=getattr(env, "action_repeat", 8),
        )
    if kind == "push":
        return PushRopePolicy(
            action_space=env.action_space,
            num_picker=num_picker,
            noise_scale=noise_scale,
            action_gain=action_gain,
            max_speed_scale=max_speed_scale,
            deadband=deadband,
            num_waypoints=max(num_waypoints, 4),
            rng=rng,
            action_repeat=getattr(env, "action_repeat", 8),
        )
    raise ValueError("unknown policy kind: %r" % kind)
