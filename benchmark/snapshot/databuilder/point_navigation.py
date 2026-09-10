"""Point-to-point path planning and following for Habitat-Sim.

The public API in this module intentionally accepts ordinary ``[x, y, z]``
coordinates.  It wraps Habitat-Sim's lower-level navmesh path finder and
greedy action follower so callers do not need to manage either directly.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import habitat_sim
import numpy as np


class PointNavigationError(RuntimeError):
    """Base error raised by the point-navigation API."""


class InvalidGoalError(PointNavigationError):
    """Raised when a requested goal cannot be placed on the navmesh."""


class UnreachableGoalError(PointNavigationError):
    """Raised when no navmesh path exists from the agent to the goal."""


@dataclass(frozen=True)
class NavigationPlan:
    """A shortest path from the agent's current position to a goal."""

    requested_goal: np.ndarray
    goal: np.ndarray
    snapped_distance: float
    geodesic_distance: float
    waypoints: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class NavigationResult:
    """Summary returned after executing navigation actions."""

    success: bool
    reason: str
    start: np.ndarray
    goal: np.ndarray
    final_position: np.ndarray
    initial_geodesic_distance: float
    final_euclidean_distance: float
    actions: Tuple[Any, ...]

    @property
    def step_count(self) -> int:
        return len(self.actions)


StepCallback = Callable[[int, Any, Mapping[str, Any]], None]


class PointNavigator:
    """Plan and execute navigation to coordinates in a Habitat scene.

    The simulator must have a loaded navmesh.  The configured agent must have
    ``move_forward``, ``turn_left`` and ``turn_right`` actions because those
    are the motion primitives used by ``GreedyGeodesicFollower``.
    """

    def __init__(
        self,
        simulator: habitat_sim.Simulator,
        *,
        agent_id: int = 0,
        goal_radius: float = 0.25,
        max_snap_distance: Optional[float] = 1.0,
    ) -> None:
        if not simulator.pathfinder.is_loaded:
            raise PointNavigationError(
                "The simulator has no loaded navmesh; load or recompute one first"
            )
        if goal_radius <= 0:
            raise ValueError("goal_radius must be positive")
        if max_snap_distance is not None and max_snap_distance < 0:
            raise ValueError("max_snap_distance cannot be negative")

        self.simulator = simulator
        self.agent_id = agent_id
        self.agent = simulator.get_agent(agent_id)
        self.goal_radius = float(goal_radius)
        self.max_snap_distance = max_snap_distance

        action_names = set(self.agent.agent_config.action_space)
        required_actions = {"move_forward", "turn_left", "turn_right"}
        missing_actions = required_actions - action_names
        if missing_actions:
            missing = ", ".join(sorted(missing_actions))
            raise PointNavigationError(
                f"Agent {agent_id} is missing required actions: {missing}"
            )

    @staticmethod
    def _coordinate(value: Sequence[float], name: str) -> np.ndarray:
        coordinate = np.asarray(value, dtype=np.float32)
        if coordinate.shape != (3,):
            raise ValueError(f"{name} must contain exactly three values: [x, y, z]")
        if not np.all(np.isfinite(coordinate)):
            raise ValueError(f"{name} must contain only finite values")
        return coordinate

    def snap_goal(self, goal: Sequence[float]) -> Tuple[np.ndarray, float]:
        """Snap a requested coordinate to the closest navigable point."""

        requested_goal = self._coordinate(goal, "goal")
        snapped_goal = np.asarray(
            self.simulator.pathfinder.snap_point(requested_goal), dtype=np.float32
        )
        if snapped_goal.shape != (3,) or not np.all(np.isfinite(snapped_goal)):
            raise InvalidGoalError(
                f"No navigable location was found near {requested_goal.tolist()}"
            )

        snapped_distance = float(np.linalg.norm(snapped_goal - requested_goal))
        if (
            self.max_snap_distance is not None
            and snapped_distance > self.max_snap_distance
        ):
            raise InvalidGoalError(
                "The closest navigable location is "
                f"{snapped_distance:.3f} m from the requested goal, exceeding "
                f"max_snap_distance={self.max_snap_distance:.3f} m"
            )
        return snapped_goal, snapped_distance

    def plan(self, goal: Sequence[float]) -> NavigationPlan:
        """Return shortest-path waypoints from the current pose to ``goal``."""

        requested_goal = self._coordinate(goal, "goal")
        snapped_goal, snapped_distance = self.snap_goal(requested_goal)
        start = np.asarray(self.agent.get_state().position, dtype=np.float32)

        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = snapped_goal
        if not self.simulator.pathfinder.find_path(path):
            raise UnreachableGoalError(
                "No path exists from "
                f"{start.tolist()} to {snapped_goal.tolist()}; the points may be "
                "on disconnected navmesh islands"
            )

        return NavigationPlan(
            requested_goal=requested_goal.copy(),
            goal=snapped_goal.copy(),
            snapped_distance=snapped_distance,
            geodesic_distance=float(path.geodesic_distance),
            waypoints=tuple(
                np.asarray(point, dtype=np.float32).copy() for point in path.points
            ),
        )

    def follow(
        self,
        goal: Sequence[float],
        *,
        max_steps: int = 1_000,
        on_step: Optional[StepCallback] = None,
    ) -> NavigationResult:
        """Plan to ``goal`` and move the simulated agent until it arrives.

        A new action is selected from the agent's current pose after every
        simulator step.  This is safer than executing a precomputed action list
        and also supports actuation noise better.
        """

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        plan = self.plan(goal)
        start = np.asarray(self.agent.get_state().position, dtype=np.float32).copy()
        actions = []
        follower = habitat_sim.nav.GreedyGeodesicFollower(
            self.simulator.pathfinder,
            self.agent,
            goal_radius=self.goal_radius,
        )

        success = False
        reason = "maximum step count reached"
        for step_index in range(max_steps):
            try:
                action = follower.next_action_along(plan.goal)
            except habitat_sim.errors.GreedyFollowerError:
                reason = "Habitat's greedy follower could not select a valid action"
                break

            if action is None:
                success = True
                reason = "goal reached"
                break

            observations_by_agent = self.simulator.step({self.agent_id: action})
            observations = observations_by_agent[self.agent_id]
            actions.append(action)
            if on_step is not None:
                on_step(step_index, action, observations)

        final_position = np.asarray(
            self.agent.get_state().position, dtype=np.float32
        ).copy()
        final_distance = float(np.linalg.norm(final_position - plan.goal))
        if not success and final_distance <= self.goal_radius:
            success = True
            reason = "goal reached"
        return NavigationResult(
            success=success,
            reason=reason,
            start=start,
            goal=plan.goal.copy(),
            final_position=final_position,
            initial_geodesic_distance=plan.geodesic_distance,
            final_euclidean_distance=final_distance,
            actions=tuple(actions),
        )
