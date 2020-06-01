#!/usr/bin/env python3

import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import yaml

import geometry
from aido_schemas import (Context, DB18RobotObservations, DB18SetRobotCommands, Duckiebot1Observations, EpisodeStart,
                          GetRobotObservations, GetRobotState, JPGImage, Metric, PerformanceMetrics,
                          protocol_simulator_duckiebot1, PWMCommands, RobotConfiguration, RobotInterfaceDescription,
                          RobotName, RobotPerformance, RobotState, SetMap, SimulationState, SpawnRobot, StateDump, Step,
                          wrap_direct)
from duckietown_world.world_duckietown.pwm_dynamics import get_DB18_nominal
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.objects import DuckiebotObj
from gym_duckietown.simulator import (NotInLane, ObjMesh, ROBOT_LENGTH, ROBOT_WIDTH, SAFETY_RAD_MULT, Simulator,
                                      WHEEL_DIST)
from zuper_nodes import TimeSpec, timestamp_from_seconds, TimingInfo


@dataclass
class MyRobotInfo:
    pose: np.ndarray
    velocity: np.ndarray
    last_action: np.ndarray
    wheels_velocities: np.ndarray


@dataclass
class MyRobotState(RobotState):
    robot_name: RobotName
    t_effective: float
    state: MyRobotInfo


@dataclass
class GymDuckiebotSimulatorConfig:
    """
        env_constructor: either "Simulator" or "DuckietownEnv"

        env_parameters: parameters for the constructor

        camera_frame_rate: frame rate for the camera. No observations
        will be generated quicker than this.

    """
    env_constructor: str = 'Simulator'
    env_parameters: dict = None
    camera_dt: float = 1 / 15.0
    render_dt: float = 1 / (15.0 * 7)
    minimum_physics_dt: float = 1 / 30.0
    blur_time: float = 0.05


class GymDuckiebotSimulator:
    config: GymDuckiebotSimulatorConfig = GymDuckiebotSimulatorConfig()

    current_time: float
    reward_cumulative: float
    episode_name: str
    env: Simulator

    robot_name: Optional[RobotName]
    spawn_pose: Any
    npcs: Dict[RobotName, DuckiebotObj]
    spawn_configuration: RobotConfiguration

    last_commands: np.array
    last_action: np.array  # 2
    last_render_time: float
    render_timestamps: List[float]
    render_observations: List[np.array]
    last_observations: np.array
    last_observations_time: float
    ro: DB18RobotObservations

    def init(self):
        env_parameters = self.config.env_parameters or {}
        environment_class = self.config.env_constructor

        name2class = {
            'DuckietownEnv': DuckietownEnv,
            'Simulator': Simulator,
        }
        if not environment_class in name2class:
            msg = 'Could not find environment class {} in {}'.format(environment_class, list(name2class))
            raise Exception(msg)

        klass = name2class[environment_class]
        self.env = klass(**env_parameters)

    def on_received_seed(self, context: Context, data: int):
        random.seed(data)
        np.random.seed(data)

    def on_received_clear(self):
        self.robot_name = None
        self.spawn_pose = None
        self.npcs = {}

    def on_received_set_map(self, data: SetMap):
        yaml_str: str = data.map_data

        map_data = yaml.load(yaml_str, Loader=yaml.SafeLoader)

        self.env._interpret_map(map_data)

    def on_received_spawn_robot(self, data: SpawnRobot):
        if data.playable:
            self.spawn_configuration = data.configuration
            self.robot_name = data.robot_name
        else:
            q = data.configuration.pose
            pos, angle = self.env.weird_from_cartesian(q)

            mesh = ObjMesh.get('duckiebot')

            obj_desc = {
                'kind': 'duckiebot',
                'mesh': mesh,
                'pos': pos,
                'rotate': np.rad2deg(angle),
                'height': 0.12,
                'y_rot': np.rad2deg(angle),
                'static': False,
                'optional': False
            }

            obj_desc['scale'] = obj_desc['height'] / mesh.max_coords[1]

            obj = DuckiebotObj(obj_desc, False, SAFETY_RAD_MULT, WHEEL_DIST,
                               ROBOT_WIDTH, ROBOT_LENGTH)

            self.npcs[data.robot_name] = obj

    def _set_pose(self, context):
        # TODO: check location
        e0 = self.env

        q = self.spawn_configuration.pose
        v = self.spawn_configuration.velocity
        c0 = q, v

        p = get_DB18_nominal(delay=0.15)
        self.state = p.initialize(c0=c0, t0=0)
        cur_pos, cur_angle = e0.weird_from_cartesian(q)
        q2 = e0.cartesian_from_weird(cur_pos, cur_angle)
        e0.cur_pos = cur_pos
        e0.cur_angle = cur_angle

        i, j = e0.get_grid_coords(e0.cur_pos)
        tile = e0._get_tile(i, j)

        msg = ''
        msg += f'\ni, j: {i}, {j}'
        msg += '\nPose: %s' % geometry.SE2.friendly(q)
        msg += '\nPose: %s' % geometry.SE2.friendly(q2)
        msg += '\nCur pos: %s' % cur_pos
        context.info(msg)

        if tile is None:
            msg = 'Current pose is not in a tile.'

            raise Exception(msg)

        kind = tile['kind']
        # angle = tile['angle']

        is_straight = kind.startswith('straight')

        context.info('Sampled tile  %s %s %s' % (tile['coords'], tile['kind'], tile['angle']))

        if not is_straight:
            context.info('not on a straight tile')

        valid = e0._valid_pose(cur_pos, cur_angle)
        context.info('valid: %s' % valid)

        try:
            lp = e0.get_lane_pos2(e0.cur_pos, e0.cur_angle)
            context.info('Sampled lane pose %s' % str(lp))
            context.info('dist: %s' % lp.dist)
        except NotInLane:
            raise

        if not valid:
            msg = 'Not valid'
            context.error(msg)

    def on_received_get_robot_interface_description(self, context: Context, data: RobotName):
        rid = RobotInterfaceDescription(robot_name=data, observations=JPGImage, commands=PWMCommands)
        context.write('robot_interface_description', rid)

    def on_received_get_robot_performance(self, context: Context, data: RobotName):
        metrics = {}
        metrics['survival_time'] = Metric(higher_is_better=True, cumulative_value=self.current_time,
                                          description="Survival time.")
        metrics['reward'] = Metric(higher_is_better=True, cumulative_value=self.reward_cumulative,
                                   description="Cumulative agent reward.")
        pm = PerformanceMetrics(metrics)
        rid = RobotPerformance(robot_name=data, t_effective=self.current_time, performance=pm)
        context.write('robot_performance', rid)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        self.current_time = 0.0
        self.reward_cumulative = 0.0
        self.episode_name = data.episode_name
        self.last_action = np.array([0.0, 0.0])
        try:
            self.env.reset()
        except BaseException as e:
            msg = 'Could not initialize environment'
            raise Exception(msg) from e

        self._set_pose(context)
        for robot_name, obj in self.npcs.items():
            self.env.objects.append(obj)

        self.last_observations_time = -100.0
        self.last_render_time = -100.0

        self.render_timestamps = []
        self.render_observations = []
        self.last_observations = None

        self.render(context)
        self.update_observations(self.config.blur_time, context)

    def on_received_step(self, context: Context, data: Step):
        delta_time = data.until - self.current_time
        if delta_time > 0:
            self.update_physics_and_observations(until=data.until, context=context)
        else:
            context.warning(f'Already at time {data.until}')

        d = self.env._compute_done_reward()
        self.reward_cumulative += d.reward * delta_time
        self.current_time = data.until

    def update_physics_and_observations(self, until, context: Context):
        # we are at self.current_time and need to update until "until"
        sensor_dt = self.config.camera_dt
        render_dt = self.config.render_dt
        snapshots = list(get_snapshots(self.last_observations_time, self.current_time, until, render_dt))

        steps = snapshots + [until]
        # context.info(f'current time: {self.current_time}')
        # context.info(f'       until: {until}')
        # context.info(f'    last_obs: {self.last_observations_time}')
        # context.info(f'   snapshots: {snapshots}')

        for t1 in steps:
            delta_time = t1 - self.current_time

            last_action = np.array([0.0, 0.0])
            self.env.update_physics(last_action, delta_time=delta_time)

            self.state = self.state.integrate(delta_time, self.last_commands.wheels)
            q = self.state.TSE2_from_state()[0]
            cur_pos, cur_angle = self.env.weird_from_cartesian(q)
            self.env.cur_pos = cur_pos
            self.env.cur_angle = cur_angle

            self.current_time = t1

            if self.current_time - self.last_render_time > render_dt:
                self.render(context)
            if self.current_time - self.last_observations_time >= sensor_dt:
                self.update_observations(self.config.blur_time, context)

    def render(self, context: Context):
        # context.info(f'render() at {self.current_time}')

        with timeit('render_obs()', context, enabled=False):
            obs = self.env.render_obs()
        # context.info(f'render {obs.shape} {obs.dtype}')
        self.render_observations.append(obs)
        self.render_timestamps.append(self.current_time)
        self.last_render_time = self.current_time

    def update_observations(self, blur_time: float, context: Context):
        context.info(f'update_observations() at {self.current_time}')
        assert self.render_observations

        to_average = []
        n = len(self.render_observations)
        # context.info(str(self.render_timestamps))
        # context.info(f'now {self.current_time}')
        for i in range(n):
            ti = self.render_timestamps[i]
            if math.fabs(self.current_time - ti) < blur_time:
                to_average.append(self.render_observations[i])

        obs0 = to_average[0].astype('float32')
        context.info(str(obs0.shape))
        for obs in to_average[1:]:
            obs0 += obs
        obs = obs0 / len(to_average)
        obs = obs.astype('uint8')
        # context.info(f'update {obs.shape} {obs.dtype}')
        jpg_data = rgb2jpg(obs)
        camera = JPGImage(jpg_data)
        obs = Duckiebot1Observations(camera)
        self.ro = DB18RobotObservations(self.robot_name, self.current_time, obs)
        self.last_observations_time = self.current_time

    def on_received_set_robot_commands(self, data: DB18SetRobotCommands, context: Context):
        l, r = data.commands.wheels.motor_left, data.commands.wheels.motor_right

        if max(math.fabs(l), math.fabs(r)) > 1:
            msg = f'Received invalid PWM commands. They should be between -1 and +1.' \
                  f' Received left = {l!r}, right = {r!r}.'
            context.error(msg)
            raise Exception(msg)
        self.last_commands = data.commands

    def on_received_get_robot_observations(self, context: Context, data: GetRobotObservations):
        _ = data
        # timing information
        t = timestamp_from_seconds(self.last_observations_time)
        ts = TimeSpec(time=t, frame=self.episode_name, clock=context.get_hostname())
        timing = TimingInfo(acquired={'image': ts})
        context.write('robot_observations', self.ro, with_schema=True, timing=timing)

    def on_received_get_robot_state(self, context: Context, data: GetRobotState):
        robot_name = data.robot_name
        env = self.env
        speed = env.speed
        omega = 0.0  # XXX
        if robot_name == self.robot_name:
            q = env.cartesian_from_weird(env.cur_pos, env.cur_angle)
            v = geometry.se2_from_linear_angular([speed, 0], omega)
            state = MyRobotInfo(pose=q,
                                velocity=v,
                                last_action=env.last_action,
                                wheels_velocities=env.wheelVels)
            rs = MyRobotState(robot_name=robot_name,
                              t_effective=self.current_time,
                              state=state)
        else:
            obj: DuckiebotObj = self.npcs[robot_name]
            q = env.cartesian_from_weird(obj.pos, obj.angle)
            # FIXME: how to get velocity?
            v = geometry.se2_from_linear_angular([0, 0], 0)
            state = MyRobotInfo(pose=q,
                                velocity=v,
                                last_action=np.array([0, 0]),
                                wheels_velocities=np.array([0, 0]))
            rs = MyRobotState(robot_name=robot_name,
                              t_effective=self.current_time,
                              state=state)
        # timing information
        t = timestamp_from_seconds(self.current_time)
        ts = TimeSpec(time=t, frame=self.episode_name, clock=context.get_hostname())
        timing = TimingInfo(acquired={'state': ts})
        context.write('robot_state', rs, timing=timing)  # , with_schema=True)

    def on_received_dump_state(self, context: Context):
        context.write('dump_state', StateDump(None))

    def on_received_get_sim_state(self, context: Context):
        d = self.env._compute_done_reward()
        done = d.done
        done_why = d.done_why
        done_code = d.done_code
        sim_state = SimulationState(done, done_why, done_code)
        context.write('sim_state', sim_state)


def get_snapshots(last_obs_time: float, current_time: float, until: float, dt: float) -> Iterator[float]:
    t = last_obs_time + dt
    while t < until:
        if t > current_time:
            yield t
        t += dt


# noinspection PyUnresolvedReferences
def rgb2jpg(rgb: np.ndarray) -> bytes:
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    compress = cv2.imencode('.jpg', bgr)[1]
    jpg_data = np.array(compress).tostring()
    return jpg_data


@contextmanager
def timeit(s, context, min_warn=0.01, enabled=True):
    t0 = time.time()
    yield

    if not enabled:
        return
    t1 = time.time()

    delta = t1 - t0
    msg = 'timeit: %d ms for %s' % ((t1 - t0) * 1000, s)
    if delta > min_warn:
        context.info(msg)


def main():
    node = GymDuckiebotSimulator()
    protocol = protocol_simulator_duckiebot1
    protocol.outputs['robot_state'] = MyRobotState
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
