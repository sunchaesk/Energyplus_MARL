# pip installed ray, tabulate, tree
# pip uninstalled ''
import argparse
import sys
import os
import threading
from tempfile import TemporaryDirectory
from typing import Dict, Any, Tuple, Optional, List
from queue import Queue, Empty, Full
from datetime import date
from datetime import datetime
import math
import time

import random
import pprint

import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

sys.path.insert(0, '/home/ck/Downloads/Eplus/')
from pyenergyplus.api import EnergyPlusAPI
from pyenergyplus.datatransfer import DataExchange

def _add_10_minutes(inp):
    year, month, day, hour, minute = inp

    # Calculate the total number of minutes
    total_minutes = (hour * 60) + minute + 10

    # Calculate the new hour and minute values
    new_hour = total_minutes // 60
    new_minute = total_minutes % 60

    # Handle hour and day overflow
    if new_hour >= 24:
        new_hour %= 24
        day += 1

    # Handle month and year overflow
    if month in [1, 3, 5, 7, 8, 10, 12] and day > 31:
        day = 1
        month += 1
    elif month in [4, 6, 9, 11] and day > 30:
        day = 1
        month += 1
    elif month == 2:
        if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
            if day > 29:
                day = 1
                month += 1
        else:
            if day > 28:
                day = 1
                month += 1

    # Handle minute overflow and represent 0 minutes as 60
    if new_minute == 0:
        new_minute = 60
        new_hour -= 1

    if new_hour == -1:
        new_hour = 23

    return (year, month, day, new_hour, new_minute)

def _get_cost_signal(day_of_week, hour, minute):
    '''get cost signal at given time. @param: minute is not used'''
    if day_of_week in [1, 7]:
        # weekend pricing
        if hour in range(0, 7) or hour in range(23, 24 + 1): # plus one is to include 7
            #self.next_obs['cost_rate'] = 2.4
            return 2.4
        elif hour in range(7, 23):
            #self.next_obs['cost_rate'] = 7.4
            return 7.4
    else:
        if hour in range(0, 7) or hour in range(23, 24 + 1):
            #self.next_obs['cost_rate'] = 2.4
            return 2.4
        elif hour in range(7, 16) or hour in range(21, 23):
            #self.next_obs['cost_rate'] = 10.2
            return 10.2
        elif hour in range(16, 21):
            #self.next_obs['cost_rate'] = 24.0
            return 24.0



class EnergyPlusRunner:

    def __init__(self, episode: int, env_config: Dict[str, Any], obs_queue: Queue, act_queue: Queue) -> None:
        pp = pprint.PrettyPrinter(indent=4)
        self.episode = episode
        self.env_config = env_config
        self.obs_queue = obs_queue
        self.act_queue = act_queue

        self.energyplus_api = EnergyPlusAPI()
        self.x: DataExchange = self.energyplus_api.exchange
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: Any = None
        self.sim_results: Dict[str, Any] = {}
        self.initialized = False
        self.warmup_complete = False
        self.warmup_queue = Queue()
        self.progress_value: int = 0
        self.simulation_complete = False

        # request variables to be available during runtime
        self.request_variable_complete = False

        # below is declaration of variables, meters and actuators
        # this simulation will interact with
        self.variables = {
            "outdoor_temp" : ("Site Outdoor Air Drybulb Temperature", "Environment"),
            "indoor_temp_living" : ("Zone Air Temperature", 'living_unit1'),
            'sky_diffuse_solar_ldf': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", 'Window_ldf_1.unit1'),
            'sky_diffuse_solar_ldb': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", 'Window_ldb_1.unit1'),
            'sky_diffuse_solar_sdr': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", 'Window_sdr_1.unit1'),
            'sky_diffuse_solar_sdl': ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", 'Window_sdl_1.unit1'),
            'site_direct_solar': ("Site Direct Solar Radiation Rate per Area", "Environment"),
            'site_horizontal_infrared': ("Site Horizontal Infrared Radiation Rate per Area", "Environment"),
            'outdoor_relative_humidity': ("Site Outdoor Air Relative Humidity", "Environment")
        }
        self.var_handles: Dict[str, int] = {}

        self.meters = {
            "elec_cooling": "Cooling:Electricity"
        }
        self.meter_handles: Dict[str, int] = {}

        self.actuators = {
            # supply air temperature setpoint (°C)
            "cooling_actuator_living" : (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "living_unit1"
            ),

            "heating_actuator_living" : (
                "Zone Temperature Control",
                "Heating Setpoint",
                "living_unit1"
            )
        }
        self.actuator_handles: Dict[str, int] = {}

    def start(self) -> None:
        self.energyplus_state = self.energyplus_api.state_manager.new_state()
        runtime = self.energyplus_api.runtime

        if not self.request_variable_complete:
            for key, var in self.variables.items():
                self.x.request_variable(self.energyplus_state, var[0], var[1])
                self.request_variable_complete = True

        # register callback used to track simulation progress
        def _report_progress(progress: int) -> None:
            self.progress_value = progress
            print(f"Simulation progress: {self.progress_value}%")

        runtime.callback_progress(self.energyplus_state, _report_progress)

        def _warmup_complete(state: Any) -> None:
            self.warmup_complete = True
            self.warmup_queue.put(True)

        # register callback used to signal warmup complete
        runtime.callback_after_new_environment_warmup_complete(self.energyplus_state, _warmup_complete)

        # register callback used to collect observations
        runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)

        # register callback used to send actions
        runtime.callback_after_predictor_after_hvac_managers(self.energyplus_state, self._send_actions)

        # run EnergyPlus in a non-blocking way
        def _run_energyplus(runtime, cmd_args, state, results):
            print(f"running EnergyPlus with args: {cmd_args}")

            # start simulation
            results["exit_code"] = runtime.run_energyplus(state, cmd_args)
            self.simulation_complete = True

        self.energyplus_exec_thread = threading.Thread(
            target=_run_energyplus,
            args=(
                self.energyplus_api.runtime,
                self.make_eplus_args(),
                self.energyplus_state,
                self.sim_results
            )
        )
        self.energyplus_exec_thread.start()

    def stop(self) -> None:
        if self.energyplus_exec_thread:
            self.simulation_complete = True
            self._flush_queues()
            self.energyplus_exec_thread.join()
            self.energyplus_exec_thread = None
            self.energyplus_api.runtime.clear_callbacks()
            self.energyplus_api.state_manager.delete_state(self.energyplus_state)

    def failed(self) -> bool:
        return self.sim_results.get("exit_code", -1) > 0

    def make_eplus_args(self) -> List[str]:
        """
        make command line arguments to pass to EnergyPlus
        """
        eplus_args = ["-r"] if self.env_config.get("csv", False) else []
        eplus_args += ["-x"]
        eplus_args += [
            "-w",
            self.env_config["epw"],
            "-d",
            f"{self.env_config['output']}/episode-{self.episode:08}-{os.getpid():05}",
            self.env_config["idf"]
        ]
        return eplus_args

    def _collect_obs(self, state_argument) -> None:
        """
        EnergyPlus callback that collects output variables/meters
        values and enqueue them
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        self.next_obs = {
            **{
                key: self.x.get_variable_value(state_argument, handle)
                for key, handle
                in self.var_handles.items()
            },
            # **{
            #     key: self.x.get_meter_value(state_argument, handle)
            #     for key, handle
            #     in self.meter_handles.items()
            # }
        }

        # setup simulation times
        year = self.x.year(self.energyplus_state)
        month = self.x.month(self.energyplus_state)
        day = self.x.day_of_month(self.energyplus_state)
        hour = self.x.hour(self.energyplus_state)
        minute = self.x.minutes(self.energyplus_state)
        day_of_week = self.x.day_of_week(self.energyplus_state)

        # decompose hour of week to day_of_week & hour
        self.next_obs['year'] = year
        self.next_obs['hour'] = hour
        self.next_obs['day_of_week'] = day_of_week
        self.next_obs['day'] = day
        self.next_obs['month'] = month



        # prev reward
        # self.next_obs['prev_cost'] = self.prev_cost
        # prev actuator value
        # self.next_obs['prev_actuator_value'] = self.prev_actuator_value

        # hour of week observation value
        b_hour_of_week_obs = False
        if b_hour_of_week_obs:
            hour_of_week = (24 * (day_of_week - 1)) + hour
            self.next_obs['hour_of_week'] = hour_of_week

        # cost_rate_signal
        cost_rate = False
        if cost_rate:
            if day_of_week in [1, 7]:
                # weekend pricing
                if hour in range(0, 7) or hour in range(23, 24 + 1): # plus one is to include 7
                    self.next_obs['cost_rate'] = 2.4
                elif hour in range(7, 23):
                    self.next_obs['cost_rate'] = 7.4
            else:
                if hour in range(0, 7) or hour in range(23, 24 + 1):
                    self.next_obs['cost_rate'] = 2.4
                elif hour in range(7, 16) or hour in range(21, 23):
                    self.next_obs['cost_rate'] = 10.2
                elif hour in range(16, 21):
                    self.next_obs['cost_rate'] = 24.0


        actuator = self.x.get_actuator_value(self.energyplus_state, self.actuator_handles['cooling_actuator_living'])

        # cost calc is based on
        # energy: joules
        # cost rate: cents

        # 13th: next time step cost signal
        next_time_step = _add_10_minutes(tuple([
            year,
            month,
            day,
            hour,
            minute
        ]))
        x_date = date(next_time_step[0], next_time_step[1], next_time_step[2])
        no = x_date.weekday()
        if no < 5:
            next_time_step_day_of_week = 2 # any weekday is fine
        else:
            next_time_step_day_of_week = 1 # any weekend [1,7] is fine
        next_timestep_cost_signal = _get_cost_signal(next_time_step_day_of_week,
                                                     next_time_step[3],
                                                     next_time_step[4])
        # self.next_obs['next_timestep_cost_signal'] = next_timestep_cost_signal

        # print('----------------------------')
        # print(self.next_obs)
        # print('LEN:', len(self.next_obs))
        # print('----------------------------') #
        self.obs_queue.put(self.next_obs)

    def _send_actions(self, state_argument):
        """
        EnergyPlus callback that sets actuator value from last decided action
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        if self.act_queue.empty():
            return

        next_action = self.act_queue.get()
        assert isinstance(next_action, float)

        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles["cooling_actuator_living"],
            actuator_value=next_action
        )
        self.x.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles["heating_actuator_living"],
            actuator_value=0
        )

    def _init_callback(self, state_argument) -> bool:
        """initialize EnergyPlus handles and checks if simulation runtime is ready"""
        self.initialized = self._init_handles(state_argument) \
            and not self.x.warmup_flag(state_argument)
        return self.initialized

    def _init_handles(self, state_argument):
        """initialize sensors/actuators handles to interact with during simulation"""
        if not self.initialized:
            if not self.x.api_data_fully_ready(state_argument):
                return False

            self.var_handles = {
                key: self.x.get_variable_handle(state_argument, *var)
                for key, var in self.variables.items()
            }

            self.meter_handles = {
                key: self.x.get_meter_handle(state_argument, meter)
                for key, meter in self.meters.items()
            }

            self.actuator_handles = {
                key: self.x.get_actuator_handle(state_argument, *actuator)
                for key, actuator in self.actuators.items()
            }

            for handles in [
                self.var_handles,
                self.meter_handles,
                self.actuator_handles
            ]:
                if any([v == -1 for v in handles.values()]):
                    available_data = self.x.list_available_api_data_csv(state_argument).decode('utf-8')
                    print(
                        f"got -1 handle, check your var/meter/actuator names:\n"
                        f"> variables: {self.var_handles}\n"
                        f"> meters: {self.meter_handles}\n"
                        f"> actuators: {self.actuator_handles}\n"
                        f"> available E+ API data: {available_data}"
                    )
                    exit(1)

            self.initialized = True

        return True

    def _flush_queues(self):
        for q in [self.obs_queue, self.act_queue]:
            while not q.empty():
                q.get()


class EnergyPlusEnv(gym.Env):

    def __init__(self, env_config: Dict[str, Any]):
        self.env_config = env_config
        self.episode = -1
        self.timestep = 0

        obs_len = 38
        low_obs = np.array(
            [-1e8] * obs_len
        )
        hig_obs = np.array(
            [1e8] * obs_len
        )
        self.observation_space = gym.spaces.Box(
            low=low_obs, high=hig_obs, dtype=np.float32
        )
        self.last_obs = {}

        # action space: supply air temperature (100 possible values)
        self.action_space: Discrete = Discrete(151)

        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        self.obs_queue: Optional[Queue] = None
        self.act_queue: Optional[Queue] = None

    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        self.episode += 1
        self.last_obs = self.observation_space.sample()

        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()

        # observations and actions queues for flow control
        # queues have a default max size of 1
        # as only 1 E+ timestep is processed at a time
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)

        self.energyplus_runner = EnergyPlusRunner(
            episode=self.episode,
            env_config=self.env_config,
            obs_queue=self.obs_queue,
            act_queue=self.act_queue,
        )
        self.energyplus_runner.start()

        # wait for E+ warmup to complete
        if not self.energyplus_runner.warmup_complete:
            self.energyplus_runner.warmup_queue.get()

        try:
            obs = self.obs_queue.get()
        except Empty:
            obs = self.last_obs

        return np.array(list(obs.values()))

    def step(self, action):
        self.timestep += 1
        done = False

        # check for simulation errors
        if self.energyplus_runner.failed():
            print(f"EnergyPlus failed with {self.energyplus_runner.sim_results['exit_code']}")
            exit(1)

        # simulation_complete is likely to happen after last env step()
        # is called, hence leading to waiting on queue for a timeout
        if self.energyplus_runner.simulation_complete:
            done = True
            obs = self.last_obs
        else:
            # rescale agent decision to actuator range
            sat_spt_value = self._rescale(
                n=int(action),  # noqa
                range1=(0, self.action_space.n),
                range2=(15, 30)
            )
            #print("SAT_SPT VALUE", sat_spt_value)

            # enqueue action (received by EnergyPlus through dedicated callback)
            # then wait to get next observation.
            # timeout is set to 2s to handle end of simulation cases, which happens async
            # and materializes by worker thread waiting on this queue (EnergyPlus callback
            # not consuming yet/anymore)
            # timeout value can be increased if E+ timestep takes longer
            timeout = 2
            try:
                self.act_queue.put(sat_spt_value, timeout=timeout)
                self.last_obs = obs = self.obs_queue.get(timeout=timeout)
            except (Full, Empty):
                done = True
                obs = self.last_obs

        # time inside simulation data
        reward = 0

        obs_vec = np.array(list(obs.values()))
        return obs_vec, reward, done, False, {}

    def render(self, mode="human"):
        pass

    @staticmethod
    def _rescale(
        n: int,
        range1: Tuple[float, float],
        range2: Tuple[float, float]
    ) -> float:
        action_nparray = np.linspace(range2[0], range2[1], (range1[1] - range1[0]))
        #print(action_nparray)
        return action_nparray[n]


default_args = {'idf': './in.idf',
                'epw': './weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                }
#
#SCORES:  [-343068118.4928892, -343058929.74458027, -343034573.5644406, -343063839.9638236, -343081534.0729704, -343076762.9154123, -343055059.71841764, -343033258.9391935, -343036122.53581744, -343047720.2466282]
if __name__ == "__main__":
    env = EnergyPlusEnv(default_args)
    print('action_space:', end='')
    print(env.action_space)
    print("OBS SHAPE:", env.observation_space.shape)
    rewards = []

    for episode in range(1):
        state = env.reset()
        done = False
        score = 0
        steps = 1

        print('type:', type(state))
        #sys.exit(1)
        while not done:
            print('------------------STEPS {}-----------------'.format(steps))
            action = env.action_space.sample()
            ret = n_state, reward, done, truncated, info = env.step(action)
            score += reward
            rewards.append(reward)
            # print('cost', reward)
            print('obs', n_state, len(n_state))

            action = env._rescale(
                n=int(action),  # noqa
                range1=(0, env.action_space.n),
                range2=(20, 26)
            )
            print('ACTION:', action)

            steps += 1
            #score += info['energy_reward']

        print('score', score)
