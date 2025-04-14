from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Ball, Box 
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
from minigrid.core.constants import COLORS, COLOR_TO_IDX, IDX_TO_COLOR

class EmptyEnvQuests(MiniGridEnv):
    """
    ## Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Empty-Quest-v0` - Empty room with a goal in a random position and other objects in random positions

    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        num_balls: int = 3, 
        num_boxes: int = 3, 
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.num_balls = num_balls
        self.num_boxes = num_boxes

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # randomize the goal position 
        goal_pos = (np.random.randint(1, width-1), np.random.randint(1, height-1))
        self.goal_pos = goal_pos

        available_colors = list(COLORS.keys())

        for _ in range(self.num_balls):
            ball_color = self._rand_elem(available_colors)
            self.place_obj(Ball(color=ball_color))
        
        for _ in range(self.num_boxes):
            box_color = self._rand_elem(available_colors)
            self.place_obj(Box(color=box_color))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.place_obj(Goal())

        # Place the agent
        self.place_agent()

        self.mission = "get to the green goal square"


    def step(self, action):
        """Override step to include custom reward calculation"""
        # Call parent class step method to handle basic environment dynamics
        obs, _, terminated, truncated, info = super().step(action)
        
        # Initialize tracker variables on first step
        if not hasattr(self, 'visited_positions'):
            self.visited_positions = set()
            self.collected_objects = set()  # Track unique object types collected
            self.last_carrying = None  # Track previous carrying state
        
        # Default reward starts negative to encourage efficiency
        reward = -0.01
        
        # --- EXPLORATION COMPONENT ---
        # Reward for visiting new positions
        pos_tuple = tuple(self.agent_pos)
        if pos_tuple not in self.visited_positions:
            self.visited_positions.add(pos_tuple)
            reward += 0.05  # Small reward for exploring new cells
        
        # --- COLLECTION COMPONENT ---
        # Reward for picking up objects (especially novel ones)
        if self.carrying is not None and self.last_carrying is None:
            obj_type = self.carrying.type
            obj_color = self.carrying.color
            obj_tuple = (obj_type, obj_color)
            
            # Higher reward for collecting new object types
            if obj_tuple not in self.collected_objects:
                self.collected_objects.add(obj_tuple)
                reward += 0.3  # Larger reward for novel objects
            else:
                reward += 0.1  # Smaller reward for already-seen objects
        
        # --- PROXIMITY COMPONENT ---
        # Get position in front of agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        
        # Small reward for being near objects (encourages investigation)
        if fwd_cell and fwd_cell.type not in ['wall', 'goal']:
            reward += 0.02
        
        # --- GOAL COMPONENT ---
        # Large reward for reaching the goal
        if np.array_equal(self.agent_pos, self.goal_pos):
            # Scale goal reward based on exploration coverage and collection
            exploration_ratio = len(self.visited_positions) / (self.width * self.height)
            collection_ratio = len(self.collected_objects) / (self.num_balls + self.num_boxes)
            
            # Balance between exploration and collection with diminishing returns
            # for step count to encourage efficiency
            goal_reward = 1.0 + exploration_ratio + collection_ratio
            goal_reward *= (1 - 0.5 * (self.step_count / self.max_steps))
            
            reward += goal_reward
            terminated = True
        
        # Update tracking variables for next step
        self.last_carrying = self.carrying
        
        return obs, reward, terminated, truncated, info