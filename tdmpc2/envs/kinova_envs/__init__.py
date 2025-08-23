from envs.kinova_envs.KinovaAgent import KinovaGen3
from envs.kinova_envs.KinovaPushCubeEnv import KinovaPushCubeEnv
from envs.kinova_envs.KinovaPickCubeEnv import KinovaPickCubeEnv
from envs.kinova_envs.KinovaStackCubeEnv import KinovaStackCubeEnv
from envs.kinova_envs.KinovaMultitaskEnv import KinovaMultitaskEnv

SUPPORTED_TASKS = [
  "kinova_push_cube",
  "kinova_pick_cube",
  "kinova_stack_cube",
  "kinova_multitask"
]

