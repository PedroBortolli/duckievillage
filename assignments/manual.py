# Intro to Robotics - MAC0318
#
# Name: Pedro Bortolli
# NUSP: 9793721
#
# ---
#
# Assignment 1 - Manual steering
# Carefully read this header and follow submission instructions!
# Failure to follow instructions may result in a zero!
#
# Task:
#  - Create a remote control Duckiebot.
#
# Don't forget that you can (and should!) read the Duckievillage code in search of anything that
# can be useful for your work.
#
# Don't forget to run this from the Duckievillage root directory!
# From within the root directory, run python with the -m flag.
#   python3 -m assignments.manual
#
# Submission instructions:
#  0. Add your name and USP number to the header's header.
#  1. Make sure everything is running fine and there are no errors during startup. If the code does
#     not even start the environment, you will receive a zero.
#  2. Test your code and make sure it's doing what it's supposed to do.
#  3. Append your NUSP to this file name.
#  4. Submit your work to PACA.

import sys
import pyglet
import numpy as np
from pyglet.window import key
import gym
import gym_duckietown
from duckievillage import DuckievillageEnv

# Constants for movement
SPEED = 1.0
ROTATION = 6.0

# We'll use our version of Duckietown: Duckievillage. This environment will be where we'll run most
# our tasks in.
env = DuckievillageEnv(
  seed = 101,
  map_name = '../maps/loop_empty.yaml',
  draw_curve = False,
  draw_bbox = False,
  domain_rand = False,
  distortion = False,
  top_down = False
)

# Let's reset the environment to get our Duckiebot somewhere random.
env.reset()
# This function is used to draw the environment to a graphical user interface using Pyglet.
env.render()

# We use this function for on-press key events (not something we use for real-time feedback,
# though). We'll register ESC as our way out of the Matrix.
@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
  if symbol == key.ESCAPE:
    env.close()
    sys.exit(0)
  env.render()

# KeyStateHandler handles key states. Pretty self-descriptive.
key_handler = key.KeyStateHandler()
# Let's register our key handler to the environment's key listener.
env.unwrapped.window.push_handlers(key_handler)

# This function handles every frame update. Parameter dt is the elapsed time, in milliseconds,
# since the last update call.
def update(dt):
  # At each step, the agent accepts an action array (also accepts numpy.arrays):
  #   action = [forward_velocity, steering_angle]
  # Play with the actions and figure out how to make your own remote control duckiebot!
  action = np.array([0.0, 0.0])

  # The key_handler object handles keyboard events. It's basically a map indexed by Pyglet keys
  # with values True if the key is being held, or False otherwise.
  if key_handler[key.W] or key_handler[key.UP]:
    action[0] += SPEED
  if key_handler[key.A] or key_handler[key.LEFT]:
    action[1] += ROTATION
  if key_handler[key.S] or key_handler[key.DOWN]:
    action[0] -= SPEED
  if key_handler[key.D] or key_handler[key.RIGHT]:
    action[1] -= ROTATION

  # Try to turn at the same angular velocity both moving or not
  if not action[0]:
    action[1] /= 4

  # At each step, the environment may (or may not) change given your actions. Function step takes
  # as parameter the array action and returns an observation (what the robot is currently
  # seeing), a reward (mostly used for reinforcement learning), whether the episode is done (also
  # used for reinforcement learning) and some info on the elapsed episode.
  # Let's ignore return values for now.
  obs, reward, done, info = env.step(action)

  # Refresh at every update.
  env.render()

# Let's call update every 1.0 / frame_rate second.
pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop.
pyglet.app.run()

env.close()