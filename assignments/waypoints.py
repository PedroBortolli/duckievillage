# Intro to Robotics - MAC0318
#
# Name: Pedro Bortolli
# NUSP: 9793721
#
# ---
#
# Assignment 2 - Waypoint navigation
# Carefully read this header and follow submission instructions!
# Failure to follow instructions may result in a zero!
#
# Task:
#  - Create a waypoint navigating Duckiebot.
#  - Implement navigation by shortest distance.
#    + Given two points, the shortest distance route is a straight line between them.
#  - Implement navigation by axes.
#    + Given two points, navigate _only_ through the x and y axes.
#
# Don't forget that you can (and should!) read the Duckievillage code in search of anything that
# can be useful for your work.
#
# Take a look at the Waypoints class in duckievillage.py. Method arrived in DuckievillageEnv may
# also help you implement waypoint navigation.
#
# Don't forget to run this from the Duckievillage root directory!
# From within the root directory, run python with the -m flag.
#   python3 -m assignments.waypoints
#
# Submission instructions:
#  0. Add your name and USP number to the header's header.
#  1. Make sure everything is running fine and there are no errors during startup. If the code does
#     not even start the environment, you will receive a zero.
#  2. Test your code and make sure it's doing what it's supposed to do.
#  3. Append your NUSP to this file name.
#  4. Submit your work to PACA.

import sys
import math
import pyglet
from pyglet.window import key
from pyglet.window import mouse
import numpy as np
import gym
import gym_duckietown
import duckievillage
from duckievillage import DuckievillageEnv

# Threshold for angle and distance
EPS_ANGLE = 1e-2
EPS_DIST = 0.5

# Keeps reference to the next goal
next_point = None

env = DuckievillageEnv(
  seed = 101,
  map_name = './maps/udem1.yaml',
  draw_curve = False,
  draw_bbox = False,
  domain_rand = False,
  distortion = False,
  top_down = False
)
# Set mark-mode whenever we run waypoints.py with the -m or --mark-waypoints flag.
waypoints = duckievillage.Waypoints(env, '--read-from-file' in sys.argv)

env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, mods):
  if symbol == key.ESCAPE:
    waypoints.write('waypoints.txt')
    env.close()
    sys.exit(0)
  env.render()

# On mouse press, register waypoint.
@env.unwrapped.window.event
def on_mouse_press(x, y, button, mods):
  if button == mouse.LEFT:
    if x < 0 or x > duckievillage.WINDOW_WIDTH or y < 0 or y > duckievillage.WINDOW_HEIGHT:
      return
    # Convert coordinates from window position to Duckietown coordinates.
    px, py = env.convert_coords(x, y)
    waypoints.mark(px, py, x//2, y)

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def arrived(dist):
  """Check if car arrived at destination"""
  return abs(dist[0]) + abs(dist[1]) < EPS_DIST

def get_angle(current, goal):
  """Find minnimum angle between two vectors"""
  dot = np.dot(current, goal)
  nc = np.linalg.norm(current)
  ng = np.linalg.norm(goal)
  cos = dot / (nc * ng)
  angle = math.acos(cos)
  return angle

def update(dt):

  action = [0.0, 0.0]

  # This is where you'll write the Duckie's logic.
  # You can fetch your duckiebot's position with env.get_position().

  # Get next waypoint (if any)
  global next_point
  if not next_point:
    next_point = waypoints.next()

  print(next_point)
  if next_point:
    # Current direction vector and goal distance vector
    current = env.get_dir_vec()[0::2]
    goal = next_point - env.get_position()

    # Checks whether the current goal was reached or not
    if arrived(goal):
      next_point = None

    else:
      angle = get_angle(current, goal)
      if angle > EPS_ANGLE:
        # Find if we should turn left or right
        cross = np.cross(current, goal)
        sign = -np.sign(cross)

        # Reduce angle by turning car towards goal vector
        action[1] = 2 * angle * sign

      else:
        # Just move the car forwards
        action[0] = 1.0

  obs, reward, done, info = env.step(action)

  env.render()
  waypoints.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

pyglet.app.run()

env.close()