# Intro to Robotics - MAC0318
#
# Name:
# NUSP:
#
# ---
#
# Assignment 3 - Topological maps
# Carefully read this header and follow submission instructions!
# Failure to follow instructions may result in a zero!
#
# Task:
#  - Implement navigation by calling the A-star path finding algorithm on the topological map.
#
# The topological map is already implemented within the duckievillage module. You can access the
# map graph through env.topo_graph. This field contains a TopoGraph object that maps all drivable
# tiles as nodes in the graph, with tiles connected by edges. To find the optimal shortest path
# between two drivable tiles, call env.topo_graph.path(p, q), where p and q are the source and
# target nodes. This'll return a list of positions corresponding to the center of each tile.
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
#  3. Copy this file to ../solutions/
#  4. Push changes to your fork. You will be evaluated from what's in your repository!

import sys
import pyglet
from pyglet.window import key
from pyglet.window import mouse
import numpy as np
import gym
import gym_duckietown
import duckievillage
from duckievillage import DuckievillageEnv


env = DuckievillageEnv(
  seed = 101,
  map_name = './maps/dense.yaml',
  draw_curve = False,
  draw_bbox = False,
  domain_rand = False,
  distortion = False,
  top_down = False
)

env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
  if symbol == key.ESCAPE:
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
    print(env.topo_graph.path(env.current_tile(), env.get_grid_coords((px, 0, py))))

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):
  action = [0.0, 0.0]

  # This is where you'll write the Duckie's logic.
  # You can fetch your duckiebot's position with env.get_position().

  obs, reward, done, info = env.step(action)

  env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

pyglet.app.run()

env.close()