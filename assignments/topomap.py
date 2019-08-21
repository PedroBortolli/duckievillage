# Intro to Robotics - MAC0318
#
# Name: Pedro Bortolli
# NUSP: 9793721
#
# ---
#
# Assignment 3 - Topological maps
# Carefully read this header and follow submission instructions!
# Failure to follow instructions may result in a zero!
#
# Task:
#  - By making use of your last assignment (navigation by waypoints), implement shortest-path
#    navigation by calling the BFS path finding algorithm on the topological map.
#  - Adapt your navigation plan to the usual traffic rules, creating a directed graph G such that
#    nodes are lanes, and an edge e=(p, q) exists only if the agent is able to go from p to q without
#    breaking any of the following rules:
#    1. No U-turns allowed.
#    2. Right lane only.
#    3. Always go forward, never backwards.
#    Run BFS to test your digraph.
#
# Don't forget that you can (and should!) read the Duckievillage code in search of anything that
# can be useful for your work.
#
# The topological map is already implemented within the duckievillage module. You can access the
# map graph through env.topo_graph. This field contains a TopoGraph object that maps all drivable
# tiles as nodes in the graph, with tiles connected by edges. To find the optimal shortest path
# between two drivable tiles, call env.topo_graph.path(p, q), where p and q are the source and
# target nodes. This'll return a list of positions corresponding to the center of each tile.
#
# Don't forget to run this from the Duckievillage root directory!
# From within the root directory, run python with the -m flag.
#   python3 -m assignments.topomap
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

TRACKS = (('./maps/dense.yaml', 5), ('./maps/large.yaml', 15))

# Constants to set grid positions
DELTA = 0.585
GAP = DELTA/4

# Threshold for angle and distance
EPS_ANGLE = 1e-2
EPS_DIST = 1e-1

# Keeps reference to the next goal
next_point = None

# Keeps reference to last goal
last_goal = None

# Decide if the entire path has been already travelled
finished = False

which = 0

env = DuckievillageEnv(
  seed = 101,
  map_name = TRACKS[which][0],
  draw_curve = False,
  draw_bbox = False,
  domain_rand = False,
  distortion = False,
  top_down = False,
  cam_height = TRACKS[which][1]
)

# Set mark-mode whenever we run waypoints.py with the -m or --mark-waypoints flag.
waypoints = duckievillage.Waypoints(env, '--read-from-file' in sys.argv)

env.reset()
env.render()

# Use G to create the directed graph mentioned in the assignment task list.
G = duckievillage.TopoGraph(env.road_tile_size)

H = env.topo_graph

first_node = H.closest_node(env.get_position())
first_direction = 'up'
queue = []
seen = dict()
queue.append((first_node, first_direction))



def get_neighbors(node):
  up    = H.edge(node, (node[0], node[1] - DELTA))
  down  = H.edge(node, (node[0], node[1] + DELTA))
  left  = H.edge(node, (node[0] - DELTA, node[1]))
  right = H.edge(node, (node[0] + DELTA, node[1]))
  return up, down, left, right


#waypoints.mark(3.2175, 0.14625, 3.2175, 0.14625)



for node in H.nodes():
  up, down, left, right = get_neighbors(node)

  if int(up) + int(down) + int(left) + int(right) <= 2:
    if up and down:
      G.add_node((node[0] - GAP, node[1]))
      G.add_node((node[0] + GAP, node[1]))
    elif left and right:
      G.add_node((node[0], node[1] - GAP))
      G.add_node((node[0], node[1] + GAP))
    elif (up and left) or (down and right):
      G.add_node((node[0] - GAP, node[1] - GAP))
      G.add_node((node[0] + GAP, node[1] + GAP))
    elif (up and right) or (down and left):
      G.add_node((node[0] - GAP, node[1] + GAP))
      G.add_node((node[0] + GAP, node[1] - GAP))
  else:
    G.add_node(node)


while len(queue) > 0:
  node, direction = queue.pop(0)
  seen[(node, direction)] = True
  up, down, left, right = get_neighbors(node)

  print(node, direction, "    ( ", up, down, left, right, ")")
  
  if int(up) + int(down) + int(left) + int(right) <= 2:
    if direction == 'up':
      if up:
        next_node = (node[0] + GAP, node[1] - DELTA)
        G.add_dir_edge((node[0] + GAP, node[1]), next_node)
        if (next_node, 'up') not in seen:
          queue.append(((node[0], node[1] - DELTA), 'up'))

      if left:
        next_node = (node[0] - DELTA, node[1] - GAP)
        G.add_dir_edge((node[0] + GAP, node[1] - GAP), next_node)
        if (next_node, 'left') not in seen:
          queue.append(((node[0], node[1]), 'left'))

      if right:
        next_node = (node[0] + DELTA, node[1] + GAP - DELTA)
        G.add_dir_edge((node[0] + GAP, node[1]), next_node)
        if (next_node, 'right') not in seen:
          queue.append(((node[0], node[1]), 'right'))

    elif direction == 'left':
      if up:
        next_node = (node[0] - DELTA + GAP, node[1] - DELTA)
        G.add_dir_edge((node[0], node[1] - GAP), next_node)
        if (next_node, 'up') not in seen:
          queue.append(((node[0], node[1]), 'up'))
      if left:
        next_node = (node[0] - DELTA, node[1] - GAP)
        G.add_dir_edge((node[0], node[1] - GAP), next_node)
        if (next_node, 'left') not in seen:
          queue.append(((node[0] - DELTA, node[1]), 'left'))
      if down:
        next_node = (node[0] - DELTA - GAP, node[1] + DELTA)
        G.add_dir_edge((node[0] - GAP, node[1] - GAP), next_node)
        if (next_node, 'down') not in seen:
          queue.append(((node[0], node[1]), 'down'))

    elif direction == 'right':
      if up:
        next_node = (node[0] + DELTA + GAP, node[1] - DELTA)
        G.add_dir_edge((node[0], node[1] + GAP), next_node)
        if (next_node, 'up') not in seen:
          queue.append(((node[0], node[1]), 'up'))
      if right:
        next_node = (node[0] + DELTA, node[1] + GAP)
        G.add_dir_edge((node[0], node[1] + GAP), next_node)
        if (next_node, 'right') not in seen:
          queue.append(((node[0] + DELTA, node[1]), 'right'))
      if down:
        next_node = (node[0] + DELTA - GAP, node[1] + DELTA)
        G.add_dir_edge((node[0], node[1] + GAP), next_node)
        if (next_node, 'down') not in seen:
          queue.append(((node[0], node[1]), 'down'))

    elif direction == 'down':
      if down:
        next_node = (node[0] - GAP, node[1] + DELTA)
        G.add_dir_edge((node[0] - GAP, node[1]), next_node)
        if (next_node, 'down') not in seen:
          queue.append(((node[0], node[1] + DELTA), 'down'))
      if right:
        next_node = (node[0] + DELTA, node[1] + DELTA - GAP)
        G.add_dir_edge((node[0] - GAP, node[1]), next_node)
        if (next_node, 'right') not in seen:
          queue.append(((node[0], node[1]), 'right'))
      if left:
        next_node = (node[0] - DELTA, node[1] + DELTA - GAP)
        G.add_dir_edge((node[0] - GAP, node[1] - GAP), next_node)
        if (next_node, 'left') not in seen:
          queue.append(((node[0], node[1]), 'left'))

  else:
    if up:
      G.add_dir_edge(node, (node[0] + GAP, node[1] - DELTA))
      queue.append(((node[0], node[1] - DELTA), 'up'))
    if down:
      G.add_dir_edge(node, (node[0] - GAP, node[1] + DELTA))
      queue.append(((node[0], node[1] + DELTA), 'down'))
    if left:
      G.add_dir_edge(node, (node[0] - DELTA, node[1] - GAP))
      queue.append(((node[0] - DELTA, node[1]), 'left'))
    if right:
      G.add_dir_edge(node, (node[0] + DELTA, node[1] + GAP))
      queue.append(((node[0] + DELTA, node[1]), 'right'))

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
  if symbol == key.ESCAPE:
    env.close()
    sys.exit(0)
  env.render()

# On mouse press, register waypoint.
@env.unwrapped.window.event
def on_mouse_press(x, y, button, mods):
  global last_goal
  if button == mouse.LEFT:
    if x < 0 or x > duckievillage.WINDOW_WIDTH or y < 0 or y > duckievillage.WINDOW_HEIGHT:
      return
    # Convert coordinates from window position to Duckietown coordinates.
    px, py = env.convert_coords(x, y)
    # The function below calls BFS from the bot's current position to your mouse's position,
    # returning a list of positions to go to.
    """
    Q = env.topo_graph.bfs(env.get_position(), (px, py))
    Q.reverse()

    for i in range(len(Q)):
      waypoints.mark(Q[i][0], Q[i][1], Q[i][0], Q[i][1])
      if i+1 == len(Q):
        last_goal = Q[i]
    """

    # Once you implement your new digraph, you should be able to call BFS in the following way:
    Q = G.bfs(env.get_position(), (px, py))
    print("\nDepois de chamar bfs:\n")
    print(Q)

    """adj = G.adj()
    for lala in adj:
      for lul in adj[lala]:
        print(lala, " -> ", lul)
      print("\n")"""



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
  global next_point, finished

  # If the final destination has been achieved don't do anything else
  if finished:
    return

  action = [0.0, 0.0]

  # Get next waypoint (if any)
  if not next_point:
    next_point = waypoints.next()

  if next_point:
    # Current direction vector and goal distance vector
    current = env.get_dir_vec()[0::2]
    goal = next_point - env.get_position()

    # Checks whether the current goal was reached or not
    if arrived(goal):
      if next_point == last_goal:
        finished = True
      next_point = None

    else:
      angle = get_angle(current, goal)
      if angle > EPS_ANGLE:
        # Find if we should turn left or right
        cross = np.cross(current, goal)
        sign = -np.sign(cross)

        # Reduce angle by turning car towards goal vector
        action[1] = 3 * angle * sign

      else:
        # Just move the car forwards
        action[0] = 1.0

  obs, reward, done, info = env.step(action)
  env.render()




pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

pyglet.app.run()

env.close()