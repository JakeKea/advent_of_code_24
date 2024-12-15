# Day 14 - B
# Completed outside of notebook due to issue with getting matplotlib wigdgets to work in notebook
#Find the number of steps before the christmas tree appears

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

#Import Libraries and settings

settings = {
    "day": 14,
    "test_data": 0
}

#Load Input
def load_input(settings):
    #Derrive input file name
    if settings["test_data"]:
        data_subdir = "test"
        grid_boundaries = (11, 7)
    else:
        data_subdir = "actual"
        grid_boundaries = (101, 103)

    data_fp = f"./input/{data_subdir}/{settings["day"]}.txt"

    #Open and read the file
    with open(data_fp) as f:
        lines = f.read().split('\n')

    robots = []
    #For line in lines
    for line in lines:
        pos_chars = line.split(" ")[0].split("=")[1].split(",")
        pos = (int(pos_chars[0]), int(pos_chars[1]))

        vel_chars = line.split("=")[2].split(",")
        vel = (int(vel_chars[0]), int(vel_chars[1]))

        robots.append({"pos":pos, "vel":vel})


    return robots, grid_boundaries

data_in, grid_boundaries = load_input(settings)

#Grid dimensions
GRID_WIDTH = grid_boundaries[0]
GRID_HEIGHT = grid_boundaries[1]
GRID_DIVIDE_V = int((GRID_HEIGHT - 1) / 2)
GRID_DIVIDE_H = int((GRID_WIDTH - 1) / 2)

#Apply robot movement for n steps
def apply_nvel(r, n):
    new_pos_x = (r["pos"][0] + r["vel"][0] * n) % GRID_WIDTH
    new_pos_y = (r["pos"][1] + r["vel"][1] * n) % GRID_HEIGHT
    return (new_pos_x, new_pos_y)

#Get the quadrant of a given space
def get_quadrant(pos):
    #If on the divide then mark as quadrant 0
    if pos[0] == GRID_DIVIDE_H or pos[1] == GRID_DIVIDE_V:
        return 0
    
    if pos[0] < GRID_DIVIDE_H:
        quadrants = [1,3]
    else:
        quadrants = [2,4]

    if pos[1] < GRID_DIVIDE_V:
        return quadrants[0]
    else:
        return quadrants[1]

#Process all robots in the input
def process_robots(robots, n=100):

    #Calculate the new position after n moves and record which quadrant
    q_counts = [0,0,0,0,0]
    for robot in robots:
        new_loc = apply_nvel(robot, n)
        final_quadrant = get_quadrant(new_loc)
        q_counts[final_quadrant] += 1

    print(q_counts)

    #Get the safety score by multiplying the quadrant scores together
    safety_score = 1
    for q_count in q_counts[1:]:
        safety_score *= q_count

    return safety_score

#Convert the input to an array of positions and velocities
def get_data_points(data_in):
    poss = []
    vels = []

    for line in data_in:
        poss.append([line["pos"][0],line["pos"][1]])
        vels.append([line["vel"][0],line["vel"][1]])

    return poss, vels

poss, vels = get_data_points(data_in)

# Initial setup
points = np.array(poss)  # Starting positions (x, y)
velocities = np.array(vels)  # Velocities (vx, vy)
grid_size = [GRID_WIDTH, GRID_HEIGHT] # Size of the grid
num_steps = 100  # Total time steps

# Precompute all positions
positions = np.zeros((num_steps, len(points), 2))
#From manual scoping, the solution exists at 7603
positions[0] = points + (velocities * 7600)
for t in range(1, num_steps):
    positions[t] = positions[t-1] + velocities
    # Wrap around the grid edges
    positions[t] %= grid_size  

# Create the figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Leave space for the slider
ax.set_xlim(0, grid_size[0])
ax.set_ylim(0, grid_size[1])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_aspect('equal')
scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], c='blue', s=100)

# Add a slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]
slider = Slider(ax_slider, 'Time', 0, num_steps-1, valinit=0, valstep=1)

# Update function for the slider
def update(val):
    time_step = int(slider.val)
    scat.set_offsets(positions[time_step])  # Update scatter plot
    fig.canvas.draw_idle()  # Redraw the canvas

# Connect the slider to the update function
slider.on_changed(update)

plt.show()