##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## basic logic
##

def basic_policy(obs):
    x, y, vx, vy, angle, angular_vel, left, right = obs

    if left and right:
        return 0
    if angle > 0.15:
        return 1
    if angle < -0.15:
        return 3
    if vy < -0.8:
        return 2
    if vx > 0.5:
        return 1
    if vx < -0.5:
        return 3
    if y < 0.5 and vy < -0.3:
        return 2
    return 0
