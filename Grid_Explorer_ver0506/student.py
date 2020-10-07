import pygame
from simulator import get_sensors, move_forward, move_backward, turn_left, turn_right, submit, set_map
# DO NOT MODIFY LINE 1
# You may import any libraries you want. But you may not import simulator_hidden
import math
import time

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
gray = (100, 100, 100)
blue = (0, 180, 255)

##############################
#### Write your code here ####
##############################

pos_x = 0
pos_y = 0
# Initial direction => south
direction = 2

move_unit = 0
rotate_unit = 65
standard_check = 0

width = 0
height = 0
corners = []
last_line_cnt = 0

lake_list = []
building_list = []
coord_visited = [(0, 0)]


def caliberate():
    global move_unit, rotate_unit, standard_check
    move_before_gray = 0
    sensor_val = get_sensors()

    while(sensor_val[0][0] != gray):
        move_forward()
        move_before_gray += 1
        sensor_val = get_sensors()

    for _ in range(move_before_gray):
        move_backward()

    for _ in range(rotate_unit):
        turn_right()

    while(True):
        for _ in range(move_before_gray):
            move_forward()

        sensor_val = get_sensors()

        for _ in range(move_before_gray):
            move_backward()

        if sensor_val[0][0] != white:
            if sensor_val[0][0] == sensor_val[0][1]:
                break

        turn_right()
        rotate_unit += 1

    inner_rotate = 0
    while(True):
        for _ in range(move_before_gray):
            move_forward()

        sensor_val = get_sensors()

        for _ in range(move_before_gray):
            move_backward()

        if sensor_val[0][0] != sensor_val[0][1]:
            break

        turn_right()
        inner_rotate += 1

    for _ in range(rotate_unit+inner_rotate):
        turn_left()

    rotate_unit += int(math.floor((inner_rotate - 1) / 2))

    sensor_val = get_sensors()
    turned = False
    if(sensor_val[1] > -1 and sensor_val[1] <= 3):
        for _ in range(rotate_unit):
            turn_left()
        turned = True

    for _ in range(move_before_gray):
        move_forward()

    sensor_val = get_sensors()
    while(sensor_val[0][0] == gray):
        move_forward()
        move_unit += 1
        sensor_val = get_sensors()

    while(sensor_val[0][0] == white):
        move_forward()
        move_unit += 1
        sensor_val = get_sensors()

    for _ in range(move_unit + move_before_gray):
        move_backward()

    if turned:
        for _ in range(rotate_unit):
            turn_right()

    standard_check = move_before_gray + 1


def set_and_save_pos(move_type):
    global pos_x, pos_y, coord_visited
    flag = 0
    if move_type == "forward":
        flag += 1
    else:
        flag -= 1

    if direction == 0:
        pos_y -= flag
    elif direction == 1:
        pos_x += flag
    elif direction == 2:
        pos_y += flag
    else:
        pos_x -= flag

    coord_visited.append((pos_x, pos_y))


def move_forward_block():
    for _ in range(move_unit):
        move_forward()
    set_and_save_pos("forward")


def move_backward_block():
    for _ in range(move_unit):
        move_backward()
    set_and_save_pos("backward")


def turn_right_90():
    global direction
    for _ in range(rotate_unit):
        turn_right()
    direction += 1
    if direction > 3:
        direction -= 4


def turn_left_90():
    global direction
    for _ in range(rotate_unit):
        turn_left()
    direction -= 1
    if direction < 0:
        direction += 4


def is_obstacle_in_front():
    sensor_val = get_sensors()
    return sensor_val[1] > -1 and sensor_val[1] <= 3


def is_last_line():
    if width == 0 or height == 0:
        return False
    total = (width + 1) * (height + 1)
    return total - len(list(set(coord_visited))) < 2 * abs(width - height)


def is_corner():
    if width == 0 and height == 0:
        return False
    elif width == 0:
        return pos_y == height
    else:
        return (pos_x, pos_y) in corners


def evade_obstacle():
    global building_list, coord_visited, pos_x, pos_y

    building_list.append(get_front_pos())
    coord_visited.append(get_front_pos())

    recur_cnt = 1
    initial_dir = direction

    ignore_flag = False
    while recur_cnt >= 0:
        if is_last_line():
            ignore_flag = False
        turn_left_90()
        if is_obstacle_in_front():
            recur_cnt += 2
            continue
        else:
            if not check_and_move(ignore_flag, False):
                break

        turn_right_90()

        if is_obstacle_in_front() and not is_front_visited():
            building_list.append(get_front_pos())
            recur_cnt += 1
            continue
        else:
            if not check_and_move(ignore_flag, False):
                break

        if is_obstacle_in_front() and not is_front_visited():
            recur_cnt += 1
            continue

        turn_right_90()
        ignore_flag = True
        recur_cnt -= 1

    coord_visited.append((pos_x, pos_y))

    while direction != initial_dir:
        turn_left_90()


def get_front_pos():
    next_pos_x = pos_x
    next_pos_y = pos_y
    if direction == 0:
        next_pos_y -= 1
    elif direction == 1:
        next_pos_x += 1
    elif direction == 2:
        next_pos_y += 1
    else:
        next_pos_x -= 1

    return (next_pos_x, next_pos_y)


def is_front_visited():
    next_pos = get_front_pos()
    return next_pos in coord_visited


def check_lake():
    global lake_list
    sensor_val = get_sensors()
    if sensor_val[0][0] == blue and sensor_val[0][1] == blue:
        # print("lake")
        lake_list.append((pos_x, pos_y))


def check_and_move(visit_flag, save_flag):
    global height, width, pos_x, pos_y, corners

    check_unit = 0
    sensor_val = get_sensors()

    is_normal_move = True

    while sensor_val[0][0] != gray and sensor_val[0][0] != black:
        move_forward()
        check_unit += 1
        sensor_val = get_sensors()

    if visit_flag and is_front_visited():
        # print("visited")
        for _ in range(standard_check):
            move_backward()
        turn_left_90()
        time.sleep(0.1)
        return False

    if sensor_val[0][0] == black or sensor_val[0][1] == black:
        if height == 0 and direction == 2:
            height = pos_y
        elif width == 0 and direction == 1:
            width = pos_x
        elif len(corners) == 0:
            top_x = 0
            top_y = 0
            bottom_x = width
            bottom_y = height
            for i in range(math.ceil(height/2)):
                corners.append((top_x+i, top_y+i))
                corners.append((bottom_x-i, bottom_y-i))
                corners.append((bottom_x-i, top_y+i))
                corners.append((bottom_x-i, top_y-i))

        for _ in range(standard_check):
            move_backward()
        turn_left_90()
        is_normal_move = False

    else:
        for _ in range(move_unit - check_unit):
            move_forward()
        if save_flag:
            set_and_save_pos("forward")
        else:
            pos_x = get_front_pos()[0]
            pos_y = get_front_pos()[1]

    time.sleep(0.1)
    return is_normal_move


def return_to_base():
    global coord_visited
    coord_visited = list(set(building_list))
    while direction != 0:
        turn_right_90()

    while pos_y != 0:
        if is_obstacle_in_front():
            evade_obstacle()
        else:
            check_and_move(True, False)

    while direction != 3:
        turn_left_90()

    while pos_x != 0:
        if is_obstacle_in_front():
            evade_obstacle()
        else:
            check_and_move(True, False)


def main():
    caliberate()

    loop_cnt = 0
    while(True):
        check_lake()
        old_pos = (pos_x, pos_y)
        if not is_front_visited() and is_obstacle_in_front():
            evade_obstacle()
        else:
            check_and_move(True, True)
        # print(coord_visited)
        if (pos_x, pos_y) == old_pos:
            loop_cnt += 1
            if loop_cnt >= 3:
                check_lake()
                return_to_base()
                submit(list(set(lake_list)), list(set(building_list)))
                break
        else:
            loop_cnt = 0
        if width * height != 0 and len(list(set(coord_visited))) == (width + 1) * (height + 1):
            check_lake()
            return_to_base()
            submit(list(set(lake_list)), list(set(building_list)))
            break


main()

##############################


# If you want to try moving around the map with your keyboard, uncomment the below lines
while True:
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_UP]:
        move_forward()
    if pressed[pygame.K_DOWN]:
        move_backward()
    if pressed[pygame.K_LEFT]:
        turn_left()
    if pressed[pygame.K_RIGHT]:
        turn_right()
    if pressed[pygame.K_n]:
        set_map((10, 5), [(8, 0), (4, 9), (2, 0), (3, 3),
                          (4, 1)], [(7, 2), (0, 1), (2, 3)])
    if pressed[pygame.K_c]:
        print(get_sensors())
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit("Closing...")
