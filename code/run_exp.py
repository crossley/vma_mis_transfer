import sys
import os
import serial
import time
import struct
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == "__main__":

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    subject = input("Subject number: ").strip()
    day = input("Day number: ").strip()
    dir_data = "../data"
    full_path = os.path.join(dir_data, f"sub_{subject}_day_{day}_data.csv")
    full_path_move = os.path.join(dir_data, f"sub_{subject}_day_{day}_data_move.csv")

    # Uncomment to check if file already exists
    if os.path.exists(full_path) or os.path.exists(full_path_move):
        print(f"File {full_path} or {full_path_move} already exists. Aborting.")
        sys.exit()

    use_liberty = False


    # This method grabs the position of the sensor
    def getPosition(ser, recordsize, averager):
        ser.reset_input_buffer()

        # Set variables
        # This defines the length of the binary header (bytes 0-7)
        header = 8
        # This defines the bytesize of IEEE floating point
        byte_size = 4

        # Obtain data
        ser.write(b"P")
        # time.sleep(0.1)
        # print("inWaiting " + str(ser.inWaiting()))
        # print("recorded size " + str(recordsize))

        # Read header to remove it from the input buffer
        ser.read(header)

        positions = []

        # Read the three coordinates
        for x in range(3):
            # Read the coordinate
            coord = ser.read(byte_size)

            # Convert hex to floating point (little endian order)
            coord = struct.unpack("<f", coord)[0]

            positions.append(coord)

        return positions


    if use_liberty:
        ser = serial.Serial()
        ser.baudrate = 115200
        ser.port = "COM3"

        print(ser)
        ser.open()

        # Checks serial port if open
        if ser.is_open == False:
            print("Error! Serial port is not open")
            exit()

        # Send command to receive data through port
        ser.write(b"P")
        time.sleep(1)

        # Checks if Liberty is responding(e.g on)
        if ser.inWaiting() < 1:
            print("Error! Check if liberty is on!")
            exit()

        # Set liberty output mode to binary
        ser.write(b"F1\r")
        time.sleep(1)

        # Set distance unit to centimeters
        ser.write(b"U1\r")
        time.sleep(0.1)

        # Set hemisphere to +Z
        ser.write(b"H1,0,0,1\r")
        time.sleep(0.1)

        # Set sample rate to 240hz
        ser.write(b"R4\r")
        time.sleep(0.1)

        # Reset frame count
        ser.write(b"Q1\r")
        time.sleep(0.1)

        # Set output to only include position (no orientation)
        ser.write(b"O1,3,9\r")
        time.sleep(0.1)
        ser.reset_input_buffer()

        # Obtain data
        ser.write(b"P")
        time.sleep(0.1)

        # Size of response
        recordsize = ser.inWaiting()
        ser.reset_input_buffer()
        averager = 4

    # useful constants but need to change / verify on each computer
    # lab computer is resolution 1920 x 1080
    # monitor size is 60 cm x 33 cm
    # px_per_cm = np.mean([1920 / 60, 1080 / 33])
    px_per_cm = 1080 / 33

    target_angles = [-30, -15, 0, 15, 30]

    rng = np.random.default_rng()

    def shuffled_target_cycles(n_trials):
        target_sequence = []
        while len(target_sequence) < n_trials:
            target_sequence.extend(rng.permutation(target_angles))
        return np.array(target_sequence[:n_trials])

    # The diagnostic phase is the nf_burst_after_clamp design.
    # The repeated motif is clamp, no feedback, no feedback,
    # no feedback, rotated, across a balanced five-target grid.
    n_diagnostic = 200
    target_angle_diagnostic = np.array(target_angles * 40)
    rng.shuffle(target_angle_diagnostic)

    feedback_type_diagnostic = []
    rotation_diagnostic = []
    clamp_error_diagnostic = []
    endpoint_visible_diagnostic = []

    motif = ["clamp", "none", "none", "none", "rotated"]
    state_centers = [0, 8, -8, 0]
    rotation_set = [-12, -8, -4, 0, 4, 8, 12]
    clamp_values = [-12, -8, -4, 4, 8, 12]

    for i in range(n_diagnostic):
        motif_item = motif[i % len(motif)]
        cycle = i // len(motif)
        center = state_centers[cycle % len(state_centers)]

        if motif_item == "none":
            feedback_type_diagnostic.append("none")
            rotation_diagnostic.append(0)
            clamp_error_diagnostic.append(np.nan)
            endpoint_visible_diagnostic.append(0)
        elif motif_item == "clamp":
            feedback_type_diagnostic.append("clamp")
            rotation_diagnostic.append(0)
            clamp_error_diagnostic.append(clamp_values[(cycle + i) % len(clamp_values)])
            endpoint_visible_diagnostic.append(1)
        else:
            feedback_type_diagnostic.append("rotated")
            base_rotation = rotation_set[(cycle + i) % len(rotation_set)]
            rotation_diagnostic.append(np.clip(center + base_rotation, -12, 12))
            clamp_error_diagnostic.append(np.nan)
            endpoint_visible_diagnostic.append(1)

    feedback_type_diagnostic = np.array(feedback_type_diagnostic)
    rotation_diagnostic = np.array(rotation_diagnostic) * np.pi / 180
    clamp_error_diagnostic = np.array(clamp_error_diagnostic) * np.pi / 180
    endpoint_visible_diagnostic = np.array(endpoint_visible_diagnostic)

    # The adaptation phase uses shuffled cycles through all five
    # target locations. Endpoint feedback is rotated except on
    # every 10th trial, where feedback is withheld.
    n_adaptation = 110
    target_angle_adaptation = shuffled_target_cycles(n_adaptation)
    feedback_type_adaptation = np.array(
        ["none" if i % 10 == 9 else "rotated" for i in range(n_adaptation)])
    endpoint_visible_adaptation = (feedback_type_adaptation != "none").astype(float)
    rotation_adaptation = np.ones(n_adaptation) * 30 * np.pi / 180
    rotation_adaptation[feedback_type_adaptation == "none"] = 0
    clamp_error_adaptation = np.ones(n_adaptation) * np.nan

    # The generalization phase uses shuffled cycles through all
    # five target locations with no feedback.
    n_generalization = 66
    target_angle_generalization = shuffled_target_cycles(n_generalization)
    feedback_type_generalization = np.array(["none"] * n_generalization)
    endpoint_visible_generalization = np.zeros(n_generalization)
    rotation_generalization = np.zeros(n_generalization)
    clamp_error_generalization = np.ones(n_generalization) * np.nan

    # concatenate all phases
    endpoint_visible = np.concatenate([
        endpoint_visible_diagnostic,
        endpoint_visible_adaptation,
        endpoint_visible_generalization
    ])

    rotation = np.concatenate([
        rotation_diagnostic,
        rotation_adaptation,
        rotation_generalization
    ])

    clamp_error = np.concatenate([
        clamp_error_diagnostic,
        clamp_error_adaptation,
        clamp_error_generalization
    ])

    feedback_type = np.concatenate([
        feedback_type_diagnostic,
        feedback_type_adaptation,
        feedback_type_generalization
    ])

    target_angle = np.concatenate([
        target_angle_diagnostic,
        target_angle_adaptation,
        target_angle_generalization
    ])

    phase = np.concatenate([
        np.array(["diagnostic"] * n_diagnostic),
        np.array(["adaptation"] * n_adaptation),
        np.array(["generalization"] * n_generalization)
    ])

    rotation_or_clamp = rotation.copy()
    clamp_trials = feedback_type == "clamp"
    rotation_or_clamp[clamp_trials] = clamp_error[clamp_trials]

    fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(8, 9))
    ax[0, 0].plot(rotation_or_clamp * 180 / np.pi, 'o')
    ax[1, 0].plot(endpoint_visible, 'o')
    ax[2, 0].plot(target_angle, 'o')
    ax[2, 0].set_yticks(np.unique(target_angle))
    ax[0, 0].set_ylabel('rotation / clamp')
    ax[1, 0].set_ylabel('endpoint visible')
    ax[2, 0].set_ylabel('target angle')
    ax[2, 0].set_xlabel('trial')
    plt.show()

    n_trial = rotation.shape[0]
    condition = "nf_burst_after_clamp"
    su = np.ones(n_trial)

    pygame.init()

    # set small window potentially useful for debugging
    # screen_width, screen_height = 800, 600
    # center_x = screen_width // 2
    # center_y = screen_height // 2
    # screen = pygame.display.set_mode((screen_width, screen_height))

    # set full screen
    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h
    center_x = screen_width // 2
    center_y = screen_height // 2
    screen = pygame.display.set_mode((screen_width, screen_height),
                                     pygame.FULLSCREEN)

    # Hide the mouse cursor
    pygame.mouse.set_visible(False)

    # Set up fonts
    font = pygame.font.Font(None, 36)

    # Define colors
    black = (0, 0, 0)
    grey = (128, 128, 128)
    white = (255, 255, 255)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)
    yellow = (255, 255, 0)
    orange = (255, 165, 0)
    green = (0, 255, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)

    # cursor circle radius
    cursor_radius = 8
    start_radius = 15
    target_radius = 15

    n_points = 1

    # relevant coords
    center_x = screen.get_width() // 2
    center_y = screen.get_height() // 2

    start_pos = (center_x, center_y + 2 * px_per_cm)

    # create clocks to keep time
    clock_state = pygame.time.Clock()
    clock_exp = pygame.time.Clock()

    t_state = 0.0
    time_exp = 0.0

    # initial state
    state_init = "state_init"

    # set the current state to the initial state
    state_current = state_init

    # behavioural measurements
    rt = -1
    mt = -1
    ep = -1
    resp = -1

    # record keeping
    trial_data = {
        'date': [],
        'condition': [],
        'subject': [],
        'day': [],
        'trial': [],
        'phase': [],
        'feedback_type': [],
        'target_angle': [],
        'su': [],
        'rotation': [],
        'clamp_error': [],
        'rt': [],
        'mt': [],
        'ep': []
    }

    trial_move = {
        'condition': [],
        'subject': [],
        'day': [],
        'trial': [],
        'state': [],
        't': [],
        'x': [],
        'y': []
    }

    if use_liberty == False:

        # set the current state to the initial state
        state_current = "state_init"

    else:

        rig_coord_upper_left = (0, 0)
        rig_coord_upper_right = (0, 0)
        rig_coord_lower_right = (0, 0)
        rig_coord_lower_left = (0, 0)

        min_x = 1
        max_x = 2
        min_y = 1
        max_y = 2

        calibrating = True
        state_current = "calibrate_upper_left"
        while calibrating:

            screen.fill((0, 0, 0))

            hand_pos = getPosition(ser, recordsize, averager)[0:2]

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        pygame.quit()
                    else:
                        resp = event.key

            if state_current == "calibrate_upper_left":
                t_state += clock_state.tick()
                text = font.render(
                    "Please move to the upper left corner of the screen", True,
                    (255, 255, 255))
                text_rect = text.get_rect(center=(screen_width / 2,
                                                  screen_height / 2))
                screen.fill(black)
                screen.blit(text, text_rect)

                pos_x = 0 + screen_width / 4
                pos_y = 0 + screen_height / 4
                pos = (pos_x, pos_y)
                pygame.draw.circle(screen, white, pos, 15, 0)

                if resp == pygame.K_SPACE:
                    resp = []
                    rig_coord_upper_left = hand_pos
                    state_current = "calibrate_upper_right"

            if state_current == "calibrate_upper_right":
                t_state += clock_state.tick()
                text = font.render(
                    "Please move to the upper right corner of the screen", True,
                    (255, 255, 255))
                text_rect = text.get_rect(center=(screen_width / 2,
                                                  screen_height / 2))
                screen.fill(black)
                screen.blit(text, text_rect)

                pos_x = screen.get_width() - screen_width / 4
                pos_y = screen_height / 4
                pos = (pos_x, pos_y)
                pygame.draw.circle(screen, white, pos, 15, 0)

                if resp == pygame.K_SPACE:
                    resp = []
                    rig_coord_upper_right = hand_pos
                    state_current = "calibrate_lower_right"

            if state_current == "calibrate_lower_right":
                t_state += clock_state.tick()
                text = font.render(
                    "Please move to the lower right corner of the screen", True,
                    (255, 255, 255))
                text_rect = text.get_rect(center=(screen_width / 2,
                                                  screen_height / 2))
                screen.fill(black)
                screen.blit(text, text_rect)

                pos_x = screen_width - screen_width / 4
                pos_y = screen_height - screen_height / 4
                pos = (pos_x, pos_y)
                pygame.draw.circle(screen, white, pos, 15, 0)

                if resp == pygame.K_SPACE:
                    resp = []
                    rig_coord_lower_right = hand_pos
                    state_current = "calibrate_lower_left"

            if state_current == "calibrate_lower_left":
                t_state += clock_state.tick()
                text = font.render(
                    "Please move to the lower left corner of the screen", True,
                    (255, 255, 255))
                text_rect = text.get_rect(center=(screen_width / 2,
                                                  screen_height / 2))

                screen.fill(black)
                screen.blit(text, text_rect)

                pos_x = screen_width / 4
                pos_y = screen_height - screen_height / 4
                pos = (pos_x, pos_y)
                pygame.draw.circle(screen, white, pos, 15, 0)

                if resp == pygame.K_SPACE:
                    resp = []
                    rig_coord_lower_left = hand_pos
                    state_current = "state_init"
                    calibrating = False

                    x_ul, y_ul = rig_coord_upper_left
                    x_ur, y_ur = rig_coord_upper_right
                    x_ll, y_ll = rig_coord_lower_left
                    x_lr, y_lr = rig_coord_lower_right

                    min_x = min(x_ul, x_ur, x_ll, x_lr)
                    max_x = max(x_ul, x_ur, x_ll, x_lr)
                    min_y = min(y_ul, y_ur, y_ll, y_lr)
                    max_y = max(y_ul, y_ur, y_ll, y_lr)

            flipped_screen = pygame.transform.flip(screen, False, True)
            screen.blit(flipped_screen, (0, 0))
            pygame.display.update()

    # set trials / phases
    trial = 0

    running = True
    while running:

        time_exp += clock_exp.tick()
        screen.fill((0, 0, 0))

        rot_mat = np.array([[np.cos(rotation[trial]), -np.sin(rotation[trial])],
                            [np.sin(rotation[trial]),
                             np.cos(rotation[trial])]])

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    pygame.quit()
                else:
                    resp = event.key

        if use_liberty:
            hand_pos = getPosition(ser, recordsize, averager)[0:2]

            x = hand_pos[0]
            y = hand_pos[1]

            x = -((x - min_x) / (max_x - min_x)) + 1
            y = ((y - min_y) / (max_y - min_y)) + 0

            x = x * screen_width
            y = y * screen_height

            hand_pos = (x, y)

        else:
            hand_pos = pygame.mouse.get_pos()

        target_pos_x = -6 * px_per_cm * np.cos(-(target_angle[trial] + 90) * np.pi / 180.0)
        target_pos_y = 6 * px_per_cm * np.sin(-(target_angle[trial] + 90) * np.pi / 180.0)
        target_pos = (start_pos[0] + target_pos_x, start_pos[1] + target_pos_y)
        cursor_pos = np.dot(np.array(hand_pos) - np.array(start_pos),
                            rot_mat) + start_pos

        if state_current == "state_init":
            t_state += clock_state.tick()
            text = font.render("Please press the space bar to begin", True,
                               (255, 255, 255))
            text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
            screen.fill(black)
            screen.blit(text, text_rect)

            if resp == pygame.K_SPACE:
                t_state = 0
                resp = -1
                state_current = "state_searching_ring"

        if state_current == "state_finished":
            t_state += clock_state.tick()
            text = font.render("You finished! Thank you for being awesome!", True,
                               (255, 255, 255))
            text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
            screen.fill(black)
            screen.blit(text, text_rect)

        if state_current == "state_iti":
            t_state += clock_state.tick()
            screen.fill(black)
            if t_state > 1000:
                resp = -1
                rt = -1
                t_state = 0
                trial += 1
                if trial == n_trial:
                    state_current = "state_finished"
                else:
                    state_current = "state_searching_ring"

        if state_current == "state_searching_ring":
            t_state += clock_state.tick()

            r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                        (hand_pos[1] - start_pos[1])**2)

            pygame.draw.circle(screen, blue, start_pos, start_radius)
            pygame.draw.circle(screen, white, start_pos, r, 2)

            if r < 2 * start_radius:
                t_state = 0
                state_current = "state_searching_cursor"

        if state_current == "state_searching_cursor":
            t_state += clock_state.tick()

            r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                        (hand_pos[1] - start_pos[1])**2)

            pygame.draw.circle(screen, blue, start_pos, start_radius)
            pygame.draw.circle(screen, white, hand_pos, cursor_radius)

            if r < start_radius:
                t_state = 0
                state_current = "state_holding"
            elif r >= 2 * start_radius:
                t_state = 0
                state_current = "state_searching_ring"

        if state_current == "state_holding":
            t_state += clock_state.tick()

            r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                        (hand_pos[1] - start_pos[1])**2)

            # smoothly transition from blue to red with
            # increasing time until next state
            if t_state < 2000:
                proportion = t_state / 2000
                red_component = int(255 * proportion)
                blue_component = int(255 * (1 - proportion))
                state_color = (red_component, 0, blue_component)
                pygame.draw.circle(screen, state_color, start_pos, start_radius)
                pygame.draw.circle(screen, white, hand_pos, cursor_radius)

            if r >= start_radius:
                t_state = 0
                state_current = "state_searching_cursor"

            elif t_state > 2000:
                rt = t_state
                t_state = 0
                t_state_2 = 0
                state_current = "state_moving"

        if state_current == "state_moving":
            t_state += clock_state.tick()

            pygame.draw.circle(screen, blue, start_pos, start_radius)
            pygame.draw.circle(screen, red, target_pos, target_radius)

            r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                        (hand_pos[1] - start_pos[1])**2)

            r_target = np.sqrt((target_pos[0] - start_pos[0])**2 +
                               (target_pos[1] - start_pos[1])**2)

            if r >= r_target:
                ep = hand_pos

                ep_theta = np.arctan2(ep[1] - start_pos[1],
                                      ep[0] - start_pos[0])
                if feedback_type[trial] == "clamp":
                    feedback_angle = target_angle[trial] + clamp_error[trial] * 180 / np.pi
                    ep_target_x = -r_target * np.cos(-(feedback_angle + 90) * np.pi / 180.0)
                    ep_target_y = r_target * np.sin(-(feedback_angle + 90) * np.pi / 180.0)
                    ep_target = (start_pos[0] + ep_target_x,
                                 start_pos[1] + ep_target_y)
                else:
                    ep_target = (r_target * np.cos(ep_theta) + start_pos[0],
                                 r_target * np.sin(ep_theta) + start_pos[1])

                mt = t_state

                cloud = np.random.multivariate_normal(
                    ep_target, [[su[trial]**2, 0], [0, su[trial]**2]], n_points)

                # rotate the cloud by the rotation angle
                rot_mat = np.array(
                    [[np.cos(rotation[trial]), -np.sin(rotation[trial])],
                     [np.sin(rotation[trial]),
                      np.cos(rotation[trial])]])

                cloud_rot = np.dot(cloud - start_pos, rot_mat) + start_pos

                t_state = 0
                state_current = "state_feedback_ep"

        if state_current == "state_feedback_ep":
            t_state += clock_state.tick()

            pygame.draw.circle(screen, blue, start_pos, start_radius)
            pygame.draw.circle(screen, red, target_pos, target_radius)

            if endpoint_visible[trial]:
                for i in range(n_points):
                    pygame.draw.circle(screen, white, cloud_rot[i], cursor_radius)

            if t_state > 1000:
                trial_data['date'].append(current_date)
                trial_data['condition'].append(condition)
                trial_data['subject'].append(subject)
                trial_data['day'].append(day)
                trial_data['trial'].append(trial)
                trial_data['phase'].append(phase[trial])
                trial_data['feedback_type'].append(feedback_type[trial])
                trial_data['target_angle'].append(target_angle[trial])
                trial_data['su'].append(np.round(su[trial], 2))
                trial_data['rotation'].append(np.round(rotation[trial], 2))
                trial_data['clamp_error'].append(np.round(clamp_error[trial], 2))
                trial_data['rt'].append(rt)
                trial_data['mt'].append(mt)
                trial_data['ep'].append(np.round(ep_theta, 2))
                pd.DataFrame(trial_data).to_csv(full_path, index=False)
                pd.DataFrame(trial_move).to_csv(full_path_move, index=False)
                t_state = 0
                state_current = "state_iti"

        trial_move['condition'].append(condition)
        trial_move['subject'].append(subject)
        trial_move['day'].append(day)
        trial_move['trial'].append(trial)
        trial_move['state'].append(state_current)
        trial_move['t'].append(time_exp)
        trial_move['x'].append(hand_pos[0])
        trial_move['y'].append(hand_pos[1])

        if use_liberty:
            flipped_screen = pygame.transform.flip(screen, False, True)
            screen.blit(flipped_screen, (0, 0))
            pygame.display.update()
        else:
            pygame.display.flip()

    pygame.quit()
