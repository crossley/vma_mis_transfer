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
    while True:
        rotation_direction = input("Rotation direction (CW/CCW): ").strip().upper()
        if rotation_direction in {"CW", "CCW"}:
            break
        print("Please enter CW or CCW.")

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
        # Set variables
        # This defines the length of the binary header (bytes 0-7)
        header = 8
        # This defines the bytesize of IEEE floating point
        byte_size = 4
        expected_size = header + 3 * byte_size

        # Obtain data
        ser.reset_input_buffer()
        ser.write(b"P")

        deadline = time.time() + 0.1
        while ser.inWaiting() < expected_size:
            if time.time() > deadline:
                raise RuntimeError(
                    "Timed out waiting for Liberty frame: "
                    f"{ser.inWaiting()}/{expected_size} bytes available")
            time.sleep(0.001)

        # Read header to remove it from the input buffer
        header_bytes = ser.read(header)
        if len(header_bytes) != header:
            raise RuntimeError(
                f"Incomplete Liberty header: {len(header_bytes)} bytes")

        positions = []

        # Read the three coordinates
        for x in range(3):
            # Read the coordinate
            coord = ser.read(byte_size)
            if len(coord) != byte_size:
                raise RuntimeError(
                    f"Incomplete Liberty coordinate: {len(coord)} bytes")

            # Convert hex to floating point (little endian order)
            coord = struct.unpack("<f", coord)[0]

            positions.append(coord)

        return positions


    def capture_stable_position(ser, recordsize, averager, n_samples=20):
        """Return a robust position estimate and within-burst movement in cm."""
        samples = []
        attempts = 0
        max_attempts = n_samples * 3

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            try:
                sample = getPosition(ser, recordsize, averager)[0:2]
            except RuntimeError as err:
                print(err)
                continue

            if np.all(np.isfinite(sample)):
                samples.append(sample)

        if len(samples) < max(5, n_samples // 2):
            return np.array([np.nan, np.nan]), np.inf

        samples = np.array(samples)
        position = np.median(samples, axis=0)
        radial_error = np.linalg.norm(samples - position, axis=1)

        clean = radial_error < 5.0
        if np.sum(clean) >= max(5, n_samples // 2):
            samples = samples[clean]
            position = np.median(samples, axis=0)
            radial_error = np.linalg.norm(samples - position, axis=1)

        movement = np.percentile(radial_error, 90)
        return position, movement


    def fit_affine_calibration(sensor_points, screen_points):
        """Fit [sensor_x, sensor_y, 1] @ transform = [screen_x, screen_y]."""
        sensor_design = np.column_stack([
            np.asarray(sensor_points),
            np.ones(len(sensor_points))
        ])
        transform, _, rank, _ = np.linalg.lstsq(
            sensor_design, np.asarray(screen_points), rcond=None)
        if rank < 3:
            raise ValueError("Calibration points do not span a 2D area")

        predicted = sensor_design @ transform
        errors = np.linalg.norm(predicted - screen_points, axis=1)
        return transform, errors


    def sensor_to_screen(sensor_position, transform):
        sensor_homogeneous = np.append(np.asarray(sensor_position), 1.0)
        return tuple(sensor_homogeneous @ transform)


    def rotate_screen_points(points, origin, angle):
        """Rotate screen points; positive angles are visually clockwise."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        origin = np.asarray(origin, dtype=float)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return (rotation_matrix @ (points - origin).T).T + origin


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

    target_angles = np.arange(-150, 151, 30)

    rng = np.random.default_rng()

    def shuffled_target_cycles(n_cycles):
        return np.concatenate([
            rng.permutation(target_angles) for _ in range(n_cycles)
        ])

    def shuffled_baseline_schedule():
        """Return 18 trials/target: 12 online and 6 without feedback."""
        schedule = []
        for angle in target_angles:
            schedule.extend([(angle, "online")] * 12)
            schedule.extend([(angle, "none")] * 6)
        rng.shuffle(schedule)
        angles, feedback = zip(*schedule)
        return np.array(angles), np.array(feedback)

    # Hewitson, Crossley, and Kaplan (2020): 33 familiarisation
    # reaches, three to each of 11 targets, with veridical cursor
    # feedback throughout the reach.
    n_familiarisation = 33
    target_angle_familiarisation = shuffled_target_cycles(3)
    feedback_type_familiarisation = np.array(
        ["online"] * n_familiarisation)
    endpoint_visible_familiarisation = np.ones(n_familiarisation)
    rotation_familiarisation = np.zeros(n_familiarisation)
    clamp_error_familiarisation = np.full(n_familiarisation, np.nan)

    # Baseline contains 18 reaches per target. Each target has 12
    # trials with continuous veridical feedback and six with no
    # visual feedback, shuffled together across the phase.
    n_baseline = 198
    target_angle_baseline, feedback_type_baseline = (
        shuffled_baseline_schedule())
    endpoint_visible_baseline = (feedback_type_baseline != "none").astype(float)
    rotation_baseline = np.zeros(n_baseline)
    clamp_error_baseline = np.full(n_baseline, np.nan)

    # Adaptation contains 110 reaches to the straight-ahead target.
    # Endpoint-only feedback is rotated 30 degrees; positive angles
    # are clockwise in the screen/target coordinate convention.
    n_adaptation = 110
    target_angle_adaptation = np.zeros(n_adaptation, dtype=int)
    feedback_type_adaptation = np.array(["rotated"] * n_adaptation)
    endpoint_visible_adaptation = np.ones(n_adaptation)
    signed_rotation_deg = 30 if rotation_direction == "CW" else -30
    rotation_adaptation = np.full(
        n_adaptation, signed_rotation_deg * np.pi / 180)
    clamp_error_adaptation = np.full(n_adaptation, np.nan)

    # Generalisation contains six shuffled cycles through all 11
    # target directions, with no visual feedback on any trial.
    n_generalization = 66
    target_angle_generalization = shuffled_target_cycles(6)
    feedback_type_generalization = np.array(["none"] * n_generalization)
    endpoint_visible_generalization = np.zeros(n_generalization)
    rotation_generalization = np.zeros(n_generalization)
    clamp_error_generalization = np.full(n_generalization, np.nan)

    # concatenate all phases
    endpoint_visible = np.concatenate([
        endpoint_visible_familiarisation,
        endpoint_visible_baseline,
        endpoint_visible_adaptation,
        endpoint_visible_generalization
    ])

    rotation = np.concatenate([
        rotation_familiarisation,
        rotation_baseline,
        rotation_adaptation,
        rotation_generalization
    ])

    clamp_error = np.concatenate([
        clamp_error_familiarisation,
        clamp_error_baseline,
        clamp_error_adaptation,
        clamp_error_generalization
    ])

    feedback_type = np.concatenate([
        feedback_type_familiarisation,
        feedback_type_baseline,
        feedback_type_adaptation,
        feedback_type_generalization
    ])

    target_angle = np.concatenate([
        target_angle_familiarisation,
        target_angle_baseline,
        target_angle_adaptation,
        target_angle_generalization
    ])

    phase = np.concatenate([
        np.array(["familiarisation"] * n_familiarisation),
        np.array(["baseline"] * n_baseline),
        np.array(["adaptation"] * n_adaptation),
        np.array(["generalization"] * n_generalization)
    ])

    fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(8, 9))
    ax[0, 0].plot(rotation * 180 / np.pi, 'o')
    ax[1, 0].plot(endpoint_visible, 'o')
    ax[2, 0].plot(target_angle, 'o')
    ax[2, 0].set_yticks(np.unique(target_angle))
    ax[0, 0].set_ylabel('rotation')
    ax[1, 0].set_ylabel('endpoint visible')
    ax[2, 0].set_ylabel('target angle')
    ax[2, 0].set_xlabel('trial')
    plt.show()

    n_trial = rotation.shape[0]
    condition = "hewitson_original"
    su = np.zeros(n_trial)

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
    movement_started = False
    movement_start_time = 0

    # record keeping
    trial_data = {
        'date': [],
        'condition': [],
        'subject': [],
        'day': [],
        'rotation_direction': [],
        'trial': [],
        'phase': [],
        'feedback_type': [],
        'target_angle': [],
        'su': [],
        'rotation': [],
        'rotation_deg': [],
        'clamp_error': [],
        'clamp_error_deg': [],
        'rt': [],
        'mt': [],
        'ep': []
    }

    trial_move = {
        'condition': [],
        'subject': [],
        'day': [],
        'rotation_direction': [],
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
        inset_x = screen_width / 4
        inset_y = screen_height / 4
        calibration_targets = [
            (center_x, center_y),
            (inset_x, inset_y),
            (screen_width - inset_x, inset_y),
            (screen_width - inset_x, screen_height - inset_y),
            (inset_x, screen_height - inset_y),
        ]
        first_pass = np.arange(len(calibration_targets))
        second_pass = np.random.default_rng().permutation(
            len(calibration_targets))
        if second_pass[0] == first_pass[-1]:
            second_pass = np.roll(second_pass, 1)
        calibration_order = np.concatenate([first_pass, second_pass])
        sensor_points = []
        calibration_index = 0
        calibration_message = ""
        calibration_message_until = 0
        calibration_transform = None
        calibrating = True
        abort_calibration = False

        while calibrating:
            clock_state.tick(60)
            screen.fill(black)

            target_index = calibration_order[calibration_index]
            prompt = font.render(
                "Hold still on point "
                f"{calibration_index + 1}/{len(calibration_order)}, then press SPACE",
                True, white)
            prompt_rect = prompt.get_rect(
                center=(screen_width / 2, screen_height / 8))
            screen.blit(prompt, prompt_rect)
            pygame.draw.circle(
                screen, white, calibration_targets[target_index], 15, 0)

            if pygame.time.get_ticks() < calibration_message_until:
                message = font.render(calibration_message, True, yellow)
                message_rect = message.get_rect(
                    center=(screen_width / 2, 7 * screen_height / 8))
                screen.blit(message, message_rect)

            flipped_screen = pygame.transform.flip(screen, False, True)
            screen.blit(flipped_screen, (0, 0))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    abort_calibration = True
                    calibrating = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        abort_calibration = True
                        calibrating = False
                    elif event.key == pygame.K_r:
                        sensor_points = []
                        calibration_index = 0
                        calibration_message = "Calibration restarted"
                        calibration_message_until = pygame.time.get_ticks() + 1200
                    elif event.key == pygame.K_SPACE:
                        position, movement = capture_stable_position(
                            ser, recordsize, averager)
                        if movement > 0.15:
                            calibration_message = (
                                "Too much movement; hold still and try this point again")
                            calibration_message_until = pygame.time.get_ticks() + 1500
                            continue

                        sensor_points.append(position)
                        calibration_index += 1

                        if calibration_index == len(calibration_order):
                            try:
                                ordered_targets = np.array([
                                    calibration_targets[i]
                                    for i in calibration_order
                                ])
                                candidate, errors = fit_affine_calibration(
                                    sensor_points, ordered_targets)
                                errors_cm = errors / px_per_cm
                                transformed_points = np.array([
                                    sensor_to_screen(point, candidate)
                                    for point in sensor_points
                                ])
                                repeat_errors_cm = []
                                for i in range(len(calibration_targets)):
                                    repeated = transformed_points[
                                        calibration_order == i]
                                    repeat_errors_cm.append(
                                        np.linalg.norm(repeated[0] - repeated[1]) /
                                        px_per_cm)
                                if (np.mean(errors_cm) <= 0.35 and
                                        np.max(errors_cm) <= 0.75 and
                                        np.max(repeat_errors_cm) <= 1.0):
                                    calibration_transform = candidate
                                    state_current = "state_init"
                                    calibrating = False
                                    print(
                                        "Calibration accepted: "
                                        f"mean error={np.mean(errors_cm):.2f} cm, "
                                        f"max error={np.max(errors_cm):.2f} cm, "
                                        "max repeat difference="
                                        f"{np.max(repeat_errors_cm):.2f} cm")
                                else:
                                    sensor_points = []
                                    calibration_index = 0
                                    calibration_message = (
                                        "Fit was inaccurate; calibration restarted")
                                    calibration_message_until = (
                                        pygame.time.get_ticks() + 2000)
                            except ValueError:
                                sensor_points = []
                                calibration_index = 0
                                calibration_message = (
                                    "Invalid point geometry; calibration restarted")
                                calibration_message_until = (
                                    pygame.time.get_ticks() + 2000)

        if abort_calibration:
            pygame.quit()
            ser.close()
            sys.exit()

    # set trials / phases
    trial = 0
    break_duration_ms = 60_000
    break_before_trial = {
        n_familiarisation: "Baseline",
        n_familiarisation + n_baseline: "Adaptation"
    }

    running = True
    while running:

        time_exp += clock_exp.tick()
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    pygame.quit()
                else:
                    resp = event.key

        if not running:
            break

        if use_liberty:
            hand_pos = getPosition(ser, recordsize, averager)[0:2]
            hand_pos = sensor_to_screen(hand_pos, calibration_transform)

        else:
            hand_pos = pygame.mouse.get_pos()

        if trial < n_trial:
            target_pos_x = -6 * px_per_cm * np.cos(
                -(target_angle[trial] + 90) * np.pi / 180.0)
            target_pos_y = 6 * px_per_cm * np.sin(
                -(target_angle[trial] + 90) * np.pi / 180.0)
            target_pos = (start_pos[0] + target_pos_x,
                          start_pos[1] + target_pos_y)

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
                elif trial in break_before_trial:
                    state_current = "state_break"
                else:
                    state_current = "state_searching_ring"

        if state_current == "state_break":
            t_state += clock_state.tick()
            screen.fill(black)
            seconds_left = max(
                0, int(np.ceil((break_duration_ms - t_state) / 1000)))
            break_text = font.render(
                f"Please rest. {break_before_trial[trial]} begins in "
                f"{seconds_left} seconds.", True, white)
            break_rect = break_text.get_rect(
                center=(screen_width / 2, screen_height / 2))
            screen.blit(break_text, break_rect)
            if t_state >= break_duration_ms:
                t_state = 0
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
                rt = -1
                t_state = 0
                movement_started = False
                movement_start_time = 0
                state_current = "state_moving"

        if state_current == "state_moving":
            t_state += clock_state.tick()

            pygame.draw.circle(screen, blue, start_pos, start_radius)
            pygame.draw.circle(screen, red, target_pos, target_radius)

            r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                        (hand_pos[1] - start_pos[1])**2)

            # On non-online trials, retain the veridical cursor until the
            # hand leaves the start target, then withhold it during the reach.
            if feedback_type[trial] == "online" or r < start_radius:
                cursor_rotation = (
                    rotation[trial]
                    if feedback_type[trial] == "online" else 0)
                cursor_pos = rotate_screen_points(
                    hand_pos, start_pos, cursor_rotation)[0]
                pygame.draw.circle(
                    screen, white, cursor_pos, cursor_radius)

            r_target = np.sqrt((target_pos[0] - start_pos[0])**2 +
                               (target_pos[1] - start_pos[1])**2)

            if not movement_started and r >= start_radius:
                rt = t_state
                movement_start_time = t_state
                movement_started = True

            if r >= r_target:
                ep = hand_pos

                ep_theta = np.arctan2(ep[1] - start_pos[1],
                                      ep[0] - start_pos[0])
                ep_target = (r_target * np.cos(ep_theta) + start_pos[0],
                             r_target * np.sin(ep_theta) + start_pos[1])

                if not movement_started:
                    rt = t_state
                    movement_start_time = 0
                mt = t_state - movement_start_time

                feedback_points = np.repeat(
                    np.asarray(ep_target)[None, :], n_points, axis=0)
                feedback_points_rot = rotate_screen_points(
                    feedback_points, start_pos, rotation[trial])

                t_state = 0
                state_current = "state_feedback_ep"

        if state_current == "state_feedback_ep":
            t_state += clock_state.tick()

            pygame.draw.circle(screen, blue, start_pos, start_radius)
            pygame.draw.circle(screen, red, target_pos, target_radius)

            if endpoint_visible[trial]:
                for i in range(n_points):
                    pygame.draw.circle(
                        screen, white, feedback_points_rot[i], cursor_radius)

            if t_state > 1000:
                trial_data['date'].append(current_date)
                trial_data['condition'].append(condition)
                trial_data['subject'].append(subject)
                trial_data['day'].append(day)
                trial_data['rotation_direction'].append(rotation_direction)
                trial_data['trial'].append(trial)
                trial_data['phase'].append(phase[trial])
                trial_data['feedback_type'].append(feedback_type[trial])
                trial_data['target_angle'].append(target_angle[trial])
                trial_data['su'].append(np.round(su[trial], 2))
                trial_data['rotation'].append(np.round(rotation[trial], 2))
                trial_data['rotation_deg'].append(np.round(rotation[trial] * 180 / np.pi, 2))
                trial_data['clamp_error'].append(np.round(clamp_error[trial], 2))
                trial_data['clamp_error_deg'].append(np.round(clamp_error[trial] * 180 / np.pi, 2))
                trial_data['rt'].append(rt)
                trial_data['mt'].append(mt)
                trial_data['ep'].append(np.round(ep_theta, 2))
                pd.DataFrame(trial_data).to_csv(full_path, index=False)
                pd.DataFrame(trial_move).to_csv(full_path_move, index=False)
                t_state = 0
                state_current = "state_iti"

        if trial < n_trial:
            trial_move['condition'].append(condition)
            trial_move['subject'].append(subject)
            trial_move['day'].append(day)
            trial_move['rotation_direction'].append(rotation_direction)
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
