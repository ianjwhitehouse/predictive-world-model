# Make spaceships faster
# Better collisions
# turn only at speed

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import pygame
from time import time
import pandas as pd

PI = 3.14593
QUARTER_CIRCLE=PI/2
SPACE_SHIP_SHAPE = tf.constant([[20.0], [7], [7]]) # dist from center to bow, port and starboard
FRAME_RATE = 20
TAGGER_SPEED = 200.0/FRAME_RATE
TAGGEE_SPEED = 180.0/FRAME_RATE
ROT_SPEED = ((2*PI)/FRAME_RATE)/2 # The last number is the number of turns for a full circle
MIN_SECONDS = 3

@tf.function
def run_frame(tagger_input, taggee_input,
              tagger_pos, taggee_pos,
              tagger_speed=tf.zeros((2,)), taggee_speed=tf.zeros((2,)),
              tagger_heading=tf.constant(0.0), taggee_heading=tf.constant(0.0),
              obstacles=tf.zeros((0, 3)),
              momentum=False
    ):
    # Function to run and update a frame of the game
    # Args
    # Tagger_input and taggee_input: Tensor of shape (4,) with values in Up, Right, Down, Left order
    # Tagger_pos, taggee_pos: Tensor of shape (2, 1) with values x, y
    # Tagger_speed, taggee_speed: Tensor of shape (2, 1) with values x_speed, y_speed
    # Tagger_heading and taggee_heading: Tensor of shape (1,) with value radian heading
    # Obstacles: Tensor of shape (Num_obstacles, 4) with values x, y, radius
    # Momentum: True or false of whether to include momentum in calculations
    # Returns
    # Tagger_pos, taggee_pos, tagger_speed, taggee_speed, tagger_heading, taggee_heading, flags
    # Flag: Bool tensor of shape (,) with values tagged, tagger_hit_obstacle, taggee_hit_obstacle
    
    # Update position
    tagger_pos += tagger_speed
    taggee_pos += taggee_speed

    # Update rotation
    tagger_heading = update_rotation(tagger_input, tagger_heading, tagger_speed)
    taggee_heading = update_rotation(taggee_input, taggee_heading, taggee_speed)

    # Update speed
    if momentum:
        tagger_speed = update_speed(tagger_input, tagger_speed, tagger_heading, TAGGER_SPEED)
        taggee_speed = update_speed(taggee_input, taggee_speed, taggee_heading, TAGGEE_SPEED)
    else:
        tagger_speed = update_speed(tagger_input, tf.zeros((2,)), tagger_heading, TAGGER_SPEED)
        taggee_speed = update_speed(taggee_input, tf.zeros((2,)), taggee_heading, TAGGEE_SPEED)
    
    return tagger_pos, taggee_pos, tagger_speed, taggee_speed, tagger_heading, taggee_heading


def update_rotation(input, heading, speed):
    # Function to update the rotation of a spaceship
    # Args
    # Input: Tensor of shape (4,) with values in Up, Right, Down, Left order
    # Heading: Tensor of shape (1,) with value radian heading
    # Speed: Tensor of shape (2, 1) with values x_speed, y_speed
    # Returns new heading
    if tf.math.reduce_sum(tf.math.abs(speed)) < 0.1:
        return heading
    normed_input = input * tf.constant([0.0, ROT_SPEED, 0, -1 * ROT_SPEED])
    return (heading + tf.math.reduce_sum(normed_input)) % (2 * PI)


def update_speed(input, prev_speed, heading, speed):
    # Function to update the speed of a spaceship
    # Args
    # Input: Tensor of shape (4,) with values in Up, Right, Down, Left order
    # Prev_speed: Tensor of shape (2, 1) with values x_speed, y_speed
    # Heading: Tensor of shape (1,) with value radian heading
    # Speed: Float speed
    normed_input = tf.math.reduce_sum(input * tf.constant([1.0 * speed, 0, -1 * speed, 0]))

    input_in_heading = tf.constant([1.0, 0]) * tf.math.cos(heading) * normed_input
    input_in_heading += tf.constant([0.0, 1]) * tf.math.sin(heading) * normed_input

    return prev_speed + input_in_heading


def pred_model(input, model):
    return model(tf.expand_dims(input, axis=0))


def pred_model_lr(input, model, temperature=1.0):
    steering = model(tf.expand_dims(input, axis=0))[0, :]
    steering += temperature * tf.random.normal((1,))
    steering *= tf.constant([-1.0, 1])
    return tf.math.maximum(steering, 0.0)
    

def random_inputs(_):
    return tf.random.uniform((4,))

def random_inputs_lr(_, frame_num=0, hold_rand_inp_for=1):
    steering = (tf.random.uniform((1,)) * 2.0) - 1.0
    steering *= tf.constant([-1.0, 1])
    return tf.math.maximum(steering, 0.0)


def key_input_wasd(_):
    keys = pygame.key.get_pressed()
    return tf.cast(tf.constant([keys[pygame.K_w], keys[pygame.K_d], keys[pygame.K_s], keys[pygame.K_a]]), float)


def key_input_arrows(_):
    keys = pygame.key.get_pressed()
    return tf.cast(tf.constant([keys[pygame.K_UP], keys[pygame.K_RIGHT], keys[pygame.K_DOWN], keys[pygame.K_LEFT]]), float)


@tf.function
def get_cos_and_sins(heading):
    # Function to calculate the sin and cosines to draw the spaceship
    # Args
    # Heading: Tensor of shape (1,) with value radian heading
    # Returns sin and cos for forward, left, and right
    cos_mask = tf.constant([[1.0, 0], [1, 0], [1, 0]])
    sin_mask = tf.constant([[0.0, 1], [0, 1], [0, 1]])

    heading += tf.constant([0.0, QUARTER_CIRCLE, 3 * QUARTER_CIRCLE])
    heading = tf.expand_dims(heading, axis=-1)

    return (cos_mask * tf.math.cos(heading)) + (sin_mask * tf.math.sin(heading))


def game_loop(input_device_tagger, input_device_taggee, num_obstacles=0, momentum=False, size=512, hidden=False, framerate=30, num_runs=500):
    # Function to run the game
    # Args
    # Input_device_tagger, input_device_taggee: functions that are inputs to the game and return tensor of shape (4,) with values in Up, Right, Down, Left order
    # Run_for: number of games to run for
    # Num_obstacles: number of obstacles to add
    # Momentum: whether to add momentum to the ships
    # Size: size of frame
    # Hidden: actually draw or not
    # Framerate: when not hidden, maximum frames per second
    
    pygame.init()
    if hidden:
        screen = pygame.display.set_mode((size, size), flags=pygame.HIDDEN)
    else:
        screen = pygame.display.set_mode((size, size))
    clock = pygame.time.Clock()
    
    succ_runs = 0
    
    while succ_runs < num_runs:
        run_name = str(int(time()))
        
        tagger_poses = []
        tagger_pos = tf.random.uniform((2,), maxval=size)
        # This code was used when I wanted them to always start around a single circle
        # tagger_pos = tf.random.uniform((1,), maxval=2 * PI)
        # tagger_pos = (tf.math.cos(tagger_pos) * tf.constant([1.0,0])) + (tf.math.sin(tagger_pos) * tf.constant([0.0,1]))
        # tagger_pos = tagger_pos * tf.constant([size/4, size/4]) + tf.constant([size/2, size/2])

        taggee_poses = []
        taggee_pos = tf.random.uniform((2,), maxval=size)
        # This code was used when I wanted them to always start around a single circle
        # taggee_pos = tf.random.uniform((1,), maxval=2 * PI)
        # taggee_pos = (tf.math.cos(taggee_pos) * tf.constant([1.0,0])) + (tf.math.sin(taggee_pos) * tf.constant([0.0,1]))
        # taggee_pos = taggee_pos * tf.constant([size/3, size/3]) + tf.constant([size/2, size/2])

        tagger_headings = []
        tagger_heading = tf.random.uniform((1,), maxval=2 * PI)
        taggee_headings = []
        taggee_heading = tf.random.uniform((1,), maxval=2 * PI)

        tagger_speeds = []
        tagger_speed=tf.zeros((2,))
        taggee_speeds = []
        taggee_speed=tf.zeros((2,))

        obstacles = tf.random.truncated_normal((num_obstacles, 2), mean=size/2, stddev=size/4)
        obstacles = tf.concat([obstacles, tf.random.truncated_normal((num_obstacles, 1), mean=size/20, stddev=size/15)], axis=-1)

        frames = []
        tagger_inputs = []
        taggee_inputs = []

        while True:
            screen.fill("black")

            for obstacle in (tf.cast(obstacles * 10.0, tf.int32) / 10).numpy():
                pygame.draw.circle(screen, "blue", obstacle[:2], obstacle[2], width=5)

            angles = get_cos_and_sins(tagger_heading)
            tagger_pos_draw_coords = tf.expand_dims(tagger_pos, axis=0) + (SPACE_SHIP_SHAPE * angles)
            tagger_rect = pygame.draw.polygon(screen, "red", (tf.cast(tagger_pos_draw_coords * 10.0, tf.int32) / 10).numpy(), width=5)

            angles = get_cos_and_sins(taggee_heading)
            taggee_pos_draw_coords = tf.expand_dims(taggee_pos, axis=0) + (SPACE_SHIP_SHAPE * angles)
            taggee_rect = pygame.draw.polygon(screen, "green", (tf.cast(taggee_pos_draw_coords * 10.0, tf.int32) / 10).numpy(), width=5)
            
            pygame.display.flip()
            pygame.event.pump()

            if not hidden:
                clock.tick(framerate)
            
            frame = tf.constant(pygame.surfarray.array3d(screen)/255.0, float)
            dense_to_sparse = lambda x: tf.SparseTensor(
                tf.where(tf.not_equal(x, tf.constant(0.0))),
                tf.gather_nd(x, tf.where(tf.not_equal(x, tf.constant(0.0)))),
                x.shape
            )
            frame = dense_to_sparse(frame)
            
            tagger_input = input_device_tagger(frame)
            taggee_input = input_device_taggee(frame)

            # Constrain inputs
            tagger_input = tf.math.maximum(tf.math.minimum(tagger_input, 1.0), 0.0)
            taggee_input = tf.math.maximum(tf.math.minimum(taggee_input, 1.0), 0.0)
            
            frames.append(frame)
            tagger_inputs.append(tagger_input)
            taggee_inputs.append(taggee_input)
            tagger_speeds.append(tagger_speed)
            taggee_speeds.append(taggee_speed)
            tagger_headings.append(tagger_heading)
            taggee_headings.append(taggee_heading)
            tagger_poses.append(tagger_pos)
            taggee_poses.append(taggee_pos)

            # print(tagger_pos)
            # print(taggee_pos)

            # So that we always go forward 1
            tagger_input = tagger_input.numpy()
            tagger_input = tf.constant([.25, tagger_input[0], 0, tagger_input[1]], float)

            taggee_input = taggee_input.numpy()
            taggee_input = tf.constant([.25, taggee_input[0], 0, taggee_input[1]], float)
            
            
            tagger_pos, taggee_pos, tagger_speed, taggee_speed, tagger_heading, taggee_heading = run_frame(
                tagger_input, taggee_input,
                tagger_pos, taggee_pos,
                tagger_speed=tagger_speed, taggee_speed=taggee_speed,
                tagger_heading=tagger_heading, taggee_heading=taggee_heading,
                obstacles=obstacles, momentum=momentum
            )

            flags = tf.constant([
                tagger_rect.colliderect(taggee_rect),
            ])
            dist = lambda obj: tf.math.sqrt(tf.math.square(obj[0] - obstacles[:, 0]) + (tf.math.square(obj[1] - obstacles[:, 1]))) <= obstacles[:, 2]
            flags = tf.concat([
                flags,
                tf.expand_dims(tf.math.reduce_any(dist(tf.expand_dims(tagger_pos, axis=-1))), axis=0),
                tf.expand_dims(tf.math.reduce_any(dist(tf.expand_dims(taggee_pos, axis=-1))), axis=0),
                tf.expand_dims(tf.math.reduce_any(tagger_pos < 0) or tf.math.reduce_any(tagger_pos > size), axis=0),
                tf.expand_dims(tf.math.reduce_any(taggee_pos < 0) or tf.math.reduce_any(taggee_pos > size), axis=0)
            ], axis=0)

            # Print update
            if len(frames) % 150 == 0:
                print(len(frames))
            
            if tf.math.reduce_any(flags) or len(frames) > 1500:
                if len(frames) < framerate * MIN_SECONDS or len(frames) > 1500:
                    print("RUN NOT SAVED: ", len(frames))
                    break
                print("RUN SAVED")
                succ_runs += 1
                
                reason = tf.math.argmax(flags)
                winner = ["TAGGER", "TAGGEE", "TAGGER", "TAGGEE", "TAGGER"][reason]
                reason = ["TAGGED", "TAGGER_HIT_OBS", "TAGGEE_HIT_OBS", "TAGGER_OOB", "TAGGEE_OOB"][reason]
                
                write_tensor = lambda name, x: tf.io.write_file("runs/%s/%s.proto_tensor" % (run_name, name), tf.io.serialize_tensor(tf.stack(x)))

                write_tensor("tagger_inputs", tagger_inputs)
                write_tensor("tagee_inputs", taggee_inputs)
                write_tensor("tagger_speeds", tagger_speeds)
                write_tensor("taggee_speeds", taggee_speeds)
                write_tensor("tagger_headings", tagger_headings)
                write_tensor("taggee_headings", taggee_headings)
                write_tensor("tagger_poses", tagger_poses)
                write_tensor("taggee_poses", taggee_poses)
                write_tensor("obsticles", obstacles)

                # Write frames as sparse
                
                frames = [tf.sparse.expand_dims(frame, axis=0) for frame in frames]
                write_tensor("frames", tf.io.serialize_many_sparse(tf.sparse.concat(0, frames)))
                # tf.io.write_file("runs/%s/frames.proto_tensor" % run_name, tf.io.serialize_many_sparse(tf.sparse.concat(0, frames)))
                
                try:
                    passed_runs = pd.read_csv("game_runs.csv", index_col=0).values.tolist()
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    passed_runs = []

                avg_speed = lambda x: tf.math.reduce_mean(tf.math.abs(tf.map_fn(lambda y: tf.math.sqrt(tf.math.square(y[0]) + tf.math.square(y[1])), x, fn_output_signature=float)))
                euclid_dist = lambda x: tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x[-1] - x[0])))
              
                pd.DataFrame(
                    passed_runs + [(
                        run_name, size, num_obstacles, momentum, avg_speed(tf.stack(tagger_speeds)).numpy(), avg_speed(tf.stack(taggee_speeds)).numpy(),
                        euclid_dist(tf.stack(tagger_poses)).numpy(), euclid_dist(tf.stack(taggee_poses)).numpy(),
                        "%.2f/%.2f" % (tf.math.reduce_mean(tf.stack(tagger_inputs), axis=0)[0].numpy(), tf.math.reduce_mean(tf.stack(tagger_inputs), axis=0)[1].numpy()), 
                        "%.2f/%.2f" % (tf.math.reduce_mean(tf.stack(taggee_inputs), axis=0)[0].numpy(), tf.math.reduce_mean(tf.stack(taggee_inputs), axis=0)[1].numpy()),
                        len(frames), reason, winner
                    )],
                    columns = ["Run", "Window Size", "Number of Obstacles", "Momentum Enabled", "Tagger Average Speed", "Taggee Average Speed", "Tagger Distance Covered", "Taggee Distance Covered", "Tagger Avg Left/Avg Right", "Taggee Avg Left/Avg Right", "Run Length", "End State", "Winner"]
                ).to_csv("game_runs.csv")
                break

# tagger_model = tf.keras.models.load_model("action_gen_TAGGER.tf")
# tagger_model.compile()
# tagger_inputs = lambda x: pred_model_lr(x, tagger_model)
tagger_inputs = lambda x: random_inputs_lr(x)

# taggee_model = tf.keras.models.load_model("action_gen_TAGGEE.tf")
# tagger_model.compile()
# taggee_inputs = lambda x: pred_model_lr(x, taggee_model)
taggee_inputs = lambda x: random_inputs_lr(x)

game_loop(tagger_inputs, taggee_inputs, num_obstacles=10, momentum=False, size=640, hidden=True, framerate=FRAME_RATE)
