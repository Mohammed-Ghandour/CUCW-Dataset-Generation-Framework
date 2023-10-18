#!/usr/bin/env python3

import glob
import os
import sys
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
#from modules import waymo_frame_format

try:
    sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        "C:/CARLA_0.9.10/WindowsNoEditor" if os.name == 'nt' else str(Path.home()) + "/CARLA_0.9.10",
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    
    sys.path.append(glob.glob('../../')[0])

except IndexError:
    pass

import carla
from carla import VehicleLightState as vls

import logging
import queue
import struct
import math
import numpy as np
import random
import threading
import string

def sensor_callback(ts, sensor_data, sensor_queue):
    sensor_queue.put(sensor_data)

class Sensor:
    initial_ts = 0.0
    initial_loc = carla.Location()
    initial_rot = carla.Rotation()

    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        self.queue = queue.Queue()
        self.bp = self.set_attributes(world.get_blueprint_library())
        self.sensor = world.spawn_actor(self.bp, transform, attach_to=vehicle)
        actor_list.append(self.sensor)
        self.sensor.listen(lambda data: sensor_callback(data.timestamp - Sensor.initial_ts, data, self.queue))
        self.sensor_id = self.__class__.sensor_id_glob;
        self.__class__.sensor_id_glob += 1
        self.folder_output = folder_output
        self.ts_tmp = 0

class Camera(Sensor):
    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        Sensor.__init__(self, vehicle, world, actor_list, folder_output, transform)
        self.sensor_frame_id = 0
        self.frame_output = self.folder_output+"/images_%s" %str.lower(self.__class__.__name__)
        os.makedirs(self.frame_output) if not os.path.exists(self.frame_output) else [os.remove(f) for f in glob.glob(self.frame_output+"/*") if os.path.isfile(f)]

        with open(self.folder_output+"/full_ts_camera.txt", 'w') as file:
            file.write("# frame_id timestamp\n")

        print('created %s' % self.sensor)

    def save(self, color_converter=carla.ColorConverter.Raw):
        while not self.queue.empty():
            data = self.queue.get()

            ts = data.timestamp-Sensor.initial_ts
            if ts - self.ts_tmp > 0.11 or (ts - self.ts_tmp) < 0: #check for 10Hz camera acquisition
                print("[Error in timestamp] Camera: previous_ts %f -> ts %f" %(self.ts_tmp, ts))
                sys.exit()
            self.ts_tmp = ts

            file_path = self.frame_output+"/%04d_%d.png" %(self.sensor_frame_id, self.sensor_id)
            x = threading.Thread(target=data.save_to_disk, args=(file_path, color_converter))
            x.start()
            print("Export : "+file_path)

            if self.sensor_id == 0:
                with open(self.folder_output+"/full_ts_camera.txt", 'a') as file:
                    file.write(str(self.sensor_frame_id)+" "+str(data.timestamp - Sensor.initial_ts)+"\n") #bug in CARLA 0.9.10: timestamp of camera is one tick late. 1 tick = 1/fps_simu seconds
            self.sensor_frame_id += 1

class RGB(Camera):
    sensor_id_glob = 0

    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        Camera.__init__(self, vehicle, world, actor_list, folder_output, transform)

    def set_attributes(self, blueprint_library):
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        camera_bp.set_attribute('image_size_x', '1392')
        camera_bp.set_attribute('image_size_y', '1024')
        camera_bp.set_attribute('fov', '72') #72 degrees # Always fov on width even if width is different than height
        camera_bp.set_attribute('enable_postprocess_effects', 'True')
        camera_bp.set_attribute('sensor_tick', '0.10') # 10Hz camera
        camera_bp.set_attribute('gamma', '2.2')
        camera_bp.set_attribute('motion_blur_intensity', '0')
        camera_bp.set_attribute('motion_blur_max_distortion', '0')
        camera_bp.set_attribute('motion_blur_min_object_screen_size', '0')
        camera_bp.set_attribute('shutter_speed', '1000') #1 ms shutter_speed
        camera_bp.set_attribute('lens_k', '0')
        camera_bp.set_attribute('lens_kcube', '0')
        camera_bp.set_attribute('lens_x_size', '0')
        camera_bp.set_attribute('lens_y_size', '0')
        return camera_bp
     
    def save(self):
        Camera.save(self)

class HDL64E(Sensor):
    def __init__(self, vehicle, world, actor_list, lidar_name, lidar_id, folder_output, transform, lidar_range, upper_vfov, lower_vfov, hfov):
        self.queue = queue.PriorityQueue()
        # set lidar attributes
        self.bp = self.set_attributes(world.get_blueprint_library(), lidar_range, upper_vfov, lower_vfov, hfov)
        self.sensor = world.spawn_actor(self.bp, transform, attach_to=vehicle)
        actor_list.append(self.sensor)
        # 1st argument of queue.put identify the priority of the data, (low numbers have higher priority) 
        self.sensor.listen(lambda data: self.queue.put((data.timestamp, data)))
        self.sensor_id = lidar_id
        
        self.i_packet = 0
        self.i_frame = 0

        self.rotation_lidar = rotation_carla(transform.rotation)
        self.rotation_lidar_transpose = self.rotation_lidar.T
        self.initial_loc = np.array([transform.location.x, transform.location.y, transform.location.z])
        self.frame_output = folder_output+"/frames_"+lidar_name
        self.name = lidar_name
        
        os.makedirs(self.frame_output) if not os.path.exists(self.frame_output) else [os.remove(f) for f in glob.glob(self.frame_output+"/*") if os.path.isfile(f)]

        settings = world.get_settings()
        self.packet_per_frame = 1/(self.bp.get_attribute('rotation_frequency').as_float()*settings.fixed_delta_seconds)
        self.packet_period = settings.fixed_delta_seconds

        self.pt_size = 4*4 #(4 float32 with x, y, z, timestamp)
        self.list_pts = []
        self.list_semantic = []
        self.list_ts = []
        self.list_trajectory = []

        self.ts_tmp = 0.0
        print('created %s' % self.sensor)

    def define_constant_waymo_lidar_features(self, nbr_pts):
        # Constant Lidar features
        self.waymo_elongation = tf.constant(np.zeros((nbr_pts,1), dtype=float), dtype=tf.float32)
        self.waymo_inside_nlz = tf.constant(np.full((nbr_pts, 1), False), dtype=tf.bool)
        self.waymo_return_number = tf.constant(np.ones((nbr_pts,1), dtype=float), dtype=tf.int32)
        self.time_of_day = 'noon'
        # weather
        self.weather = 'sunny'

    def define_waymo_lidar_features(self, buffer, nbr_pts, map_name, timestamp, frame_num):
    
        self.define_constant_waymo_lidar_features(nbr_pts)
        # frame_name
        self.frame_name = str(timestamp) + "_location_" + map_name + "_" + str(frame_num)
        # location
        self.location = map_name
        # timestamp
        self.timestamp = timestamp
        # point_cloud positions
        self.waymo_positions = tf.transpose(tf.constant([buffer[:]['x'], -buffer[:]['y'], buffer[:]['z']], dtype=tf.float32))
        # point_cloud intensties
        self.waymo_intensity = tf.reshape(tf.constant(buffer[:]['intensity'], dtype=tf.float32), [nbr_pts, 1])
        # point_cloud extrinsic transformation matrix
        self.waymo_extrinsics_R = tf.constant(self.rotation_lidar_transpose, dtype=tf.float32)
        #self.waymo_extrinsics_T = np.array([1.22, 0, -1.78]) - (self.waymo_extrinsics_R.dot(self.initial_loc))
        self.waymo_extrinsics_T = tf.constant([1.22, 0, -1.78], dtype=tf.float32) + tf.constant(self.initial_loc, dtype=tf.float32)

    def save_lidar_features_in_TFRECORD(self):
        lidar_features = {
            # lidar id (e.g 1, 3).
            'id': tf.constant(self.sensor_id, dtype=tf.int64).numpy(),
            # lidar name (e.g. TOP, REAR).
            'name': tf.constant(self.name, dtype=tf.string).numpy(),
            # 3D pointcloud data from the lidar with N points.
            'pointcloud': {
                # Pointcloud positions (Nx3).
                'positions': self.waymo_positions.numpy(),
                # Pointcloud intensity (Nx1).
                'intensity': self.waymo_intensity.numpy(),
                # Pointcloud elongation (Nx1).
                'elongation': self.waymo_elongation.numpy(),
                # Pointcloud No Label Zone information (NLZ) (Nx1).
                'inside_nlz': self.waymo_inside_nlz.numpy(),
                # Pointcloud return number (first return, second return, or third)
                'return_number': self.waymo_return_number.numpy(),
            },
            # lidar extrinsics.
            'extrinsics': {
                'R': self.waymo_extrinsics_R.numpy(),
                't': self.waymo_extrinsics_T.numpy(),
            },
            # lidar pointcloud to camera image correspondence for N lidar points.
            'camera_projections': {
                # Camera id each 3D point projects to (Nx1).
                'ids': tf.constant(np.zeros((len(self.waymo_inside_nlz),1), dtype=int), dtype=tf.int64).numpy(),
                # Image location (x, y) of each 3D point's projection (Nx2).
                'positions': tf.constant(np.zeros((len(self.waymo_inside_nlz),2), dtype=float), dtype=tf.float32).numpy(),
            }
        }
        return lidar_features
    def init(self, client):
        #self.initial_loc = translation_carla(self.sensor.get_location())
        #self.initial_rot_transpose = (rotation_carla(self.sensor.get_transform().rotation).dot(self.rotation_lidar)).T
        # Waymo provides an extrinsic calibration matrix transforms the lidar frame to the vehicle frame
        # Inverse of the rotation matrix provided to the sensor frame will transform it to the vehicle frame
        #self.initial_rot_transpose = self.sensor.get_transform().get_inverse_matrix()
        # Enables the recording feature, which will start saving every information possible needed by the server to replay the simulation.
        client.start_recorder(os.path.dirname(os.path.realpath(__file__))+"/../"+self.frame_output+"/recording.log")


    def save(self, map_name):
        while not self.queue.empty():
            data = self.queue.get()[1]
            ts = data.timestamp-Sensor.initial_ts
            if ts-self.ts_tmp > self.packet_period*1.5 or ts-self.ts_tmp < 0:
                print("[Error in timestamp] HDL64E: previous_ts %f -> ts %f" %(self.ts_tmp, ts))
                sys.exit()

            self.ts_tmp = ts

            nbr_pts = len(data.raw_data)//16 #4 float32
            self.list_ts.append(np.broadcast_to(ts, nbr_pts))

            #### numpy.frombuffer(buffer, dtype=float, count=- 1, offset=0, *, like=None)
            ####    Interpret a buffer as a 1-dimensional array.
            ####    dtype: data-type, optional
            ####        Data-type of the returned array; default: float.
            ####    count: int, optional
            ####        Number of items to read. -1 means all data in the buffer.
            ####    offset: int, optional
            ####        Start reading the buffer from this offset (in bytes); default: 0.
            #### Examples
            #### >>> s = b'hello world'
            #### >>> np.frombuffer(s, dtype='S1', count=5, offset=6)
            #### array([b'w', b'o', b'r', b'l', b'd'], dtype='|S1')
            #### >>> np.frombuffer(b'\x01\x02\x03\x04\x05', dtype=np.uint8, count=3)
            #### array([1, 2, 3], dtype=uint8)
            ## for semantic LiDAR
            #buffer = np.frombuffer(data.raw_data, dtype=np.dtype([('x','f4'),('y','f4'),('z','f4'),('cos','f4'),('index','u4'),('semantic','u4')]))
            buffer = np.frombuffer(data.raw_data, dtype=np.dtype([('x','f4'),('y','f4'),('z','f4'),('intensity','f4')]))
            
            # * We're negating the y to correctly visualize a world that matches what we see in Unreal since we uses a right-handed coordinate system
            # * XYZ position is computed in the sensor reference frame
            #self.list_pts.append(np.array([buffer[:]['x'], -buffer[:]['y'], buffer[:]['z'], buffer[:]['cos']]))
            #self.list_semantic.append(np.array([buffer[:]['index'], buffer[:]['semantic']]))
            self.list_pts.append(np.array([buffer[:]['x'], -buffer[:]['y'], buffer[:]['z'], buffer[:]['intensity']]))
    
            self.define_waymo_lidar_features(buffer, nbr_pts, map_name, data.timestamp, self.i_frame)
            self.i_frame += 1
            #self.i_packet += 1
            #if self.i_packet%self.packet_per_frame == 0:
            #    pts_all = np.hstack(self.list_pts)
            #    pts_all[0:3,:] = self.rotation_lidar_transpose.dot(pts_all[0:3,:])
            #    pts_all = pts_all.T
            #    #semantic_all = np.hstack(self.list_semantic).T
            #    ts_all = np.concatenate(self.list_ts)
            #    self.list_pts = []
            #    self.list_semantic = []
            #    self.list_ts = []

        lidar_spec_features = self.save_lidar_features_in_TFRECORD()
        return self.i_frame, lidar_spec_features, data.timestamp        

    def set_attributes(self, blueprint_library, lidar_range, upper_vfov, lower_vfov, hfov):
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', lidar_range)   # in meters
        # 177K points per frame
        # 10 frame per second (10 Hz lidar)
        # points_per_second = 177K points per frame * 10 frames per second = 1770K  points/sec
        # Number of points per second_per_Lidar = 1770K/5 = 354K
        lidar_bp.set_attribute('points_per_second', "354000")
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('upper_fov', upper_vfov)
        lidar_bp.set_attribute('lower_fov', lower_vfov)
        lidar_bp.set_attribute('horizontal_fov', hfov) #default
        lidar_bp.set_attribute('atmosphere_attenuation_rate', '0.004') # default
        lidar_bp.set_attribute('dropoff_general_rate', '0.0')  # no random dropoff
        lidar_bp.set_attribute('dropoff_intensity_limit', '0.3') # For intensity dropoff (2 strongest returns are saved, so will discard low intensity values)
        lidar_bp.set_attribute('dropoff_zero_intensity', '1.0') # For intensity dropoff, the probability of each point with zero intensity being dropped.
        
        return lidar_bp

# Function to change rotations in CARLA from left-handed to right-handed reference frame
def rotation_carla(rotation):
    cr = math.cos(math.radians(rotation.roll))
    sr = math.sin(math.radians(rotation.roll))
    cp = math.cos(math.radians(rotation.pitch))
    sp = math.sin(math.radians(rotation.pitch))
    cy = math.cos(math.radians(rotation.yaw))
    sy = math.sin(math.radians(rotation.yaw))
    return np.array([[cy*cp, -cy*sp*sr+sy*cr, -cy*sp*cr-sy*sr],[-sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],[sp, cp*sr, cp*cr]])

# Function to change translations in CARLA from left-handed to right-handed reference frame
def translation_carla(location):
    #if isinstance(location, np.ndarray):
    #    return location*(np.array([[1],[-1],[1]]))
    #else:
    #    return np.array([location.x, -location.y, location.z])
    if isinstance(location, np.ndarray):
        return location*(np.array([[1],[1],[1]]))
    else:
        return np.array([location.x, location.y, location.z])

def screenshot(vehicle, world, actor_list, folder_output, transform, i):
    sensor = world.spawn_actor(RGB.set_attributes(RGB, world.get_blueprint_library()), transform, attach_to=vehicle)
    actor_list.append(sensor)
    screenshot_queue = queue.Queue()
    sensor.listen(screenshot_queue.put)
    print('created %s' % sensor)

    while screenshot_queue.empty(): world.tick()

    file_path = folder_output+"/screenshot"+str(i)+".png"
    screenshot_queue.get().save_to_disk(file_path)
    print("Export : "+file_path)
    actor_list[-1].destroy()
    print('destroyed %s' % actor_list[-1])
    del actor_list[-1]

##### Adjust world and traffic manager parameters #####
## 1- Set Synchronous mode between client and server in world (i.e server will wait for a client tick to proceed to the next step)
## 2- Set the time step size
## 3- Set the traffic manager mode to synchronous mode
## 4- Set other traffic manager properties
def init_world_settings_and_traffic_manager(client, fps_simu):

        # * When synchronous mode is set to true, the server will wait for a client tick in order to move forward.
        # * If the server is set to synchronous mode, the Traffic Manager must be set to synchronous mode too in the same client that does the tick.
        # for more information, refer to https://carla.readthedocs.io/en/0.9.13/adv_traffic_manager/#synchronous-mode
        world = client.get_world()
        settings = world.get_settings()

        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            # Set fixed time between each two consecutive steps in the simulation
            #TODO: Check Physics determinism (physics substepping)
            settings.fixed_delta_seconds = 1.0/fps_simu
        else:
            synchronous_master = False
   
        # False by default
        settings.no_rendering_mode = False 
        world.apply_settings(settings)
        
        traffic_manager = client.get_trafficmanager()
   
        # Settings below are set to make the vehicles drive 5% slower than the current speed limit, leaving at least 1 meters between themselves and other vehicles.
        # refer to https://carla.readthedocs.io/en/0.9.13/adv_traffic_manager/#vehicle-behavior-considerations
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        #traffic_manager.set_hybrid_physics_mode(True)
        #traffic_manager.set_random_device_seed(args.seed) # set for deterministic simulation
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.global_percentage_speed_difference(5.0)
        
        '''
        # TM can configures dangerous behavior for a specific vehicle. So it will:
        #   * Ignore all traffic lights
        #   * Leave no safety distance from other vehicles
        #   * Drive 20% faster than the current speed limit
        danger_car = my_vehicles[0]
        traffic_manager.ignore_lights_percentage(danger_car,100)
        traffic_manager.distance_to_leading_vehicle(danger_car,0)
        traffic_manager.vehicle_percentage_speed_difference(danger_car,-20)
        '''
        '''
        # TODO: Check traffic lights (check line 485 first)
        # By default, vehicle lights (brake, indicators, etc...) managed by the TM are never updated. It is possible to delegate the TM to update the vehicle lights automatically
        for actor in my_vehicles:
            traffic_manager.update_vehicle_lights(actor, True)
        '''
        return synchronous_master

def spawn_npc(client, synchronous_master, nbr_vehicles, nbr_walkers, vehicles_list, all_walkers_controllers_id, walkers_list):
        world = client.get_world()
        traffic_manager = client.get_trafficmanager()

        blueprints = world.get_blueprint_library().filter('vehicle.*')
        blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

        safe = True
        if safe:
                blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
                blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
                blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
                blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
                blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        print("Number of spawn points : ", number_of_spawn_points)

        if nbr_vehicles <= number_of_spawn_points:
                random.shuffle(spawn_points)
        elif nbr_vehicles > number_of_spawn_points:
                msg = 'requested %d vehicles, but could only find %d spawn points'
                logging.warning(msg, nbr_vehicles, number_of_spawn_points)
                nbr_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
                if n >= nbr_vehicles:
                        break
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                        color = random.choice(blueprint.get_attribute('color').recommended_values)
                        blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                        driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                        blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')

                # prepare the light state of the cars to spawn
                """
                Lights are off by default in any situation and should be managed by the user via script. 
                * NONE
                    All lights off.
                * Position
                * LowBeam
                * HighBeam
                * Brake
                * RightBlinker
                * LeftBlinker
                * Reverse
                * Fog
                * Interior
                * Special1
                    This is reserved for certain vehicles that can have special lights, like a siren.
                * Special2
                    This is reserved for certain vehicles that can have special lights, like a siren.
                * All
                    All lights on. 
                """
                light_state = vls.NONE
                car_lights_on = False
                if car_lights_on:
                        light_state = vls.Position | vls.LowBeam | vls.LowBeam

                # spawn the cars and set their autopilot and light state all together
                batch.append(SpawnActor(blueprint, transform)
                        .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                        .then(SetVehicleLightState(FutureActor, light_state)))

        for response in client.apply_batch_sync(batch, synchronous_master):
                if response.error:
                        logging.error(response.error)
                else:
                        vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        # The valid values are between 0 and 1
        percentagePedestriansRunning = 0.3            # how many pedestrians will run
        percentagePedestriansCrossing = 0.4         # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        all_loc = []
        i = 0
        while i < nbr_walkers:
                spawn_point = carla.Transform()
                loc = world.get_random_location_from_navigation()
                if ((loc != None) and not(loc in all_loc)):
                        spawn_point.location = loc
                        spawn_points.append(spawn_point)
                        all_loc.append(loc)
                        i = i + 1
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
                walker_bp = random.choice(blueprintsWalkers)
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                        walker_bp.set_attribute('is_invincible', 'false')
                # set the max speed, CARLA provides a library of blueprints for actors. Each of these blueprints has a series of attributes defined internally. Some of these are modifiable, others are not. A list of recommended values is provided for those that can be set. 
                if walker_bp.has_attribute('speed'):
                        if (random.random() > percentagePedestriansRunning):
                                # walking
                                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                        else:
                                # running
                                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                        print("Walker has no speed")
                        walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_point))
        # Executes a list of commands on a single simulation step, blocks until the commands are linked, and returns a list of command.Response that can be used to determine whether a single command succeeded or not.
        results = client.apply_batch_sync(batch, True)
        # Get ID and Speed of spawned walkers list
        walker_speed2 = []
        for i in range(len(results)):
                if results[i].error:
                        logging.error(results[i].error)
                else:
                        walkers_list.append({"id": results[i].actor_id})
                        walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
                if results[i].error:
                        logging.error(results[i].error)
                else:
                        walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
                all_walkers_controllers_id.append(walkers_list[i]["con"])
                all_walkers_controllers_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_walkers_controllers_id)

        # Send a tick to ensure server receives the last transform of the walkers we have just created
        world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road. Value should be between 0.0 and 1.0
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_walkers_controllers_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(world.get_random_location_from_navigation())
                # max speed
                all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('Spawned %d vehicles and %d walkers' % (len(vehicles_list), len(walkers_list)))

def getActorsBoundingBoxes(world, walkers_list, egoVehicle):

    walkers_bb_list = world.get_actors([x["id"] for x in walkers_list])

    signs_bbs_list = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
    # extend the sign list with traffic lights
    #signs_list.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficLight))

    # Get all vehicles in the map and filter the ego vehicle
    all_vechiles_bikes_bbs_list = world.get_actors().filter('*vehicle*')
    vechiles_bikes_bbs_list = [x for x in all_vechiles_bikes_bbs_list if x.id != egoVehicle.id]

    return vechiles_bikes_bbs_list, signs_bbs_list, walkers_bb_list
    
def follow(transform, world):    # Transforme carla.Location(x,y,z) from sensor to world frame
    rot = transform.rotation
    rot.pitch = -25 
    world.get_spectator().set_transform(carla.Transform(transform.transform(carla.Location(x=-15,y=0,z=5)), rot))

def get_object_pose(egoVehicle, npc):
    #rot_global_to_ego =  rotation_carla(egoVehicle.get_transform().rotation)
    inverse_rot_global_to_ego = rotation_carla(egoVehicle.get_transform().rotation).T
    trans_global_to_ego = translation_carla(egoVehicle.get_transform().location)
    #inverse_rot_global_to_object = rotation_carla(npc.get_transform().rotation).T
    rot_global_to_object = rotation_carla(npc.get_transform().rotation)
    trans_global_to_object = translation_carla(npc.get_transform().location)
    # the change from the waymo reference to the carla reference is removed since all translations should be measured taking the same point of reference in all vehicles
    # so applying it, should apply a translation to all vehicles.
    #trans_ego_to_waymo_reference = np.array([1.22, 0, -1.78])
    
    #feature_objects_pose_T = trans_global_to_ego + trans_ego_to_waymo_reference - rot_global_to_ego.dot(inverse_rot_global_to_object.dot(trans_global_to_object))
    #feature_objects_pose_R = rot_global_to_ego.dot(inverse_rot_global_to_object)
    #object_pose_T = trans_global_to_object - rot_global_to_object.dot(inverse_rot_global_to_ego.dot(trans_ego_to_waymo_reference)) - \
    #                rot_global_to_object.dot(inverse_rot_global_to_ego.dot(trans_global_to_ego))
    #object_pose_R = rot_global_to_object.dot(inverse_rot_global_to_ego)
   
    object_pose_T = rot_global_to_object.dot(trans_global_to_ego) - rot_global_to_object.dot(trans_global_to_object) #- rot_global_to_object.dot(inverse_rot_global_to_ego.dot(trans_ego_to_waymo_reference))
    object_pose_R = rot_global_to_object.dot(inverse_rot_global_to_ego)

    return object_pose_R, object_pose_T

def get_signs_pose(egoVehicle, bbox):
    #rot_global_to_ego =  rotation_carla(egoVehicle.get_transform().rotation)
    inverse_rot_global_to_ego = rotation_carla(egoVehicle.get_transform().rotation).T
    trans_global_to_ego = translation_carla(egoVehicle.get_transform().location)
    #inverse_rot_global_to_object = rotation_carla(bbox.get_transform().rotation).T
    rot_global_to_object = rotation_carla(bbox.rotation)
    trans_global_to_object = translation_carla(bbox.location)
    trans_ego_to_waymo_reference = np.array([1.22, 0, -1.78])
    
    #feature_objects_pose_T = trans_global_to_ego + trans_ego_to_waymo_reference - rot_global_to_ego.dot(inverse_rot_global_to_object.dot(trans_global_to_object))
    #feature_objects_pose_R = rot_global_to_ego.dot(inverse_rot_global_to_object)
    #object_pose_T = trans_global_to_object - rot_global_to_object.dot(inverse_rot_global_to_ego.dot(trans_ego_to_waymo_reference)) - \
    #                rot_global_to_object.dot(inverse_rot_global_to_ego.dot(trans_global_to_ego))
    #object_pose_R = rot_global_to_object.dot(inverse_rot_global_to_ego)
    object_pose_T = rot_global_to_object.dot(trans_global_to_ego) - rot_global_to_object.dot(trans_global_to_object) #- rot_global_to_object.dot(inverse_rot_global_to_ego.dot(trans_ego_to_waymo_reference))
    object_pose_R = rot_global_to_object.dot(inverse_rot_global_to_ego)

    return object_pose_R, object_pose_T

def get_object_category(npc):
    if len(npc.get_physics_control().wheels) == 4: # A Car
        category = 1
        text = 'vehicle'
    elif len(npc.get_physics_control().wheels) == 2: # A bike
        category = 4
        text = 'cyclist'
    else:
        print("Error: Found a vehicle object that isn't a bike or a car")
    return category, text

def save_objects_labels(vechiles_bikes_bbs_list, walkers_bb_list, signs_bbs_list, egoVehicle):
    feature_objects_pose_R = []
    feature_objects_pose_T = []
    feature_bounding_box_shape = []
    feature_object_category_label= []
    feature_object_category_text = []

    # Save vehicles and bicycles then walkers then Traffic Signs
    for npc in vechiles_bikes_bbs_list:
            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(egoVehicle.get_transform().location)

            # Filter for the vehicles within 75m (Maximum LiDAR range)
            if dist <= 75:
                # Include current object in Waymo dataset
                #1- Get Object position and orientation with respect to the waymo vehicle
                objects_pose_R, objects_pose_T = get_object_pose(egoVehicle, npc)
                feature_objects_pose_R.append(objects_pose_R)
                feature_objects_pose_T.append(objects_pose_T)

                #2- Get object bounding box dimension
                feature_bounding_box_shape.append(np.array([2* bb.extent.x, 2* bb.extent.y, 2* bb.extent.z]))
                
                #3- Get object type
                category, text = get_object_category(npc)
                feature_object_category_label.append(category)
                feature_object_category_text.append(text)     
                
    for npc in walkers_bb_list:
            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(egoVehicle.get_transform().location)

            # Filter for the vehicles within 75m (Maximum LiDAR range)
            if dist <= 75:
                # Include current object in Waymo dataset
                #1- Get Object position and orientation with respect to the waymo vehicle
                objects_pose_R, objects_pose_T = get_object_pose(egoVehicle, npc)
                feature_objects_pose_R.append(objects_pose_R)
                feature_objects_pose_T.append(objects_pose_T)
                
                #2- Get object bounding box dimension
                feature_bounding_box_shape.append(np.array([2* bb.extent.x, 2* bb.extent.y, 2* bb.extent.z]))
                
                #3- Get object type
                feature_object_category_label.append(2)
                feature_object_category_text.append('pedestrian')     

    for bbox in signs_bbs_list:
            dist = bbox.location.distance(egoVehicle.get_transform().location)

            # Filter for the vehicles within 75m (Maximum LiDAR range)
            if dist <= 75:
                # Include current object in Waymo dataset
                #1- Get Object position and orientation with respect to the waymo vehicle
                objects_pose_R, objects_pose_T = get_signs_pose(egoVehicle, bbox)
                feature_objects_pose_R.append(objects_pose_R)
                feature_objects_pose_T.append(objects_pose_T)
                
                #2- Get object bounding box dimension
                feature_bounding_box_shape.append(np.array([2* bbox.extent.x, 2* bbox.extent.y, 2* bbox.extent.z]))
                
                #3- Get object type
                feature_object_category_label.append(3)
                feature_object_category_text.append('sign')
    
    object_features = create_label_in_tfrecord_format(len(feature_object_category_label), feature_objects_pose_R, feature_objects_pose_T, feature_bounding_box_shape, feature_object_category_label, feature_object_category_text)

    return object_features

def create_label_in_tfrecord_format(num_objects, feature_objects_pose_R, feature_objects_pose_T, feature_bounding_box_shape, feature_object_category_label, feature_object_category_text):
    
    object_names=[]
    indices=[]
    i = 0
    characters = string.ascii_letters + string.digits
    while i < num_objects :
        rand_name = ''.join(random.choice(characters) for i in range(22))
        object_names.append(rand_name)
        indices.append(i)
        i = i + 1
    
    objects_label = {
        # object id
        'id': tf.constant(indices, dtype=tf.int64).numpy(),
        # object name
        'name': tf.constant(object_names, dtype=tf.string).numpy(),
        # object category (class).
        'category': {
            # integer label id
            'label': tf.constant(feature_object_category_label, dtype=tf.int64).numpy(),
            # text label id
            'text': tf.constant(feature_object_category_text, dtype=tf.string).numpy(),
        },
        # object shape
        'shape': {
            # object size (length, width, height) along object's (x, y, z) axes.
            'dimension': tf.constant(feature_bounding_box_shape, dtype=tf.float32).numpy(),
        },
        # object pose
        'pose': {
            'R': tf.constant(feature_objects_pose_R, dtype=tf.float32).numpy(),
            't': tf.constant(feature_objects_pose_T, dtype=tf.float32).numpy()
        },
        # object difficulty level
        # The higher the level, the harder it is.
        'difficulty_level': {
            'detection': tf.constant(np.zeros((num_objects,), dtype=int), dtype=tf.int64).numpy(),
            'tracking': tf.constant(np.zeros((num_objects,), dtype=int), dtype=tf.int64).numpy(),
        },
    }
    return objects_label

def Get_Serialized_Frame_Example(map_name, frame_current, timestamp, front_lidar_features, rear_lidar_features, sl_lidar_features, sr_lidar_features, top_lidar_features, object_features):
    # Composite FeatureConnetor for camera data.
    CAMERA_FEATURE_SPEC = tfds.features.FeaturesDict({
        # camera id (e.g 0, 3).
        'id': tf.int64,
        # camera name (e.g. FRONT, LEFT).
        'name': tfds.features.Text(),
        # image data.
        'image': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'),
        # camera instrinsics.
        'intrinsics': {
            # Camera intrinsics matrix K (3x3 matrix).
            'K': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
            # Distortion coeffiecients (k1, k2, p1, p2, k3).
            'distortion': tfds.features.Tensor(shape=(5,), dtype=tf.float32),
        },
        # camera extrinsics.
        'extrinsics': {
            'R': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
            't': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
        },
        # camera shutter type
        'shutter_type': tfds.features.Text(),
    })


    # Composite FeatureConnetor for Lidar data.
    LIDAR_FEATURE_SPEC = tfds.features.FeaturesDict({
        # lidar id (e.g 1, 3).
        'id': tf.int64,
        # lidar name (e.g. TOP, REAR).
        'name': tfds.features.Text(),
        # 3D pointcloud data from the lidar with N points.
        'pointcloud': {
            # Pointcloud positions (Nx3).
            'positions': tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
            # Pointcloud intensity (Nx1).
            'intensity': tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
            # Pointcloud elongation (Nx1).
            'elongation': tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
            # Pointcloud No Label Zone information (NLZ) (Nx1).
            'inside_nlz': tfds.features.Tensor(shape=(None, 1), dtype=tf.bool),
            # Pointcloud return number (first return, second return, or third)
            'return_number': tfds.features.Tensor(shape=(None, 1), dtype=tf.int32),
        },
        # lidar extrinsics.
        'extrinsics': {
            'R': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
            't': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
        },
        # lidar pointcloud to camera image correspondence for N lidar points.
        'camera_projections': {
            # Camera id each 3D point projects to (Nx1).
            'ids': tfds.features.Tensor(shape=(None, 1), dtype=tf.int64),
            # Image location (x, y) of each 3D point's projection (Nx2).
            'positions': tfds.features.Tensor(shape=(None, 2), dtype=tf.float32),
        }
    })


    # Object category labels.
    OBJECT_CATEGORY_LABELS = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']


    # Composite FeatureConnetor for object data.
    OBJECT_FEATURE_SPEC = tfds.features.FeaturesDict({
        # object id
        'id': tf.int64,
        # object name
        'name': tfds.features.Text(),
        # object category (class).
        'category': {
            # integer label id
            'label': tfds.features.ClassLabel(names=OBJECT_CATEGORY_LABELS),
            # text label id
            'text': tfds.features.Text(),
        },
        # object shape
        'shape': {
            # object size (length, width, height) along object's (x, y, z) axes.
            'dimension': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
        },
        # object pose
        'pose': {
            'R': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
            't': tfds.features.Tensor(shape=(3,), dtype=tf.float32)
        },
        # object difficulty level
        # The higher the level, the harder it is.
        'difficulty_level': {
            'detection': tf.int64,
            'tracking': tf.int64,
        },
    })


    # Feature specification of frame dataset.
    FRAME_FEATURE_SPEC = tfds.features.FeaturesDict({
        # A unique name that identifies the sequence the frame is from.
        'scene_name': tfds.features.Text(),
        # A unique name that identifies this particular frame.
        'frame_name': tfds.features.Text(),
        # Frame *start* time (set to the timestamp of the first top laser spin).
        'timestamp': tf.int64,
        # Day, Dawn/Dusk, or Night, determined from sun elevation.
        'time_of_day': tfds.features.Text(),
        # Human readable location (e.g. CHD, SF) of the run segment.
        'location': tfds.features.Text(),
        # Sunny or Rain.
        'weather': tfds.features.Text(),
        # Camera sensor data.
        'cameras': {
            'front': CAMERA_FEATURE_SPEC,
            'front_left': CAMERA_FEATURE_SPEC,
            'front_right': CAMERA_FEATURE_SPEC,
            'side_left': CAMERA_FEATURE_SPEC,
            'side_right': CAMERA_FEATURE_SPEC,
        },
        # lidar sensor data.
        'lidars': {
            'top': LIDAR_FEATURE_SPEC,
            'front': LIDAR_FEATURE_SPEC,
            'side_left': LIDAR_FEATURE_SPEC,
            'side_right': LIDAR_FEATURE_SPEC,
            'rear': LIDAR_FEATURE_SPEC,
        },
        # objects annotations data.
        'objects': tfds.features.Sequence(OBJECT_FEATURE_SPEC)
    })

    empty_camera_spec = {
        # camera id (e.g 0, 3).
        'id': tf.constant(1, dtype=tf.int64).numpy(),
        # camera name (e.g. FRONT, LEFT).
        'name': tf.constant('front', dtype=tf.string).numpy(),
        # image data.
        'image': tf.constant(np.zeros((1280, 1920, 3), dtype=np.uint8), dtype=tf.uint8).numpy(),
        # camera instrinsics.
        'intrinsics': {
            # Camera intrinsics matrix K (3x3 matrix).
            'K': tf.constant(np.zeros((3, 3), dtype=np.float32), dtype=tf.float32).numpy(),
            # Distortion coeffiecients (k1, k2, p1, p2, k3).
            'distortion': tf.constant([4.7340468e-02, -3.3457774e-01,  1.6343805e-03, -1.3573887e-04, 0], dtype=tf.float32).numpy(),
        },
        # camera extrinsics.
        'extrinsics': {
            'R': tf.constant(np.identity(3, dtype=np.float32), dtype=tf.float32).numpy(),
            't': tf.constant(np.zeros((3,), dtype=np.float32), dtype=tf.float32).numpy(),
        },
        # camera shutter type
        'shutter_type': tf.constant('right_to_left', dtype=tf.string).numpy(),
    }
    
    front_left_camera = empty_camera_spec
    front_left_camera['id'] = tf.constant(2, dtype=tf.int64).numpy()
    front_left_camera['name'] = tf.constant('front_left', dtype=tf.string).numpy()
    
    front_right_camera = empty_camera_spec
    front_right_camera['id'] = tf.constant(3, dtype=tf.int64).numpy()
    front_right_camera['name'] = tf.constant('front_right', dtype=tf.string).numpy()
    
    side_left_camera = empty_camera_spec
    side_left_camera['id'] = tf.constant(4, dtype=tf.int64).numpy()
    side_left_camera['name'] = tf.constant('side_left', dtype=tf.string).numpy()
    side_left_camera['image'] = tf.constant(np.zeros((886, 1920, 3), dtype=np.uint8), dtype=tf.uint8).numpy()
    
    side_right_camera = empty_camera_spec
    side_right_camera['id'] = tf.constant(5, dtype=tf.int64).numpy()
    side_right_camera['name'] = tf.constant('side_right', dtype=tf.string).numpy()
    side_right_camera['image'] = tf.constant(np.zeros((886, 1920, 3), dtype=np.uint8), dtype=tf.uint8).numpy()

    frame_spec = {
        # A unique name that identifies the sequence the frame is from.
        'scene_name': tf.constant(map_name, dtype=tf.string).numpy(),
        # A unique name that identifies this particular frame.
        'frame_name': tf.constant(map_name + '_' + str(frame_current) + "_" + str(timestamp), dtype=tf.string).numpy(),
        # Frame *start* time (set to the timestamp of the first top laser spin).
        'timestamp': tf.constant(round(timestamp), dtype=tf.int64).numpy(),
        # Day, Dawn/Dusk, or Night, determined from sun elevation.
        'time_of_day': tf.constant("Dawn/Dusk", dtype=tf.string).numpy(),
        # Human readable location (e.g. CHD, SF) of the run segment.
        'location': tf.constant("location_phx", dtype=tf.string).numpy(),
        # Sunny or Rain.
        'weather': tf.constant("sunny", dtype=tf.string).numpy(),
        # Camera sensor data.
        'cameras': {
            'front': empty_camera_spec,
            'front_left': front_left_camera,
            'front_right': front_right_camera,
            'side_left': side_left_camera,
            'side_right': side_right_camera,
        },
        # lidar sensor data.
        'lidars': {
            'top': top_lidar_features,
            'front': front_lidar_features,
            'side_left': sl_lidar_features,
            'side_right': sr_lidar_features,
            'rear': rear_lidar_features,
        },
        # objects annotations data.
        'objects': object_features
    }

    frame_example_bytes = FRAME_FEATURE_SPEC.serialize_example(frame_spec)
    return frame_example_bytes


def write_synthetic_waymo_tfrecord(filepath, syn_waymo_dataset):
    with tf.io.TFRecordWriter(path=filepath) as writer:
        for example_bytes in syn_waymo_dataset:
            writer.write(example_bytes)