import glob
import os
import sys
from pathlib import Path
 
try:
    sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        "C:/Users/moghando/Downloads/CARLA_0.9.13/WindowsNoEditor" if os.name == 'nt' else str(Path.home()) + "/CARLA_0.9.13",
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    print(sys.path)
except IndexError:
    pass

import carla
import time
from modules import generator_WAYMO as gen

def main():
    start_record_full = time.time()

    fps_simu = 10.0 # Recommended value from carla is 10Hz https://carla.readthedocs.io/en/0.9.13/ref_sensors/#lidar-raycast-sensor
    time_stop = 2.0
    nbr_frame = 200 # Sensor Frequency = 10Hz, Waymo dataset spans 20 secs for each scene. #MAX = 10000
    nbr_walkers = 120
    nbr_vehicles = 500

    actor_list = []
    vehicles_list = []
    all_walkers_id = []
    
    carla_preset_weather = [ 
    carla.WeatherParameters.ClearNoon, 
    carla.WeatherParameters.CloudyNoon, 
    carla.WeatherParameters.WetNoon, 
    carla.WeatherParameters.WetCloudyNoon, 
    carla.WeatherParameters.SoftRainNoon, 
    carla.WeatherParameters.MidRainyNoon, 
    carla.WeatherParameters.HardRainNoon, 
    carla.WeatherParameters.ClearSunset, 
    carla.WeatherParameters.CloudySunset, 
    carla.WeatherParameters.WetSunset, 
    carla.WeatherParameters.WetCloudySunset, 
    carla.WeatherParameters.SoftRainSunset, 
    carla.WeatherParameters.MidRainSunset,
    carla.WeatherParameters.HardRainSunset]
    
    spawn_points = [23,46,0,125,53,257,62]
    
    init_settings = None

    try:
        # Connect to CARLA Simulator Server
        client = carla.Client('localhost', 2000)
        init_settings = carla.WorldSettings()
        
        for i_map in [0, 1, 2, 3, 4, 5, 6]: #7 maps from Town01 to Town07
            
            walkers_list = []
            # Sets the maximum time a network call is allowed before blocking it and raising a timeout error
            client.set_timeout(150.0)
            
            # Creates a new world with default settings using map_name map (Town01 to Town07)
            map_name = "Town0"+str(i_map+1)
            world = client.load_world(map_name)
            print("Map Town0"+str(i_map+1))
            # wait few seconds for loading the map
            time.sleep(5.0)
            
            # Create directory for the generated Waymo Open Dataset and remove old files
            folder_output = "WAYMO_DATASET_CARLA_v%s/%s/generated/" %(client.get_client_version(), world.get_map().name)
            os.makedirs(folder_output) if not os.path.exists(folder_output) else [os.remove(f) for f in glob.glob(folder_output+"/*") if os.path.isfile(f)]
                        
            # Weather (TODO change to either noon or night)
            world.set_weather(carla.WeatherParameters.WetCloudyNoon)
            

            # Create WAYMO vehicle and spawn it in one of the recommended locations returned from get_spawn_points
            blueprint_library = world.get_blueprint_library()
            bp_WAYMO = blueprint_library.find('vehicle.tesla.model3')
            bp_WAYMO.set_attribute('color', '228, 239, 241')
            bp_WAYMO.set_attribute('role_name', 'WAYMO')
            #get_spawn_points Returns a list of recommendations made by the creators of the map to be used as spawning points for the vehicles. 
            #choose from the returned list the location and orientation indexed by (spawn_points list and i_map)
            start_pose = world.get_map().get_spawn_points()[spawn_points[i_map]]
            WAYMO = world.spawn_actor(bp_WAYMO, start_pose)
            # get_waypoint: Returns a list of pairs of waypoints. Every tuple on the list contains first an initial and then a final waypoint within the intersection boundaries that describe the beginning and the end of said lane along the junction.
            #waypoint = world.get_map().get_waypoint(start_pose.location)
            actor_list.append(WAYMO)
            print('Created %s' % WAYMO)

            ##### Adjust world and traffic manager parameters #####
            # 1- Set Synchronous mode between client and server in world (i.e server will wait for a client tick to proceed to the next step)
            # 2- Set the time step size
            # 3- Set the traffic manager mode to synchronous mode
            # 4- Set other traffic manager properties
            synchronous_master = gen.init_world_settings_and_traffic_manager(client, fps_simu)

            # Spawn vehicles and walkers (nothing needs to be changed here)
            gen.spawn_npc(client, synchronous_master, nbr_vehicles, nbr_walkers, vehicles_list, all_walkers_id, walkers_list)
            
            # Get list for each actor type bounding boxes
            vechiles_bikes_bbs_list, signs_bbs_list, walkers_bbs_list = gen.getActorsBoundingBoxes(world, walkers_list, WAYMO)
            
            # Wait for WAYMO to stop
            start = world.get_snapshot().timestamp.elapsed_seconds
            # tick(): This method is used in synchronous mode, when the server waits for a client tick before computing the next frame. This method will send the tick, and give way to the server
            print("Waiting for WAYMO vehicle to stop ...")
            while world.get_snapshot().timestamp.elapsed_seconds-start < time_stop: world.tick()
            print("WAYMO vehicle stopped")

            # Set sensors transformation from WAYMO
            # CARLA uses left handed coordinate system, so every coordinate is in the same direction as waymo except for y-axis is in the opposite direction
            # so, will negate all y values when reading data from sensor and won't apply any rotation here unless needed
            # N.B.
            #   * The vehicle origin is on the ground, at the geometric center of the vehicle. So, all sensors have z-component to raise them from the ground
            #   * The tranforms below are done relative to the parent actors (i.e. the distances in the axes are relative locations not absolute)
            FrontLidar_transform     = carla.Transform(carla.Location(x=2.52, y=0, z=0.7) , carla.Rotation(pitch=0, yaw=0, roll=0))
            # rotate by 180 deg to make x pointing backward for rear sensor
            RearLidar_transform      = carla.Transform(carla.Location(x=-2.44, y=0, z=0.7), carla.Rotation(pitch=0, yaw=180, roll=0)) 
            SideLeftLidar_transform  = carla.Transform(carla.Location(x=0, y=-1.1, z=0.7), carla.Rotation(pitch=0, yaw=270, roll=0))
            SideRightLidar_transform = carla.Transform(carla.Location(x=0, y=1.1, z=0.7) , carla.Rotation(pitch=0, yaw=90, roll=0))
            TopLidar_transform       = carla.Transform(carla.Location(x=0, y=0, z=1.8), carla.Rotation(pitch=0, yaw=0, roll=0))

            # Take a screenshot for each LiDAR sensor view
            gen.screenshot(WAYMO, world, actor_list, folder_output, carla.Transform(carla.Location(x=2.52, y=0, z=0.7), carla.Rotation(pitch=0, yaw=0, roll=0)), 0)
            gen.screenshot(WAYMO, world, actor_list, folder_output, carla.Transform(carla.Location(x=-2.44, y=0, z=0.7), carla.Rotation(pitch=0, yaw=180, roll=0)), 1)
            gen.screenshot(WAYMO, world, actor_list, folder_output, carla.Transform(carla.Location(x=0, y=-1.1, z=0.7), carla.Rotation(pitch=0, yaw=270, roll=0)), 2)
            gen.screenshot(WAYMO, world, actor_list, folder_output, carla.Transform(carla.Location(x=0, y=1.1, z=0.7), carla.Rotation(pitch=0, yaw=90, roll=0)), 3)
            gen.screenshot(WAYMO, world, actor_list, folder_output, carla.Transform(carla.Location(x=0, y=0, z=1.8), carla.Rotation(pitch=0, yaw=0, roll=0)), 4)
            gen.screenshot(WAYMO, world, actor_list, folder_output, carla.Transform(carla.Location(x=-1.22, y=0, z=1.8), carla.Rotation(pitch=0, yaw=0, roll=0)), '_Waymo_Vehicle_Origin')

            # Create our sensors
            TopVelodyneHDL64 =       gen.HDL64E(WAYMO, world, actor_list, 'top', 1, folder_output, TopLidar_transform      , '75', '2.4', '-17.6', '360')
            FrontVelodyneHDL64 =     gen.HDL64E(WAYMO, world, actor_list, 'front', 2, folder_output, FrontLidar_transform    , '20', '30', '-90', '180')
            SideLeftVelodyneHDL64 =  gen.HDL64E(WAYMO, world, actor_list, 'side_left', 3, folder_output, SideLeftLidar_transform , '20', '30', '-90', '180')
            SideRightVelodyneHDL64 = gen.HDL64E(WAYMO, world, actor_list, 'side_right', 4, folder_output, SideRightLidar_transform, '20', '30', '-90', '180')
            RearVelodyneHDL64 =      gen.HDL64E(WAYMO, world, actor_list, 'rear', 5, folder_output, RearLidar_transform     , '20', '30', '-90', '180')

            # Launch WAYMO
            WAYMO.set_autopilot(True)

            # Pass to the next simulator frame to spawn sensors and to retrieve first data
            world.tick()
            ######################################## Initialize the LIDAR Sensors ###########################################
            FrontVelodyneHDL64.init(client)
            RearVelodyneHDL64.init(client)
            SideLeftVelodyneHDL64.init(client)
            SideRightVelodyneHDL64.init(client)
            TopVelodyneHDL64.init(client)
            # Set the CAMERA view to the ego vehicle
            gen.follow(WAYMO.get_transform(), world)
            
            # All sensors produce first data at the same time (this ts)
            gen.Sensor.initial_ts = world.get_snapshot().timestamp.elapsed_seconds
            
            start_record = time.time()
            print("Start record : ")
            frame_current = 0
            syn_waymo_dataset = []
            while (frame_current < nbr_frame):
                frame_current, front_lidar_features, timestamp = FrontVelodyneHDL64.save(map_name)
                tmp, rear_lidar_features, ts = RearVelodyneHDL64.save(map_name)
                tmp, sl_lidar_features, ts = SideLeftVelodyneHDL64.save(map_name)
                tmp, sr_lidar_features, ts = SideRightVelodyneHDL64.save(map_name)
                tmp, top_lidar_features, ts = TopVelodyneHDL64.save(map_name)
                
                object_features = gen.save_objects_labels(vechiles_bikes_bbs_list, walkers_bbs_list, signs_bbs_list, WAYMO)
                
                frame_bytes = gen.Get_Serialized_Frame_Example(map_name, frame_current, timestamp, front_lidar_features, rear_lidar_features, sl_lidar_features, sr_lidar_features, top_lidar_features, object_features)
                syn_waymo_dataset.append(frame_bytes)
                
                gen.follow(WAYMO.get_transform(), world)
                world.tick()    # Pass to the next simulator frame
           
            print("Writing Synthetic Waymo Dataset: ")
            gen.write_synthetic_waymo_tfrecord(folder_output+'train-syn-carla-' + map_name + '.tfrecords', syn_waymo_dataset)
            syn_waymo_dataset = []
            ############################################### Stop Recording and Destroy Spawned Vehicles ###############################################
            client.stop_recorder()
            print("Stop record")
            
            # Stopping synchronous mode before finishing to prevent the server blocking, waiting forever for a tick.
            if synchronous_master:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.no_rendering_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
                
                traffic_manager = client.get_trafficmanager()
                traffic_manager.set_synchronous_mode(False)

            print('Destroying %d vehicles' % len(vehicles_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
            vehicles_list.clear()
            
            # Stop walker controllers (list is [controller, actor, controller, actor ...])
            all_actors = world.get_actors(all_walkers_id)
            for i in range(0, len(all_walkers_id), 2):
                all_actors[i].stop()
            print('Destroying %d walkers' % (len(all_walkers_id)//2))
            client.apply_batch([carla.command.DestroyActor(x) for x in all_walkers_id])
            all_walkers_id.clear()
                
            print('Destroying WAYMO vehicle and its sensors')
            # Waymo Vehicle doesn't have stop attribute as other sensors
            for i in range(1, len(actor_list)):
                actor_list[i].stop()
            # Destroy the waymo vehicle
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            actor_list.clear()
                
            print("Elapsed time : ", time.time()-start_record)
            print()
                
            time.sleep(5.0)

    finally:
        print("Elapsed total time : ", time.time()-start_record_full)
        world.apply_settings(init_settings)
        
        time.sleep(2.0)
        

if __name__ == '__main__':
    main()
