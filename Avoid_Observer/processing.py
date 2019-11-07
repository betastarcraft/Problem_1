import numpy as np
from core.common.processor import Processor



class ObsProcessor(Processor):
    def __init__(self):
        super(ObsProcessor, self).__init__()
        self.highest_height = 1900
        self.last_action = None
    
    
    def process_observation(self, observation, verbose=False):

        # scurge's map
        map_of_scurge = np.zeros(shape=(64, 64))

        me_x = observation.my_unit[0].pos_x
        me_y = observation.my_unit[0].pos_y

        me_x_t = np.clip(int(me_x/32), 0, 64)
        me_y_t = np.clip(int(me_y/32), 0, 64)
        if verbose:
            print('my location:', [me_x_t, me_y_t])

        # Safe zone : left-top (896, 1888) right-bottom (1056, 2048) with additional (marginal) space
        for x in range(int(896/32), int(1056/32)): # 28~33
            for y in range(int(1888/32), int(2048/32)): # 59~64
                map_of_scurge[y][x] = -1  # masking safe zone

        # Goal line : left-top (0, 0) right-bottom (2048, 64) with additional (marginal) space
        for x in range(int(0/32), int(2048/32)): # 0~64
            for y in range(int(0/32), int(64/32)): # 0~2
                map_of_scurge[y][x] = -1  # masking safe zone

        # masking observer's location
        map_of_scurge[me_y_t][me_x_t] = 1
        map_of_scurge = np.expand_dims(map_of_scurge, -1)

        
        LOCAL_OBSERVABLE_TILE_SIZE = 10
        
        # observer map
        map_of_observer = np.zeros(shape=(LOCAL_OBSERVABLE_TILE_SIZE*2+1, LOCAL_OBSERVABLE_TILE_SIZE*2+1))
        map_of_observer[LOCAL_OBSERVABLE_TILE_SIZE, LOCAL_OBSERVABLE_TILE_SIZE] = -1

        for ob in observation.en_unit:
            en_x_t = ob.pos_x / 32
            en_y_t = ob.pos_y / 32

            # scurge를 중앙에 두기 위해
            rel_x = int(en_x_t - me_x_t) + LOCAL_OBSERVABLE_TILE_SIZE
            rel_y = int(en_y_t - me_y_t) + LOCAL_OBSERVABLE_TILE_SIZE

            rel_x = np.clip(rel_x, 0, LOCAL_OBSERVABLE_TILE_SIZE*2-1)
            rel_y = np.clip(rel_y, 0, LOCAL_OBSERVABLE_TILE_SIZE*2-1)
            if verbose:
                print('enemy location:', [en_x_t, en_y_t], '(relevant:', [rel_x, rel_y], ')')

            map_of_observer[rel_y][rel_x] = map_of_observer[rel_y][rel_x] + 1  # if two or more observers are duplicated, we use sum

        # display out of map where scurge can't go based on current location of scurge
        scurge_out_of_map_left = me_x_t - LOCAL_OBSERVABLE_TILE_SIZE
        scurge_out_of_map_right = me_x_t + LOCAL_OBSERVABLE_TILE_SIZE
        scurge_out_of_map_up = me_y_t - LOCAL_OBSERVABLE_TILE_SIZE
        scurge_out_of_map_down = me_y_t + LOCAL_OBSERVABLE_TILE_SIZE

        if scurge_out_of_map_left < 0:
            map_of_observer[:, 0:-scurge_out_of_map_left] = -2
        if scurge_out_of_map_right > 64:
            map_of_observer[:, -(scurge_out_of_map_right-64):] = -2
        if scurge_out_of_map_up < 0:
            map_of_observer[0:-scurge_out_of_map_up,:] = -2
        if scurge_out_of_map_down > 64:
            map_of_observer[-(scurge_out_of_map_down-64):,:] = -2

        map_of_observer = np.expand_dims(map_of_observer, -1)

        if not self.last_action:
            self.last_action = np.full((64, 64), -1)
        else:
            self.last_action = np.full((64, 64), self.last_action)

        if verbose:
            print(map_of_scurge.shape)
            print(map_of_observer.shape)
            print(self.last_action.shape)

        return map_of_scurge, map_of_observer