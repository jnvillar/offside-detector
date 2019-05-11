import cv2
import numpy as np
from matplotlib import pyplot as plt
from constants import Constants
from drawer import Drawer
import statistics
import math

class Classifier:

    def __init__(self, attack_team, defense_team):
        self.player_histograms = []
        self.buckets_per_color = 200
        self.teams_set = False
        self.attacking_idx = 0
        self.teams = []
        self.drawer = Drawer()
        self.attack_team = attack_team
        self.defense_team = defense_team

    def restart(self):
        self.player_histograms = []

    def generate_histograms(self, players_bb, frame):
        self.restart()
        for i in range(len(players_bb)):
            bb = players_bb[i]
            bounding_box = bb[2]
            player_histogram = self.calculate_histogram(frame, bounding_box)
            self.player_histograms.append(player_histogram)

    def _is_grass(self, pixel):
        rgbColor = np.uint8([[[0, 255, 0]]])
        hsvColor = cv2.cvtColor(rgbColor, cv2.COLOR_BGR2HSV)
        sensitivity = 20
        COLOR_MIN = np.array([hsvColor[0][0][0] - sensitivity, 100, 70])
        COLOR_MAX = np.array([hsvColor[0][0][0] + sensitivity, 255, 255])
        pixel = cv2.cvtColor(np.uint8([[[pixel[0], pixel[1], pixel[2]]]]), cv2.COLOR_BGR2HSV)
        return COLOR_MIN[0] <= pixel[0][0][0] <= COLOR_MAX[0]

    def calculate_mask(self, frame, bounding_box):
        mask = np.zeros(frame.shape[:2], np.uint8)

        x = int(bounding_box[0])
        y = int(bounding_box[1])
        w = int(bounding_box[2])
        h = int(bounding_box[3])

        initial_x = int(x - (w / 2))
        initial_y = int(y - (h / 2))

        mask[initial_y: initial_y + h, initial_x: initial_x + w] = 255
        for j in range(initial_y, initial_y + h):
            for i in range(initial_x, initial_x + w):
                if self._is_grass(frame[j, i]):
                    mask[j, i] = 0
        return mask

    def calculate_histogram(self, frame, bounding_box):
        mask = self.calculate_mask(frame, bounding_box)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h1 = cv2.calcHist([hsv_frame], [1], mask, [255], [0, 255])

        return {'h': h1, 'bb': bounding_box}

    def histogram_distance(self, h1, h2):
        diff = cv2.compareHist(h1['h'], h2['h'], method=cv2.HISTCMP_CHISQR)
        return diff

    def max_histogram_diff(self, histogram, players_histograms):
        max_distance = 0
        i = 0
        for idx, player_histogram in enumerate(players_histograms):
            distance = self.histogram_distance(histogram, player_histogram)
            if distance > max_distance:
                max_distance = distance
                i = idx
        return i

    def calculate_distances(self, player_histograms):
        players = len(player_histograms)
        distances = np.zeros((players, players))
        in_process = 0
        for player_hh in player_histograms:
            next = in_process + 1
            while next < players:
                distance = self.histogram_distance(player_hh, player_histograms[next])
                distances[in_process, next] = distance
                distances[next, in_process] = distance
                next += 1
            in_process += 1
        return distances

    def max_difference(self, matrix):
        max_i = 0
        max_j = 0
        max = 0
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if max < matrix[i, j]:
                    max_i = i
                    max_j = j
                    max = matrix[i, j]
        return max_i, max_j

    def difference_with_others(self, i, distances):
        sum = 0
        for x in range(len(distances)):
            sum += distances[i][x]
        return sum

    def max_distance_with_others(self, distances):
        i = 0
        max = 0
        for p in range(len(distances)):
            dist = self.difference_with_others(p, distances)
            if max < dist:
                max = dist
                i = p
        return i

    def delete_outliers(self, referees, distances, players_bb):
        for _ in range(referees):
            i = self.max_distance_with_others(distances)
            for p in range(len(distances)):
                distances[i][p] = 0
            players_bb.pop(i)
        return players_bb

    def check_points(self, points):
        if len(points) < 1:
            print("No points detected")
            exit()

    def get_attacking_player_index(self, frame):
        plt.imshow(frame)
        points = plt.ginput(1, show_clicks=True, timeout=60)

        for i in plt.get_fignums():
            plt.close(i)

        self.check_points(points)
        click = points[-1]
        for idx, player_bb in enumerate(self.player_histograms):
            x = int(player_bb['bb'][0])
            y = int(player_bb['bb'][1])
            w = int(player_bb['bb'][2] / 2)
            h = int(player_bb['bb'][3] / 2)
            if click[0] > (x - w) and click[0] < (x + w) and click[1] < (y + h) and click[1] > (y - h):
                self.attacking_idx = idx
                return idx

    def closest_bb_to_player(self, player_bb, previous_bbs):
        min_distance = None
        team = None

        for idx, bb in enumerate(previous_bbs):

            if bb is None:
                continue

            x = abs(int(player_bb['bb'][0]) - int(bb[0]))
            y = abs(int(player_bb['bb'][1]) - int(bb[1]))
            w = abs(int(player_bb['bb'][2]) - int(bb[2]))
            h = abs(int(player_bb['bb'][3]) - int(bb[3]))
            distance = x + y + w + h

            if (min_distance is None or distance < min_distance) and self.teams[idx] is not None:
                min_distance = distance
                team = self.teams[idx]

        return min_distance, team

    def closest_player_with_team(self, players_bb, previous_bbs):
        min_distance = None
        player = None
        previous_team = None

        for player_bb in players_bb:
            distance, team = self.closest_bb_to_player(player_bb, previous_bbs)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                player = player_bb
                previous_team = team

        return player, previous_team

    def distance_between_colors(self, one_color, another_color):
        x = (one_color[0] - another_color[0])**2
        y = (one_color[1] - another_color[1])**2
        z = (one_color[2] - another_color[2])**2
        return math.sqrt(x + y + z)

    def add_distance_to_color(self, frame, player_bb, color):
        
        distance_to_color = 0

        x = int(player_bb['bb'][0])
        y = int(player_bb['bb'][1])
        w = int(player_bb['bb'][2])
        h = int(player_bb['bb'][3])
        
        
        initial_x = int(x - (w / 2))
        initial_y = int(y - (h / 2))

        count = 0
        for j in range(initial_y, initial_y + h):
            for i in range(initial_x, initial_x + w):
                
                if self.distance_between_colors(frame[j, i], color) < 100:
                    count +=1

        player_bb['distance_to_color'] = count/float(h*w)
        return player_bb

    def median_to_color(self, player_bbs):
        distances_to_color = [player_bb['distance_to_color'] for player_bb in player_bbs]
        return statistics.median(distances_to_color)

    def delete_closest_to_color(self, frame, player_bbs, color, tolerance=True):
        is_deleted = False
        player_bbs_with_distances_to_color = []
        for player_bb in player_bbs:
            player_bbs_with_distances_to_color.append(self.add_distance_to_color(frame, player_bb, color))

        player_bbs_with_distances_to_color.sort(key=lambda x: -x['distance_to_color'])

        if not tolerance:
            return player_bbs_with_distances_to_color.pop(0)

        median_to_color = self.median_to_color(player_bbs_with_distances_to_color)

        if player_bbs_with_distances_to_color[0]['distance_to_color'] > 0.075:
            if Constants.debug_classifier: print('classifier: deleted closest to color', color)
            player_bbs_with_distances_to_color.pop(0)
            is_deleted = True

        return player_bbs_with_distances_to_color, is_deleted

    def _is_color(self, pixel, color_range):
        COLOR_MIN, COLOR_MAX = getattr(Constants, '{}_COLOR_RANGE'.format(color_range.upper()))
        pixel = cv2.cvtColor(np.uint8([[[pixel[0], pixel[1], pixel[2]]]]), cv2.COLOR_BGR2HSV)
        return (COLOR_MIN[0] <= pixel[0][0][0] <= COLOR_MAX[0] 
            and COLOR_MIN[1] <= pixel[0][0][1] <= COLOR_MAX[1] 
            and COLOR_MIN[2] <= pixel[0][0][2] <= COLOR_MAX[2]
        )

    def get_color_score(self, frame, player_bb, color_range):
        
        x = int(player_bb[2][0])
        y = int(player_bb[2][1])
        w = int(player_bb[2][2])
        h = int(player_bb[2][3])
                
        initial_x = int(x - (w / 2))
        initial_y = int(y - (h / 2))

        count = 0
        for j in range(initial_y, initial_y + h):
            for i in range(initial_x, initial_x + w):
                if self._is_color(frame[j, i], color_range):
                    count +=1

        return count

    def calculate_teams_from_params(self, frame, bounding_boxes):
        THRESHOLD = 30
        res = []
        for i in range(len(bounding_boxes)):
            attacking_value = self.get_color_score(frame, bounding_boxes[i], self.attack_team)
            defending_value = self.get_color_score(frame, bounding_boxes[i], self.defense_team)
            print('attacking_value', attacking_value)
            print('defending_value', defending_value)
            
            if max(attacking_value, defending_value) < THRESHOLD:
                team = None
            elif attacking_value > defending_value:
                team = Constants.attacking_team
            else:
                team = Constants.defending_team
            res.append(team)
        return res

    def calculate_teams(self, frame, previous_bbs=[], outliers=0):
        try:
            players_bb = list(self.player_histograms)
            if not self.teams_set:
                self.get_attacking_player_index(frame)

            # tries to delete referee , it may delete nothing
            has_deleted = True
            while has_deleted:
                 players_bb, has_deleted = self.delete_closest_to_color(frame, players_bb, Constants.yellow) 
            #players_bb = self.delete_closest_to_color(frame, players_bb, Constants.black)

            # delete one outlier to delete goalkeeper, it always deletes something
            distances = self.calculate_distances(players_bb)
            players_bb = self.delete_outliers(outliers, distances, players_bb)

            distances = self.calculate_distances(players_bb)
            player_one_i, player_two_j = self.max_difference(distances)

            attacking_team = [players_bb[player_one_i]]
            defending_team = [players_bb[player_two_j]]

            players_bb.pop(player_one_i)
            players_bb.pop(player_two_j - 1)

            for idx, player_histogram in enumerate(players_bb):
                d1 = self.histogram_distance(attacking_team[0], player_histogram)
                d2 = self.histogram_distance(defending_team[0], player_histogram)
                if d1 < d2:
                    attacking_team.append(players_bb[idx])
                else:
                    defending_team.append(players_bb[idx])

            attacking_team = [player['bb'] for player in attacking_team]
            defending_team = [player['bb'] for player in defending_team]
        except Exception as e:
            print('Team classifier failed', e)
            attacking_team, defending_team = [], []

        if not self.teams_set:
            # if teams are not set, choose attacking team based on input click
            if players_bb[self.attacking_idx]['bb'] in defending_team:
                copy = list(attacking_team)
                attacking_team = defending_team
                defending_team = copy
            self.teams_set = True
        else:
            # if teams were set previously, find closest bounding box and assign same team
            player, previous_team = self.closest_player_with_team(self.player_histograms, previous_bbs)
            #if Constants.debug_classifier: self.drawer.bigger_bb(frame, player['bb'], previous_team)
            if (player['bb'] in attacking_team and previous_team is Constants.defending_team) or (
                    player['bb'] in defending_team and previous_team is Constants.attacking_team):
                if Constants.debug_classifier: print('cambio equipos')
                copy = list(attacking_team)
                attacking_team = defending_team
                defending_team = copy

        self.teams = [
            Constants.attacking_team if player['bb'] in attacking_team else
            Constants.defending_team if player['bb'] in defending_team else
            None for player in self.player_histograms]
        return self.teams
