from constants import Constants
import cv2


class Drawer:

    def __init__(self):
        self.frame = 0

    def draw_team(self, frame, team_bounding_boxes, color):
        for b in team_bounding_boxes:
            x = int(b[0])
            y = int(b[1])
            w = int(b[2] / 2)
            h = int(b[3] / 2)
            cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), color, 2)

    def draw_player(self, frame, bounding_box, team):
        color = (0, 0, 255) if team == Constants.attacking_team else (0, 255, 0)
        x = int(bounding_box[0])
        y = int(bounding_box[1])
        w = int(bounding_box[2] / 2)
        h = int(bounding_box[3] / 2)
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), color, 2)
        #cv2.putText(frame, str(self.frame), (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),lineType=cv2.LINE_AA)

    def bigger_bb(self, frame, bounding_box, team):
        color = (0, 0, 255) if team == Constants.attacking_team else (0, 255, 0)
        x = int(bounding_box[0])
        y = int(bounding_box[1])
        w = int(bounding_box[2])
        h = int(bounding_box[3])
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), color, 2)

    def update_players(self, frame, bounding_boxes, teams):
        self.frame += 1
        for idx, bb in enumerate(bounding_boxes):
            if bb is not None and teams[idx] is not None:
                x, y, w, h = bb
                self.draw_player(frame, (int(x + w / 2), int(y + h / 2), w, h), teams[idx])

    def draw_offside_line(self, frame, offside_line):
        if offside_line:
            cv2.line(frame, offside_line[0],offside_line[1],(255,255,0),2)
            

    def draw_all_players(self, players_bbs, teams, frame):
        self.frame += 1
        for i in range(len(players_bbs)):
            if teams[i] is not None:
                self.draw_player(frame, players_bbs[i][2], teams[i])
