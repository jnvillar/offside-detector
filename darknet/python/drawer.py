import cv2

class Drawer:

    def draw_team(self, frame, team_bounding_boxes, color):
        for b in team_bounding_boxes:
            x = int(b[0])
            y = int(b[1])
            w = int(b[2] / 2)
            h = int(b[3] / 2)
            cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), color, 2)
