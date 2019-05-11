import cv2
from field_detector import FieldDetector
from darknet import PlayerDetector
from classifier import Classifier
from drawer import Drawer
from vanishing_point import get_offside_line, get_vanishing_point
from player_tracker import PlayerTracker
from constants import Constants
from store import Storer
import math
import click


@click.command()
@click.option("--input_video", default='video', help="Name of video to process", show_default=True)
@click.option("--start_frame", default=0, help="Start frame", show_default=True)
@click.option("--end_frame", default=100000000, help="End frame", show_default=True)
@click.option("--vp_validation", default=False, help="Validate vanishing point using the last vp calculated", show_default=True)
@click.option("--debug", default=True, help="Should display debug info", show_default=True)
@click.option("--attack_team", default=None, help="The attacking team", show_default=True)
@click.option("--defense_team", default=None, help="The defending team", show_default=True)
def main(input_video, start_frame, end_frame, vp_validation, debug, attack_team, defense_team):
    storer = Storer()
    storer.use_last_yolo_result(input_video, start_frame)
    field_detector = FieldDetector()
    classifier = Classifier(attack_team, defense_team)
    drawer = Drawer()
    player_detector = PlayerDetector()
    player_tracker = PlayerTracker()
    cap = cv2.VideoCapture('./videos/{}.mp4'.format(input_video))
    out = cv2.VideoWriter('./videos/output-{}.mp4'.format(input_video), 0x7634706d, 30.0, (1280, 720))

    automatic_team_detection = False
    if attack_team is None or defense_team is None:
        automatic_team_detection = True

    if not cap.isOpened():
        raise Exception('Video cound not be opened')

    bounding_boxes = []

    for frame_index in range(0, end_frame):

        ret, frame = cap.read()
        if frame_index < start_frame:
            continue

        if ret == False:
            break
        img_h, img_w = frame.shape[:2]

        vp = get_vanishing_point(frame) or vp


        if frame_index == start_frame:
            first_vp = vp
        else:
            ## si el vp cambio mucho, entonces probablemente sea un outlier. Mejor usar el ultimo vp calculado. Esto asume que el primer vp es 'bueno'.
            if math.sqrt((vp[0] - vp_copy[0]) ** 2 + (vp[1] - vp_copy[1]) ** 2) > 100:
                if vp_validation:
                    vp = vp_copy
        vp_copy = vp

        # frame = field_detector.detect_field(frame, first_vp)

        frame = field_detector.detect_field(frame, vp)
        if (frame_index-start_frame) % Constants.yolo_frame_period == 0:
            players_bbs = player_detector.detect_with_yolo(frame, frame_index == start_frame)
            players_bbs = field_detector.filter_outside_bounding_boxes(players_bbs)

            player_tracker.track_players(players_bbs, frame)
            if automatic_team_detection:
                classifier.generate_histograms(players_bbs, frame)
                teams = classifier.calculate_teams(frame, bounding_boxes, outliers=0)
            else:
                teams = classifier.calculate_teams_from_params(frame, players_bbs)
            drawer.draw_all_players(players_bbs, teams, frame)
            out.write(frame)
            continue
        
        bounding_boxes = player_tracker.update(frame)
        
        drawer.update_players(frame, bounding_boxes, teams)
        leftmost_player = player_tracker.get_leftmost_player(bounding_boxes, vp, Constants.defending_team, teams)
        
        offside_line = get_offside_line(vp, leftmost_player, img_h, img_w)
        drawer.draw_offside_line(frame, offside_line)

        out.write(frame)
        if debug: print('frame: ' + str(frame_index) + ' processed')
        continue

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
