import pickle
from constants import Constants


class Storer:

    def save_current_video_info(self, input_video, start_frame):
        file = open('./data/last_video_info.txt', 'w')
        res = {
            "input_video": input_video,
            "start_frame": start_frame,
        }
        pickle.dump(res, file)
        file.close()

    def load_last_yolo_result(self):
        file = open('./data/last_yolo_res.txt', 'r')
        try:
            res = pickle.load(file)
            return res
        except:
            return {}

    def store_yolo_result(self, res):
        file = open('./data/last_yolo_res.txt', 'w')
        pickle.dump(res, file)
        file.close()

    def use_last_yolo_result(self, input_video, start_frame):
        file = open('./data/last_video_info.txt', 'r')
        try:
            res = pickle.load(file)
        except:
            self.save_current_video_info(input_video, start_frame)
            return

        if res['input_video'] == input_video and res['start_frame'] == start_frame:
            Constants.use_last_yolo = True
        else:
            self.save_current_video_info(input_video, start_frame)
