import numpy as np
import cv2 as cv2
from numpy.linalg import norm


class FieldDetector:

    def __init__(self, debug=False):
        self.debug = debug

    def print_line(self, rho, theta, color):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(self.img, (x1, y1), (x2, y2), color, 1)
        cv2.imshow('img', self.img)
        cv2.waitKey(0)

    def detect_field(self, frame, vp):
        self.vp = vp
        self.frame = frame
        self.field_mask = self.calculate_field_mask()
        self.bounding_rectangle = self.calculate_bounding_rectangle()
        self.left_corner_line, self.right_corner_line = self.calculate_corner_lines()
        self.bounding_polygon = self.calculate_bounding_polygon()
        
        if self.debug:
            cv2.drawContours(self.frame, np.array([self.bounding_polygon]), -1, (0, 0, 255), 4)
            cv2.imshow('Frame with field', self.frame)
            cv2.waitKey(0)

        return self.frame

    def calculate_field_mask(self):
        # color verde en BGR
        rgbColor = np.uint8([[[0, 255, 0]]])
        hsvColor = cv2.cvtColor(rgbColor, cv2.COLOR_BGR2HSV)
        sensitivity = 20
        COLOR_MIN = np.array([hsvColor[0][0][0] - sensitivity, 100, 70])
        COLOR_MAX = np.array([hsvColor[0][0][0] + sensitivity, 255, 255])
        frame_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        
        # genera mascara que deja en blanco los colores dentro del rango 
        frame_mask = cv2.inRange(frame_hsv, COLOR_MIN, COLOR_MAX)

        # mascara auxiliar usada para el flood fill. El tamano debe ser dos pixeles mayor que el de la imagen original
        h, w = frame_mask.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # flood fill a partir del pixel (0, 0)
        im_floodfill = frame_mask.copy()
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        
        # invierte mascara y luego combina con la mascara previamente calculada. Asi eliminamos "manchas" en el campo
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        field_mask = frame_mask | im_floodfill_inv

        if self.debug:
            print("VP: ")
            print(self.vp)
            cv2.imshow('field_mask', field_mask)
            cv2.waitKey(0)

        return field_mask

    def calculate_bounding_rectangle(self):
        # calcula componentes conexas con estaditicas
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.field_mask, connectivity = 8)

        # valores iniciales de los maximos y minimos
        min_x = 100000
        min_y = 100000
        max_x = -1
        max_y = -1

        # calcula la maxima y minima coordenada para cada eje, solo teniendo en cuenta componentes de color blanco en la mascara (que son las detectadas inicialmente como verde), y con area mayor a 500 pixeles (para evitar manchas aisladas) 
        label = 0
        for label_stats in stats:
            area = label_stats[cv2.CC_STAT_AREA]
            xc = int(centroids[label][0])
            yc = int(centroids[label][1])

            left_x = label_stats[cv2.CC_STAT_LEFT]
            right_x = left_x + label_stats[cv2.CC_STAT_WIDTH]
            top_y = label_stats[cv2.CC_STAT_TOP]
            bot_y = top_y + label_stats[cv2.CC_STAT_HEIGHT]
            
            # verifica si es una componente de interes
            if self.field_mask[yc, xc] == 255 and area > 500:
                # verifica si hay nuevo maximo y/o minimo en cada coordenada
                if left_x < min_x:
                    min_x = left_x

                if right_x > max_x:
                    max_x = right_x

                if top_y < min_y:
                    min_y = top_y

                if bot_y > max_y:
                    max_y = bot_y
            label += 1

        bounding_rectangle = {'topLeft': (min_x, min_y), 'topRight': (max_x, min_y), 'botLeft': (min_x, max_y), 'botRight': (max_x, max_y)}

        if self.debug:
            cv2.drawContours(self.frame, np.array([[(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]]), -1, (0, 250, 0), 4)
            cv2.imshow('Frame with field', self.frame)
            cv2.waitKey(0)

        return bounding_rectangle

    def calculate_corner_lines(self):
        # calculan los puntos verdes mas a la derecha e izquierda respectivamente
        leftmost_green_point = self.leftmost_green_point()
        rightmost_green_point = self.rightmost_green_point()

        # construye lineas de corner utilizando el vanishing point
        left_corner_line = {'p1': self.vp, 'p2': leftmost_green_point}
        right_corner_line = {'p1': self.vp, 'p2': rightmost_green_point}

        if self.debug:
            cv2.line(self.frame, self.vp, leftmost_green_point, (255, 0, 0), 4)
            cv2.line(self.frame, self.vp, rightmost_green_point, (255, 0, 0), 4)
            cv2.imshow('Frame with field', self.frame)
            cv2.waitKey(0)

        return left_corner_line, right_corner_line

    def leftmost_green_point(self):
        height = self.field_mask.shape[0]
        width = self.field_mask.shape[1]

        # itera las primeras 15 columnas de pixeles desde la izquierda. Se queda con el pixel verde mas alto (es decir el de menor valor)
        # se hace asi porque si solo se busca en la columna del borde, en muchos casos queda muy abajo el pixel porque la mascara no es perfecta.
        leftmost_green_point = (-1, 10000)
        for x in range(0, width - 1):
            for y in range(0, height - 1):
                if self.field_mask[y, x] == 255 and y < leftmost_green_point[1]:
                    leftmost_green_point = (x, y)
                if leftmost_green_point != (-1, 10000) and x > 15:
                    if self.debug:
                        print("LEFTMOST GP:")
                        print(leftmost_green_point)
                    return leftmost_green_point

    def rightmost_green_point(self):
        height = self.field_mask.shape[0]
        width = self.field_mask.shape[1]

        # itera las primeras 15 columnas de pixeles desde la derecha. Se queda con el pixel verde mas alto (es decir el de menor valor)
        # se hace asi porque si solo se busca en la columna del borde, en muchos casos queda muy abajo el pixel porque la mascara no es perfecta.
        rightmost_green_point = (-1, 10000)
        for i in range(0, width - 1):
            x = (width - 1) - i
            for y in range(0, height - 1):
                if self.field_mask[y, x] == 255 and y < rightmost_green_point[1]:
                    rightmost_green_point = (x, y)
                if rightmost_green_point != (-1, 10000) and x < ((width - 1) - 15):
                    if self.debug:
                        print("RIGHTMOST GP:")
                        print(rightmost_green_point)
                    return rightmost_green_point

    def calculate_bounding_polygon(self):
        # left_corner_line = {'p1': vp, 'p2': leftmost_green_point}
        # right_corner_line = {'p1': vp, 'p2': rightmost_green_point}
        # rectangle = {'topLeft': (minX, minY), 'topRight': (maxX, minY), 'botLeft': (minX, maxY), 'botRight': (maxX, maxY)}

        # construye lineas correspondientes a cada uno de los lado del rectangulo
        top_side = {'p1': self.bounding_rectangle['topLeft'], 'p2': self.bounding_rectangle['topRight']}
        bot_side = {'p1': self.bounding_rectangle['botLeft'], 'p2': self.bounding_rectangle['botRight']}
        left_side = {'p1': self.bounding_rectangle['topLeft'], 'p2': self.bounding_rectangle['botLeft']}
        right_side = {'p1': self.bounding_rectangle['topRight'], 'p2': self.bounding_rectangle['botRight']}
        
        # calcula intersecciones entre la linea de corner izquierda y el lado superior, el izquierdo y el inferior
        left_intersection_top = self.lines_intersection(self.left_corner_line, top_side)
        left_intersection_left = self.lines_intersection(self.left_corner_line, left_side)
        left_intersection_bot = self.lines_intersection(self.left_corner_line, bot_side)

        # calcula intersecciones entre la linea de corner derecha y el lado superior, el derecho y el inferior
        right_intersection_top = self.lines_intersection(self.right_corner_line, top_side)
        right_intersection_right = self.lines_intersection(self.right_corner_line, right_side)
        right_intersection_bot = self.lines_intersection(self.right_corner_line, bot_side)

        # define los puntos del poligono. Las intersecciones fuera de rango son descartadas
        polygon_points = []
        if self.bounding_rectangle['topLeft'][0] <= left_intersection_top[0] <= self.bounding_rectangle['topRight'][0]:
            polygon_points.append(left_intersection_top)
        else:
            polygon_points.append(self.bounding_rectangle['topLeft'])
        
        if self.bounding_rectangle['topLeft'][1] <= left_intersection_left[1] <= self.bounding_rectangle['botLeft'][1]:
            polygon_points.append(left_intersection_left)

        if self.bounding_rectangle['botLeft'][0] <= left_intersection_bot[0] <= self.bounding_rectangle['botRight'][0]:
            polygon_points.append(left_intersection_bot)
        else:
            polygon_points.append(self.bounding_rectangle['botLeft'])
        
        if self.bounding_rectangle['botLeft'][0] <= right_intersection_bot[0] <= self.bounding_rectangle['botRight'][0]:
            polygon_points.append(right_intersection_bot)
        else:
            polygon_points.append(self.bounding_rectangle['botRight'])

        if self.bounding_rectangle['topRight'][1] <= right_intersection_right[1] <= self.bounding_rectangle['botRight'][1]:
            polygon_points.append(right_intersection_right)

        if self.bounding_rectangle['topLeft'][0] <= right_intersection_top[0] <= self.bounding_rectangle['topRight'][0]:
            polygon_points.append(right_intersection_top)
        else:
            polygon_points.append(self.bounding_rectangle['topRight'])

        return polygon_points

    def lines_intersection(self, line1, line2):
        # calcula interseccion entre dos lineas (las lineas se representan mediante 2 puntos)
        p1, p2, p3, p4 = line1['p1'], line1['p2'], line2['p1'], line2['p2']
        x = ((p1[0]*p2[1] - p1[1]*p2[0]) * (p3[0]-p4[0]) - (p1[0]-p2[0]) * (p3[0]*p4[1] - p3[1]*p4[0])) / ((p1[0]-p2[0]) * (p3[1]-p4[1]) - (p1[1]-p2[1]) * (p3[0]-p4[0]))
        y = ((p1[0]*p2[1] - p1[1]*p2[0]) * (p3[1]-p4[1]) - (p1[1]-p2[1]) * (p3[0]*p4[1] - p3[1]*p4[0])) / ((p1[0]-p2[0]) * (p3[1]-p4[1]) - (p1[1]-p2[1]) * (p3[0]-p4[0]))
        return (x,y)

    def filter_outside_bounding_boxes(self, bounding_boxes):
        filtered_bounding_boxes = []
        for bounding_box in bounding_boxes:
            if self.field_includes_bounding_box(bounding_box):
                filtered_bounding_boxes.append(bounding_box)
            
        return filtered_bounding_boxes

    def field_includes_bounding_box(self, bounding_box):
        #considera incluida la bounding box si al menos uno de sus vertices inferiores esta incluido en el poligono
        x_center_point = int(bounding_box[2][0])
        y_center_point = int(bounding_box[2][1])
        width = int(bounding_box[2][2])
        height = int(bounding_box[2][3])

        bot_left = (int(x_center_point - (width / 2)), int(y_center_point + (height / 2)))
        bot_right = (int(x_center_point + (width / 2)), int(y_center_point + (height / 2)))

        if cv2.pointPolygonTest(np.array([self.bounding_polygon]), bot_left, measureDist=False) >= 0:
            return True

        if cv2.pointPolygonTest(np.array([self.bounding_polygon]), bot_right, measureDist=False) >= 0:
            return True

        return False