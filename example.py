import cv2


cap = cv2.VideoCapture('video.mp4')

out = cv2.VideoWriter('output.mp4', 0x7634706d, 30.0, (1280,720))

if (not cap.isOpened()):
	raise Exception('Video cound not be opened')


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()

cv2.destroyAllWindows()