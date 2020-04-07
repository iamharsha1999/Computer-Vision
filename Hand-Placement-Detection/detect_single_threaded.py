from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import matplotlib.pyplot as plt
import screeninfo

detection_graph, sess = detector_utils.load_inference_graph()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.4, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0, help='Device index of the camera.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')

    args = parser.parse_args()
    print(args.video_source)

    screen = screeninfo.get_monitors()[0]
    width, height = screen.width, screen.height

	##Capture Live Video Feed
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))

    # max number of hands we want to detect/track
    num_hands_detect = 1
    window_name = 'Live Feed'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    lb = [1,int((im_height//2)-50),125,int((im_height//2)+100)]
    rb = [int(im_width-125),int((im_height//2)-50),int(im_width),int((im_height//2)+100)]

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        image_np = cv2.flip(image_np, 1)

        ## Left Box
        image_np = cv2.rectangle(image_np,(1,int((im_height//2)-50)),(125,int((im_height//2)+100)),(255,0,0),3)

        ## Right Box
        image_np = cv2.rectangle(image_np,(int(im_width-125),int((im_height//2)-50)),(int(im_width),int((im_height//2)+100)),(255,0,0),3)


        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # actual detection
        boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
        # draw bounding boxes
        mp = detector_utils.draw_box_on_image(
            num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)
        print(lb)
        print(rb)
        try:
            dir = detector_utils.check_midpoint(mp,lb,rb)
        except:
            dir = None 
            pass

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image(
                    "FPS : " + str(int(fps)), image_np)
            try:
                # Display the direction on the image
                detector_utils.put_direction_on_image(dir,image_np)
            except:
                pass
            cv2.imshow(window_name, cv2.cvtColor(
                image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ",  num_frames,
                  "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
