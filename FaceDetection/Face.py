from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from IPython import display


def get_camera():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)

    cv2.namedWindow("preview")
    cv2.namedWindow("whole")

    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
        
    while rval:
        boxes, _ = mtcnn.detect(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_draw = Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB))
        if boxes is not None:
            draw = ImageDraw.Draw(frame_draw)
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

            croped = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(boxes[0])
            print(np.array(croped)[:, :, ::-1])
            cv2.imshow("preview", np.array(croped)[:, :, ::-1])

        cv2.imshow("whole", np.array(frame_draw)[:, :, ::-1])
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyAllWindows()



def crop_face(img):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)
    boxes, _ = mtcnn.detect(img)
    
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    croped = []
    for box in boxes:
        print(box)
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        croped.append(img.crop(box)) 
    img_draw.show()

    return croped


if __name__ == "__main__":
    get_camera()