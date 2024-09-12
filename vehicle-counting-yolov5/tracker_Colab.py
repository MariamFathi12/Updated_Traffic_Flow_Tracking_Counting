#################### Tracker #################################################
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

up_count = 0
down_count = 0
car_count = 0
truck_count = 0
bus_count=0
tracker1 = []
tracker2 = []

dir_data = {}

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.weights, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    save_vid=True
    # Initialize DeepSort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # Half precision only supported on CUDA

    # Capture the input video resolution
    cap = cv2.VideoCapture(opt.source)  # Open the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the input video
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the input video

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to its own .txt file
    if not evaluate:
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make directory

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # Check image size

    # Set video writer to match input resolution
    fourcc = cv2.VideoWriter_fourcc(*opt.fourcc)  # Define codec for video writer
    vid_path, vid_writer = None, None
    try:
        
        for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(LoadImages(source, img_size=imgsz, stride=stride)):
            # print(f"Image: {img.shape} ")
            # print(f"Image Type: {type(img)} ")
            
            t1 = time_sync()

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # Normalize
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()

            # Inference
            pred = model(img, augment=opt.augment, visualize=opt.visualize)
            t3 = time_sync()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                im0 = im0s.copy()
                annotator = Annotator(im0, line_width=2, pil=not ascii)
                w, h = im0.shape[1],im0.shape[0]

                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # Pass detections to DeepSort
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    # Draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            label = f'{id} {names[int(cls)]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(int(cls), True))
                            #count
                            count_obj(bboxes,w,h,id,"South",int(cls))

                im0 = annotator.result()
                if show_vid:
                    global up_count,down_count
                    color=(0,0,255)
                    # print(f"Shape: {im0.shape}")

                    # Left Lane Line
                    #cv2.line(im0, (0, h-300), (600, h-300), (255,0,0), thickness=3)

                    # Right Lane Line
                    cv2.line(im0,(500,h-300),(w,h-300),(0,0,255),thickness=3)
                    
                    thickness = 3 # font thickness
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 2.5 
                    #cv2.putText(im0, "Outgoing Traffic:  "+str(up_count), (60, 150), font, 
                    #   fontScale, (0,0,255), thickness, cv2.LINE_AA)

                    cv2.putText(im0, "Incoming Traffic:  "+str(down_count), (700,150), font, 
                    fontScale, (0,255,0), thickness, cv2.LINE_AA)
                    
                    # -- Uncomment the below lines to computer car and truck count --
                    # It is the count of both incoming and outgoing vehicles 
                    
                    #Objects 
                    cv2.putText(im0, "Cars:  "+str(car_count), (60, 250), font, 
                    1.5, (20,255,0), 3, cv2.LINE_AA)                

                    cv2.putText(im0, "Trucks:  "+str(truck_count), (60, 350), font, 
                    1.5, (20,255,0), 3, cv2.LINE_AA)  

                    cv2.putText(im0, "Busses:  "+str(bus_count), (60, 450), font, 
                    1.5, (20,255,0), 3, cv2.LINE_AA)  
                    

                    
                    end_time = time.time()
                    cv2.putText(im0, "FPS: " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                    im0 = cv2.resize(im0, (1000,700))

                
                # Define save_path to save the processed video
                save_path = str(Path(save_dir) / Path(path).name)

                # Save video results
                if save_vid:
                    if vid_path != save_path:  # New video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # Release previous video writer

                        # Set video writer to match input resolution
                        vid_writer = cv2.VideoWriter('saved_updated_output.mp4', fourcc, fps, (frame_width, frame_height))

                    vid_writer.write(im0)

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    
    finally:
        # Ensure the video writer is released properly
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
        print("Video writer released and process completed.")




def count_obj(box,w,h,id,direct,cls):
    global up_count,down_count,tracker1, tracker2, car_count, truck_count,bus_count
    cx, cy = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))



    # For South

    if cy<= int(h//2):
        return

    if direct=="South":

        if cy > (h - 300):
            if id not in tracker1:
                print(f"\nID: {id}, H: {h} South\n")
                down_count +=1
                tracker1.append(id)

                if cls==2:
                    car_count+=1
                elif cls==7:
                    truck_count+=1
                elif cls==5:
                    bus_count+=1

            
    elif direct=="North":
        if cy < (h - 150):
            if id not in tracker2:
                print(f"\nID: {id}, H: {h} North\n")
                up_count +=1
                tracker2.append(id)
                
                #if cls==2:
                #    car_count+=1
                #elif cls==7:
                #    truck_count+=1
                #elif cls==5:
                #    bus_count+=1



def direction(id,y):
    global dir_data

    if id not in dir_data:
        dir_data[id] = y
    else:
        diff = dir_data[id] -y

        if diff<0:
            return "South"
        else:
            return "North"


if __name__ == '__main__':
    __author__ = '-'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='input.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default='store_true', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
