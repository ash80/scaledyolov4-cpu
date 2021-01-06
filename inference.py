import argparse
import cv2
import torch
from utils.torch_utils import select_device
from models.models import *


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def test(opt, cv_img):
    device = select_device(opt.device, batch_size=None)
    model = Darknet(opt.cfg).to(device)

    # load model
    try:
        ckpt = torch.load(opt.weights[0], map_location=device)  # load checkpoint
        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt['model'], strict=False)
    except:
        load_darknet_weights(model, opt.weights[0])
    imgsz = check_img_size(opt.img_size, s=32)  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    
    # Configure
    model.eval()

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    cv_img = cv_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    cv_img = cv_img[np.newaxis, ...] # to 4 dimensions
    cv_img = np.ascontiguousarray(cv_img)
    print('shape: ', cv_img.shape)
    img = torch.from_numpy(cv_img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    with torch.no_grad():
        inf_out, _ = model(img, augment=False)
        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, merge=opt.merge)
        print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inference.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4.pt', help='model.pt path(s)')
    # parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    # parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-src', type=int, default='', help='source image')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    # parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    # parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    options = parser.parse_args()
    # options.save_json |= options.data.endswith('coco.yaml')
    # options.data = check_file(options.data)  # check file
    print(options)

    src_image = options.src_image
    cv_image = cv2.imread(src_image)
    cv_image = cv2.resize(cv_image, (options.img_size, options.img_size), cv2.INTER_AREA)
    test(options, cv_image)
