from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from vujade import vujade_nms as nms_
from vujade import vujade_profiler as prof_


parser = argparse.ArgumentParser(description='RetinaFace')
# Parameters that can be changed.
parser.add_argument('--aws_neuron', action="store_true", default=False, help='use the AWS neuron')
parser.add_argument('--nms_type', type=str, default='nms_torchvision', help='NMS type: nms_cy_ndarr; nms_py_ndarr; nms_torchvision; nms_py_tensor')
# Fixed parameters
parser.add_argument('--trained_model_pth', default='./weights/Resnet50_Final.pth', type=str, help='PyTorch model')
parser.add_argument('--trained_model_neuron', default='./weights/Resnet50_Final_neuron.pt', type=str, help='PyTorch model')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
# NMS parameters
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    # Usage: python3 ./detect_optimized.py
    # Usage: python3 ./detect_optimized.py --aws_neuron
    torch.set_grad_enabled(False)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model

    if args.aws_neuron is False:
        # PyTorch
        net = RetinaFace(cfg=cfg, phase = 'test')
        net = load_model(net, args.trained_model_pth, args.cpu)
        net.eval()
        net = net.to(device)
    else:
        # AWS-Neuron
        if os.path.isfile(args.trained_model_neuron) is False:
            raise FileNotFoundError('The AWS neuron weights are not existed.')
        else:
            net = torch.jit.load(args.trained_model_neuron)
    # print('Finished loading model!')

    # Ignore the first attempt
    ndarr_img = np.random.rand(1, 3, 432, 768).astype(np.float32)
    tensor_img = torch.from_numpy(ndarr_img).to(device)
    loc, conf, landms = net(tensor_img)

    resize = 1
    avgmeter_time_net = prof_.AverageMeterTime(_warmup=0)
    avgmeter_time_total = prof_.AverageMeterTime(_warmup=0)

    # testing begin
    for i in range(10):
        image_path = "./curve/test.jpg"
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_ori = cv2.resize(img_raw, dsize=(768, 432), interpolation=cv2.INTER_LINEAR) # Fixed ndarray shape for the AWS-Neuron
        img = img_ori.copy().astype(np.float32)

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        im_batch, im_channel, im_height, im_width = img.shape

        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
        scale1 = torch.Tensor([im_width, im_height, im_width, im_height,
                               im_width, im_height, im_width, im_height,
                               im_width, im_height]).to(device)

        # Prior anchor box
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data

        avgmeter_time_total.tic()
        avgmeter_time_net.tic()
        loc, conf, landms = net(img) # forward pass
        avgmeter_time_net.toc()

        dets = nms_.nms_cpu(_loc=loc,
                            _conf=conf,
                            _landms=landms,
                            _prior_data=prior_data,
                            _scale_boxes=scale,
                            _scale_landms=scale1,
                            _scaling_ratio=resize,
                            _variance=cfg['variance'],
                            _confidence_threshold=args.confidence_threshold,
                            _nms_threshold=args.nms_threshold,
                            _top_k=args.top_k,
                            _keep_top_k=args.keep_top_k, _nms=args.nms_type)

        avgmeter_time_total.toc()

        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_ori, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_ori, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_ori, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_ori, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_ori, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_ori, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_ori, (b[13], b[14]), 1, (255, 0, 0), 4)

            # save image
            cv2.imwrite('test.jpg', img_ori)

    print('[CNN]   Total time: {:.2f}, Avg. time: {:.2f}'.format(avgmeter_time_net.time_sum, avgmeter_time_net.time_avg))
    print('[Total] Total time: {:.2f}, Avg. time: {:.2f}'.format(avgmeter_time_total.time_sum, avgmeter_time_total.time_avg))
