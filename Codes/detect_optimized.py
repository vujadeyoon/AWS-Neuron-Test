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


parser = argparse.ArgumentParser(description='RetinaFace')
parser.add_argument('--trained_model_pth', default='./weights/Resnet50_Final.pth', type=str, help='PyTorch model')
parser.add_argument('--trained_model_neuron', default='./weights/Resnet50_Final_neuron.pt', type=str, help='PyTorch model')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--aws_neuron', action="store_true", default=False, help='use the AWS neuron')
parser.add_argument('--nms_type', type=str, default='nms_torchvision', help='NMS type: nms_cy_ndarr; nms_py_ndarr; nms_torchvision; nms_py_tensor')
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
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
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
    #

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
    print('Finished loading model!')

    # print(net)
    resize = 1

    # testing begin
    for i in range(100):
        image_path = "./curve/test.jpg"
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_raw = cv2.resize(img_raw, dsize=(768, 432), interpolation=cv2.INTER_LINEAR) # Fixed ndarray shape for the AWS-Neuron

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize

        if args.nms_type != 'nms_torchvision':
            boxes = boxes.cpu().numpy()
            landms = landms.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        dets = nms_.nms_cpu(_loc=loc,
                            _conf=conf,
                            _landms=landms,
                            _prior_data=prior_data,
                            _scale_boxes=scale,
                            _scale_landms=scale1,
                            _scaling_ratio=resize, _variance=cfg['variance'],
                            _confidence_threshold=args.confidence_threshold,
                            _nms_threshold=args.nms_threshold,
                            _top_k=args.top_k,
                            _keep_top_k=args.keep_top_k, _nms=args.nms_type)

        # # ignore low scores
        # inds = np.where(scores > args.confidence_threshold)[0]
        # boxes = boxes[inds]
        # landms = landms[inds]
        # scores = scores[inds]
        #
        # # keep top-K before NMS
        # order = scores.argsort()[::-1][:args.top_k]
        # boxes = boxes[order]
        # landms = landms[order]
        # scores = scores[order]
        #
        # # do NMS
        # dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)
        # # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        # dets = dets[keep, :]
        # landms = landms[keep]
        #
        # # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]
        #
        # dets = np.concatenate((dets, landms), axis=1)

        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            name = "test.jpg"
            cv2.imwrite(name, img_raw)
