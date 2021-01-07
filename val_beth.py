import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import numpy as np
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score, dice_coef
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args() 
    # num_class = config['num_classes']
    num_class = 29

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](num_class,
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=num_class,
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()
    avg_meters = {'iou': AverageMeter(),
                'dice': AverageMeter()}

    COLORS = {0: [0, 0, 142],
			 1: [70, 130, 180],
			 2: [153, 153, 153],
			 3: [70, 70, 70],
			 4: [107, 142, 35],
			 5: [220, 220, 0],
			 6: [0, 0, 142],
			 7: [255, 0, 0],
			 8: [220, 20, 60],
			 9: [119, 11, 32],
			 10: [244, 35, 232],
			 11: [152, 251, 152],
			 12: [128, 64, 128],
			 13: [250, 170, 30],
			 14: [190, 153, 153],
			 15: [102, 102, 156],
			 16: [255, 255, 255],
			 17: [81, 0, 81],
			 18: [0, 0, 230],
			 19: [250, 170, 160],
			 20: [0, 60, 100],
			 21: [150, 100, 100],
			 22: [0, 0, 110],
			 23: [150, 120, 90],
			 24: [0, 0, 90],
			 25: [0, 0, 70],
			 26: [180, 165, 180],
			 27: [230, 150, 140],
			 28: [0, 80, 100]}

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            avg_meter.update(iou, input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
  
            for i in range(len(output)):
                #mask6 = (output[i, 6] + output[i, 10] * 255).astype('uint8')
                # mask = np.zeros((256, 512))
                colorMatr = np.zeros((256, 512, 3))
                for j in range(num_class):
                	if j != 16:
	                	color = COLORS[j]
	                	R, G, B = color[0], color[1], color[2]
	                	mask = (output[i, j] * 255).astype('uint8')
	                	hg, wd = 256, 512
	                	colorMatr_j = np.zeros((hg, wd, 3))
	                	for y in range(hg):
	                		for x in range(wd):
	                			if mask[y][x] >= 170:
	                				pixel = colorMatr_j[y][x]
	                				pixel[0], pixel[1], pixel[2] = R, G, B
	                	colorMatr += colorMatr_j
                cv2.imwrite(os.path.join('outputs', config['name'], str(6), meta['img_id'][i] + '.jpg'), colorMatr)
    print('IoU avg_meter: %.4f' % avg_meter.avg)
    print('IoU avg_meters: %.4f' % avg_meters['iou'].avg)
    print('Dice Coefficient: %.4f' % avg_meters['dice'].avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
