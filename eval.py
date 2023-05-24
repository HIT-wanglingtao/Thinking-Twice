from pycocotools.cocoeval import COCOeval
import numpy as np
import json
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from torchvision import transforms
import cv2
from model.network import FCOSDetector
import torch
import argparse

class COCOGenerator(CocoDetection):
    CLASSES_NAME = (
    '__back_ground__', 'benign', 'malignant')
    def __init__(self,imgs_path,anno_path,resize_size=[800,1024]):
        super().__init__(imgs_path,anno_path)

        print("INFO====>check annos, filtering invalid data......")
        ids=[]
        for id in self.ids:
            ann_id=self.coco.getAnnIds(imgIds=id,iscrowd=None)
            ann=self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                ids.append(id)
        self.ids=ids
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.resize_size=resize_size
        self.mean=[0.40789654, 0.44719302, 0.47026115]
        self.std=[0.28863828, 0.27408164, 0.27809835]
        

    def __getitem__(self,index):
        
        img,ann=super().__getitem__(index)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes=np.array(boxes,dtype=np.float32)
        #xywh-->xyxy
        boxes[...,2:]=boxes[...,2:]+boxes[...,:2]
        img=np.array(img)

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]

        img=transforms.ToTensor()(img)
        img= transforms.Normalize(self.mean, self.std,inplace=True)(img)
        classes=np.array(classes,dtype=np.int64)
        return img,boxes,classes

    def preprocess_img_boxes(self,image,boxes,input_ksize):
        '''
        resize image and bboxes 
        Returns
        image_paded: input_ksize  
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w, _  = image.shape

        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        pad_w=32-nw%32
        pad_h=32-nh%32
        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale 
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale 
            return image_paded, boxes,scale



    def _has_only_empty_bbox(self,annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self,annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True
    

def evaluate_coco(generator, model, threshold=0.05):
    """ Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        oU NMSgenerator : The generator for g
        model     : The model to evaluate.
        threshold : The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    for index in tqdm(range(len(generator))):
        img,gt_boxes,gt_labels = generator[index]
        # run network
        scores, labels,boxes  = model(img.unsqueeze(dim=0).cuda())
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
   
        # correct boxes for image scale
        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # compute predicted labels and scores
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:
                break

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.ids[index],
                'category_id' : generator.id2category[label],
                'score'       : float(score),
                'bbox'        : box.tolist(),
            }

            # append detection to results
            results.append(image_result)

        # append image to list of processed images
        image_ids.append(generator.ids[index])

    if not len(results):
        return

    # write output
    json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)
    coco_true = generator.coco
    coco_pred = coco_true.loadRes('coco_bbox_results.json')

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(coco_eval.stats)
    return coco_eval.stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset", type=str, default='./evaldataset', help="eval dataset")
    parser.add_argument("--eval_instances", type=str, default='./evalinstances', help="eval instances")
    parser.add_argument("--ckpt", type=str, default='./evalckpt', help="eval checkpoint")
    opt = parser.parse_args()
    generator=COCOGenerator(opt.eval_dataset,opt.eval_instances)
    model=FCOSDetector(mode="inference")
    model = torch.nn.DataParallel(model)
    model = model.cuda().eval()
    model.load_state_dict(torch.load(opt.ckpt,map_location=torch.device('cpu')))
    evaluate_coco(generator,model)
