from .head import ClsCntRegHead
from .fpn_neck import FPN
import torch.nn as nn
from .loss import GenTargets,LOSS,coords_fmap2orig
import torch
from .config import DefaultConfig
from .backbone.convnext import ConvNeXt
from .saspp import SASPP
class FCOS(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        self.backbone=ConvNeXt()
        self.backbone.load_state_dict(torch.load("./convnext.pth")['model'],strict=False)
        self.fpn=FPN(config.fpn_out_channels,use_p5=True)
        self.saspp3=SASPP(256,256)
        self.saspp4=SASPP(256,256)
        self.saspp5=SASPP(256,256)
        self.rfp_conv3=nn.Conv2d(256, 192, kernel_size=1, padding=0, bias=True)
        self.rfp_conv4=nn.Conv2d(256, 384, kernel_size=1, padding=0, bias=True)
        self.rfp_conv5=nn.Conv2d(256, 768, kernel_size=1, padding=0, bias=True)
        self.rfp_conv3.weight.data.fill_(0)
        self.rfp_conv3.bias.data.fill_(0)
        self.rfp_conv4.weight.data.fill_(0)
        self.rfp_conv4.bias.data.fill_(0)
        self.rfp_conv5.weight.data.fill_(0)
        self.rfp_conv5.bias.data.fill_(0)
        self.rfp_weight = torch.nn.Conv2d(
            256,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        
        
        self.rfp2_weight = torch.nn.Conv2d(
            256,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.rfp3_weight = torch.nn.Conv2d(
            256,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
            
        self.rfp_weight.weight.data.fill_(0)
        self.rfp_weight.bias.data.fill_(0)
        self.rfp2_weight.weight.data.fill_(0)
        self.rfp2_weight.bias.data.fill_(0)
        self.rfp3_weight.weight.data.fill_(0)
        self.rfp3_weight.bias.data.fill_(0)  
        self.head=ClsCntRegHead(config.fpn_out_channels,config.class_num,
                                config.use_GN_head,config.cnt_on_reg,config.prior)
        self.config=config


    def forward(self,x):
        out=self.backbone(x)
        C3, C4, C5=out['C1'],out['C2'],out['C3']
        all_P=self.fpn([C3,C4,C5])
        P3,P4,P5,P6,P7=all_P[0],all_P[1],all_P[2],all_P[3],all_P[4]
        P3=self.saspp3(P3)
        P4=self.saspp4(P4)
        P5=self.saspp5(P5)
        R3=self.rfp_conv3(P3)
        R4=self.rfp_conv4(P4)
        R5=self.rfp_conv5(P5)
        rfpout=self.backbone(x,[R3,R4,R5])
        F3, F4, F5=rfpout['C1'],rfpout['C2'],rfpout['C3']
        rfp_F=self.fpn([F3,F4,F5])
        W3,W4,W5,W6,W7=rfp_F[0],rfp_F[1],rfp_F[2],rfp_F[3],rfp_F[4]
        add_weight3 = torch.sigmoid(self.rfp_weight(W3))
        add_weight4 = torch.sigmoid(self.rfp_weight(W4))
        add_weight5 = torch.sigmoid(self.rfp_weight(W5))
        add_weight6 = torch.sigmoid(self.rfp2_weight(W6))
        add_weight7 = torch.sigmoid(self.rfp3_weight(W7))
        Z3=add_weight3 * W3 + (1 - add_weight3) * P3
        Z4=add_weight4 * W4 + (1 - add_weight4) * P4
        Z5=add_weight5 * W5 + (1 - add_weight5) * P5
        Z6=add_weight6 * W6 + (1 - add_weight6) * P6
        Z7=add_weight7 * W7 + (1 - add_weight7) * P7
        cls_logits,cnt_logits,reg_preds=self.head([Z3,Z4,Z5,Z6,Z7])
        return [cls_logits,cnt_logits,reg_preds]

class DetectHead(nn.Module):
    def __init__(self,score_threshold,nms_iou_threshold,max_detection_boxes_num,strides,config=None):
        super().__init__()
        self.score_threshold=score_threshold
        self.nms_iou_threshold=nms_iou_threshold
        self.max_detection_boxes_num=max_detection_boxes_num
        self.strides=strides
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
    def forward(self,inputs):
        cls_logits,coords=self._reshape_cat_out(inputs[0],self.strides)
        cnt_logits,_=self._reshape_cat_out(inputs[1],self.strides)
        reg_preds,_=self._reshape_cat_out(inputs[2],self.strides)
        cls_preds=cls_logits.sigmoid_()
        cnt_preds=cnt_logits.sigmoid_()
        coords =coords.cuda() if torch.cuda.is_available() else coords

        cls_scores,cls_classes=torch.max(cls_preds,dim=-1)
        if self.config.add_centerness:
            cls_scores = torch.sqrt(cls_scores*(cnt_preds.squeeze(dim=-1)))
        cls_classes=cls_classes+1
        boxes=self._coords2boxes(coords,reg_preds)
        max_num=min(self.max_detection_boxes_num,cls_scores.shape[-1])
        topk_ind=torch.topk(cls_scores,max_num,dim=-1,largest=True,sorted=True)[1]
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])
            _boxes.append(boxes[batch][topk_ind[batch]])
        cls_scores_topk=torch.stack(_cls_scores,dim=0)
        cls_classes_topk=torch.stack(_cls_classes,dim=0)
        boxes_topk=torch.stack(_boxes,dim=0)
        assert boxes_topk.shape[-1]==4
        return self._post_process([cls_scores_topk,cls_classes_topk,boxes_topk])

    def _post_process(self,preds_topk):
        _cls_scores_post=[]
        _cls_classes_post=[]
        _boxes_post=[]
        cls_scores_topk,cls_classes_topk,boxes_topk=preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask=cls_scores_topk[batch]>=self.score_threshold
            _cls_scores_b=cls_scores_topk[batch][mask]#[?]
            _cls_classes_b=cls_classes_topk[batch][mask]#[?]
            _boxes_b=boxes_topk[batch][mask]#[?,4]
            nms_ind=self.batched_nms(_boxes_b,_cls_scores_b,_cls_classes_b,self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores,classes,boxes= torch.stack(_cls_scores_post,dim=0),torch.stack(_cls_classes_post,dim=0),torch.stack(_boxes_post,dim=0)
        
        return scores,classes,boxes
    
    @staticmethod
    def box_nms(boxes,scores,thr):
        if boxes.shape[0]==0:
            return torch.zeros(0,device=boxes.device).long()
        assert boxes.shape[-1]==4
        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas=(x2-x1+1)*(y2-y1+1)
        order=scores.sort(0,descending=True)[1]
        keep=[]
        while order.numel()>0:
            if order.numel()==1:
                i=order.item()
                keep.append(i)
                break
            else:
                i=order[0].item()
                keep.append(i)
            
            xmin=x1[order[1:]].clamp(min=float(x1[i]))
            ymin=y1[order[1:]].clamp(min=float(y1[i]))
            xmax=x2[order[1:]].clamp(max=float(x2[i]))
            ymax=y2[order[1:]].clamp(max=float(y2[i]))
            inter=(xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
            iou=inter/(areas[i]+areas[order[1:]]-inter)
            idx=(iou<=thr).nonzero().squeeze()
            if idx.numel()==0:
                break
            order=order[idx+1]
        return torch.LongTensor(keep)

    def batched_nms(self,boxes, scores, idxs, iou_threshold):
        
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self,coords,offsets):
        x1y1=coords[None,:,:]-offsets[...,:2]
        x2y2=coords[None,:,:]+offsets[...,2:]
        boxes=torch.cat([x1y1,x2y2],dim=-1)
        return boxes


    def _reshape_cat_out(self,inputs,strides):
        batch_size=inputs[0].shape[0]
        c=inputs[0].shape[1]
        out=[]
        coords=[]
        for pred,stride in zip(inputs,strides):
            pred=pred.permute(0,2,3,1)
            coord=coords_fmap2orig(pred,stride).to(device=pred.device)
            pred=torch.reshape(pred,[batch_size,-1,c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out,dim=1),torch.cat(coords,dim=0)

class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes

        
class FCOSDetector(nn.Module):
    def __init__(self,mode="training",config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        self.mode=mode
        self.fcos_body=FCOS(config=config)
        if mode=="training":
            self.target_layer=GenTargets(strides=config.strides,limit_range=config.limit_range)
            self.loss_layer=LOSS()
        elif mode=="inference":
            self.detection_head=DetectHead(config.score_threshold,config.nms_iou_threshold,
                                            config.max_detection_boxes_num,config.strides,config)
            self.clip_boxes=ClipBoxes()
        
    
    def forward(self,inputs):
        if self.mode=="training":
            batch_imgs,batch_boxes,batch_classes=inputs
            out=self.fcos_body(batch_imgs)
            targets=self.target_layer([out,batch_boxes,batch_classes])
            losses=self.loss_layer([out,targets])
            return losses
        elif self.mode=="inference":
            batch_imgs=inputs
            out=self.fcos_body(batch_imgs)
            scores,classes,boxes=self.detection_head(out)
            boxes=self.clip_boxes(batch_imgs,boxes)
            return scores,classes,boxes



    


