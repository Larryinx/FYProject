from transformers import pipeline
import cv2
import os
import argparse
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
import torchvision
#nms per class!

def inverse_sigmoid(scores):
    return torch.log(scores/(torch.ones_like(scores,dtype=torch.float)-scores))

class FrameProcessor:
    def __init__(self) -> None:
        # checkpoint = "google/owlvit-base-patch32"
        checkpoint = "google/owlv2-base-patch16-ensemble"
        self.model = Owlv2ForObjectDetection.from_pretrained(checkpoint)
        self.processor = Owlv2Processor.from_pretrained(checkpoint)

    def process_image(self, image, text_queries, device='cpu'):
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            target_sizes = [(image.shape[0], image.shape[1])]
            outputs.logits = outputs.logits.to('cpu')
            outputs.pred_boxes = outputs.pred_boxes.to('cpu')
            raw_results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.05)
            assert(len(raw_results) == 1)

        keep_indexs = torchvision.ops.batched_nms(raw_results[0]["boxes"], raw_results[0]["scores"], raw_results[0]['labels'], 0.3)

        # Sort the indices based on scores
        sorted_idxs = torch.argsort(raw_results[0]["scores"][keep_indexs], descending=True)

        # Ensure that only valid indices are used
        valid_sorted_idxs = sorted_idxs[sorted_idxs < len(keep_indexs)]

        # Select up to the top 5 indices, but not more than the number of available results
        top_indices = valid_sorted_idxs[:min(len(valid_sorted_idxs), 5)]
        selected_keep_indexs = keep_indexs[top_indices]

        results = {
            'scores': raw_results[0]["scores"][selected_keep_indexs].tolist(),
            'labels': raw_results[0]["labels"][selected_keep_indexs].tolist(),
            'logits': inverse_sigmoid(raw_results[0]["scores"][selected_keep_indexs]).tolist(),
            'boxes': raw_results[0]["boxes"][selected_keep_indexs].tolist()
        }
        return results



    def image_query(self,image,image_query,device='cpu'):
        #This processor will resize both frame and qury image to 768*768
        #We do not want this since it will squash image of a person to a square
        #However, if we set do_resize=False,do_center_crop=False, we need to change model config as well
        inputs = self.processor(images=image, query_images=image_query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.image_guided_detection(**inputs)
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = [[image.shape[0],image.shape[1]]]
        target_sizes = torch.Tensor(target_sizes)
        # Convert outputs (bounding boxes and class logits) to COCO API
        #print(outputs)
        raw_results = self.processor.post_process_image_guided_detection(
            outputs=outputs, threshold=0.8, nms_threshold=0.3, target_sizes=target_sizes
        )
        assert(len(raw_results)==1)
        #no label for image query
        results={
            'scores':raw_results[0]["scores"].tolist(),
            'labels':torch.zeros_like(raw_results[0]["scores"],dtype=torch.int).tolist(),
            'boxes':raw_results[0]["boxes"].tolist(),
            'logits':inverse_sigmoid(raw_results[0]["scores"]).tolist(),
        }
        return results

def visualize_results(image,result,classes,top_left_caption):
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    #different color for different bounding box
    color_step=255//len(classes)
    colors=[(255-i*color_step,0,i*color_step) for i in range(1,len(classes)+1)]
    cv2.putText(image, top_left_caption,(5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    for index,label in enumerate(result['labels']):
        point=result['boxes'][index]
        p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
        #print(f"P1： ({p1_x},{p1_y}) P2: ({p2_x},{p2_y})")
        cv2.rectangle(img=image,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=colors[label],thickness=3)
        score_str="{:.2f}".format(result['scores'][index])
        cv2.putText(image, f"{label+1} ({score_str})",(p1_x+5, p1_y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return image

#singleton object
owl_processor = FrameProcessor()