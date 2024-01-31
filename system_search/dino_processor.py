from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
BOX_TRESHOLD = 0.4
TEXT_TRESHOLD = 0.25
import torchvision
import torchvision.transforms.transforms as T
import cv2
import numpy as np
from PIL import Image
#TODO: extract a parent class
class DinoProcessor:
    def __init__(self,box_thresh=BOX_TRESHOLD,text_thresh=TEXT_TRESHOLD) -> None:
        config_file="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        checkpoint = "./GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.model = load_model(config_file, checkpoint)
        self.box_thresh=box_thresh
        self.text_thresh=text_thresh

    #TODO: support CPU
    def process_image(self,image,text_queries:list,visualize=False):
        assert(len(text_queries)==1)
        text_queries=text_queries[0]
        def load_frame_np(frame):
            transform = T.Compose(
                [
                    T.Resize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            # image_source = Image.open(frame).convert("RGB")
            image = np.asarray(frame)
            frame_image = Image.fromarray(frame)
            image_transformed = transform(frame_image)
            return image, image_transformed
        if isinstance(image,str):
            image_source, image = load_image(image)
        else:
            image_source,image=load_frame_np(image)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_queries,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh
        )
        h, w, _ = image_source.shape
        boxes_scaled = boxes * torch.Tensor([w, h, w, h])
        xyxy = torchvision.ops.box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        results={
            'scores':logits.tolist(),
            'labels':torch.zeros_like(logits,dtype=torch.int32).tolist(),
            'logits':logits.tolist(),
            'boxes':xyxy.tolist()
        }
        print(results)

        log_file = 'results.log'

        if visualize:
            #annotate methods does RGB to BGR
            # print("annotated images are going to log")
            # save_results_to_log(results, log_file)
            # save_annotated_images(image_source, boxes, logits, phrases, './image_result')
            return results,annotate(image_source,boxes,logits,phrases)

        else:
            return results,None



    def image_query(self,image,image_query):
        raise NotImplementedError

dino_processor = DinoProcessor()

if __name__=="__main__":
    dino_processor=DinoProcessor()
    result,annotated_frame=dino_processor.process_image("misc/frame_550.jpg",["A man with two suitcases ."],visualize=True)
    cv2.imwrite("dino_example.jpg",annotated_frame)