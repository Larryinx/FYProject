import matplotlib.pyplot as plt
import json
import cv2
import os
import torchvision
from PIL import Image
import bisect
import argparse
import matplotlib.pyplot as plt
from utils.fn import max_with_default,square_ma
from utils.utils import file_name,split_suffix

prop_loc={
        'scores':0,
        'logits':1,
        'boxes':2,
        'labels':3,
    }

class FrameResult:

    def __init__(self) -> None:
        #data: [score, logit,bbox,label] if query is image then all labels are 0.
        #bbox: []
        self.data=None
        self.frame_index=None


    def from_data_dict(self,data):
        self.data=[]
        self.frame_index=data['frame']
        if 'scores' in data:
            for i,_ in enumerate(data['scores']):
                self.data.append([data['scores'][i],data['logits'][i],data['boxes'][i],data['labels'][i]])
        else:
            self.data=None

    def to_dict(self):
        raise NotImplemented

    def get_prop(self,prop:str):
        return [record[prop_loc[prop]] for record in self.data]

    def get_result_by_label(self,label:int):
        return [record for record in self.data if record[3]==label]

    #the original video is 30fps and we are processing at 10fps
    #len(data) wouldn't work, because there are processed frames with no bounding box
    def skipped(self):
        return self.data==None


    def nms(self,thresh):
        pass

    def remove_zero_box(self):
        pass


class VideoResult:

    def __init__(self) -> None:
        self.frame_results:list[FrameResult]=[]
        self.labels:list[str]=[]
        self.query_type=None

    #create the object based on the json generated by lf.py
    def from_data_dict(self,data):
        self.query_type = data['type']
        self.labels = data['query']
        for frame_result in data['result']:
            new_frame_result=FrameResult()
            new_frame_result.from_data_dict(frame_result)
            self.frame_results.append(new_frame_result)


    #return a dict: label to frame number sorted by logit field
    #only includes non empty frames
    def get_bbox_logits(self)->dict[str,list[int]]:
        results={}
        non_empty_frames=[fr.frame_index for fr in self.frame_results if not fr.skipped()]
        for label in self.labels:
            results[label]=[]
        for f in non_empty_frames:
            f_labels=self.frame_results[f].get_prop('labels')
            f_logits=self.frame_results[f].get_prop('logits')
            for index,l in enumerate(f_labels):
                results[self.labels[l]].append(f_logits[index])
        return results

        raise NotImplementedError

    #return top k frames of each label in self.labels;
    #use the max logit of each frame as the score of the frame
    def sort_logits_frame_max(self,top_k=None,default=-100)->dict[str,list[int]]:
        max_scorer=max_with_default(default)
        return self.sort_logits_frame(max_scorer,top_k)
    #return top k frames of each label in self.labels


    def get_frame_scores(self,frame_scorer):
        frame_scores={k:[] for k in self.labels}
        non_empty_frames=[fr.frame_index for fr in self.frame_results if not fr.skipped()]
        print(len(non_empty_frames))
        for label in self.labels:
            for frame in non_empty_frames:
                frame_logits=self.frame_results[frame].get_prop('logits')
                frame_labels=self.frame_results[frame].get_prop('labels')
                same_label_logits=[frame_logits[i] for i,l in enumerate(frame_labels) if self.labels[l]==label]
                frame_scores[label].append(frame_scorer(same_label_logits))
        print({k:len(frame_scores[k]) for k in frame_scores})
        return frame_scores


    #sort all the frame based on its score: the score of a frame
    #is calculated using frame_scoror, which takes in the logits of all the bbox and output a scalar score
    def sort_logits_frame(self,frame_scorer,top_k=None)->dict[str,list[int]]:
        sorted_frames={}
        non_empty_frames=[fr.frame_index for fr in self.frame_results if not fr.skipped()]
        for label in self.labels:
            def key_func(x,label):
                frame_logits=self.frame_results[x].get_prop('logits')
                frame_labels=self.frame_results[x].get_prop('labels')
                same_label_logits=[frame_logits[i] for i,l in enumerate(frame_labels) if self.labels[l]==label]
                return frame_scorer(same_label_logits)
            sorted_frames[label]=sorted(non_empty_frames,key=lambda x:key_func(x,label),reverse=True)
        if top_k is not None:
            for k in sorted_frames:
                sorted_frames[k]=sorted_frames[k][:top_k]
        return sorted_frames

    def get_props(self,prop):
        result={}
        for k in self.labels:
            result[k]=[r.get_prop(prop) for r in self.frame_results if not r.skipped()]
        return result

    #moving average
    def sort_logits_chunks_sqr_ma(self,chunk_len):
        #the default value should be set to the minimum value of frame scores
        min_value=self.box_logits_min()
        max_scorer=max_with_default(min_value)
        sqr_ma=square_ma(min_value)
        return self.sort_logits_chunks_partitioned(max_scorer,sqr_ma,chunk_len)

    def sort_logits_chunks_ma(self,chunk_len):
        #the default value should be set to the minimum value of frame scores
        min_value=self.box_logits_min()
        max_scorer=max_with_default(min_value)
        def ma(scores):
            return sum(scores)/len(scores)
        return self.sort_logits_chunks_partitioned(max_scorer,ma,chunk_len)


    #sliding window: slide over n-chunk_len chunks (might have overlapping chunks)
    def sort_logits_chunks_slide_window(self,frame_scorer,chunk_scorer,chunk_len)->dict[str,list[tuple[int,int]]]:
        intervals={}
        non_empty_frames=[fr.frame_index for fr in self.frame_results if not fr.skipped()]
        for label in self.labels:
            print(self.get_props('logits'))
            logits=self.get_props('logits')[label]
            assert(len(logits)==len(non_empty_frames))
            frame_scorers=[frame_scorer(frame_logits) for frame_logits in logits]
            chunk_scores=[]
            for i in range(len(non_empty_frames)-chunk_len+1):
                chunk_scores.append((non_empty_frames[i],chunk_scorer(frame_scorers[i:i+chunk_len])))
            chunk_scores.sort(reverse=True,key=lambda a:a[1])
            #can be overlapping chunks
            interval=[(c[0],c[0]+chunk_len) for c in chunk_scores]
            intervals[label]=interval
        return intervals

    #everest implementation: partition n frames into n/chunk_len chunks
    def sort_logits_chunks_partitioned(self,frame_scorer,chunk_scorer,chunk_len:int)->dict[str,list[tuple[int,int]]]:
        intervals={}
        non_empty_frames=self.non_skipped_frames()
        for label in self.labels:
            logits=self.get_props('logits')[label]
            assert(len(logits)==len(non_empty_frames))
            frame_scores=[frame_scorer(frame_logits) for frame_logits in logits]
            chunk_scores=[]
            for i in range(len(self.frame_results)//chunk_len):
                partition_start=i*chunk_len
                partition_end=(i+1)*chunk_len
                #TODO: now using binary search; change to sliding window
                start=bisect.bisect_right(non_empty_frames,partition_start)
                end=bisect.bisect_left(non_empty_frames,partition_end)
                #print(partition_start,non_empty_frames[start],non_empty_frames[end],partition_end)
                #here, we require start to be the minimun value, end to be the maximum value such that partition_start<=start<end<=partition_end
                score_of_chunk=chunk_scorer(frame_scores[start:end])
                chunk_start=i*chunk_len
                chunk_scores.append((chunk_start,score_of_chunk))
            chunk_scores.sort(reverse=True,key=lambda a:a[1])
            interval=[(c[0],c[0]+chunk_len) for c in chunk_scores]
            intervals[label]=interval
        return intervals

    def get_frame_result(self,index):
        return self.frame_results[index]

    #apply a reducer function to all the logits of the boxes
    # The reducer should be (float,float)->float
    def box_logits_min(self):
        result=float('+inf')
        for frame_result in self.frame_results:
            if not frame_result.skipped():
                result=min([result,*frame_result.get_prop('logits')])
        return result

    #write the top k frames and return the result dir
    def dump_top_k_frames(self,top_k,video_path:str):
        video_name=file_name(video_path)
        top_k_frames=self.sort_logits_frame_max(top_k)
        final_result_dir=[]
        for label in top_k_frames:
            connected_name=label.replace(' ','_')
            image_name=file_name(connected_name)
            raw_name,suffix=split_suffix(image_name)
            video=cv2.VideoCapture(video_path)
            result_save_path=f'./results/{video_name}_{raw_name}'
            os.makedirs(result_save_path,exist_ok=True)
            print(f'Writing {len(top_k_frames[label])} image for {label}')
            print(f"Top-{top_k} frames saved to: {result_save_path}")
            for top_frame_index,current_index in enumerate(top_k_frames[label]):
                #TODO: write the frames in single pass
                #random access in video is actually slow
                video.set(cv2.CAP_PROP_POS_FRAMES, current_index)
                ret, frame = video.read()
                if not ret:
                    print(f"Error: Unable to read frame {current_index}")
                    continue
                write_path=f'./results/{video_name}_{raw_name}/{top_frame_index}.jpg'
                label_location=self.labels.index(label)
                box_filtered=self.get_frame_result(current_index).get_result_by_label(label_location)
                try:
                    max_key_location,_=max(enumerate(box_filtered), key=(lambda x: x[1]))
                except ValueError:
                    print(f"Warning: This object appeared in {top_frame_index} frames, which is less than {top_k}.")
                    break
                print(f"Max box location: {max_key_location}")
                for i,box in enumerate(box_filtered):
                    point=box[2]
                    p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
                    if i==max_key_location:
                        print(f"Max box: {box}")
                        color=(0,215,255)
                    else:
                        color=(0,0,255)
                    cv2.rectangle(img=frame,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=color,thickness=3)
                    cv2.putText(frame, "{:.2f}".format(box[1]),(p1_x+5, p1_y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame, f"Frame {current_index} Rank {top_frame_index}",(25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                cv2.imwrite(write_path,frame)
            video.release()
            final_result_dir.append(result_save_path)
        return final_result_dir


    def non_skipped_frames(self):
        return [f.frame_index for f in self.frame_results if not f.skipped() ]

    def plot_frame_scores(self,non_empty_frame_index,frame_scores,save_path,label_connected,intervals):
        plt.figure(figsize=(12,4))
        plt.ylabel("Frame Score")
        plt.xlabel("Frame Index")
        image_file_name=file_name(label_connected)
        image_name_without_suffix,suffix=split_suffix(image_file_name)
        plt.title(f"Frame Scores of {label_connected}")
        lim=[min(frame_scores)-1,max(frame_scores)+1]
        plt.ylim(lim[0],lim[1])
        #red mightest
        fill_between_handels=[]
        for index,interval in enumerate(intervals):
            h=plt.fill_between([i for i in range(interval[0],interval[1])],lim[0],lim[1], color=(1-(index+1)/len(intervals),(index+1)/len(intervals),0), alpha=0.5)
            fill_between_handels.append(h)
        plt.legend(handles=fill_between_handels, labels=[f"{i}" for i,_ in enumerate(fill_between_handels)])
        # Fill between the intervals
        plt.plot(non_empty_frame_index,frame_scores,'-',linewidth=1)
        fig_path=f"{save_path}/{image_name_without_suffix}_frame_scores.jpg"
        print(f"Figure saved to {fig_path}")
        plt.savefig(fig_path)

    def dump_top_k_chunks(self, video_path, sorted_results, top_k:int):
        print(f'video_path={video_path}')
        print(f'sorted_results={sorted_results}')
        video_name=file_name(video_path)
        final_result_dir=[]
        max_scorer=max_with_default(self.box_logits_min())
        frame_results=self.get_frame_scores(max_scorer)
        for label in sorted_results:
            connected_name=label.replace(' ','_')
            chunk_name=file_name(connected_name)
            raw_name,suffix=split_suffix(chunk_name)
            result_save_path=f'./results/{video_name.split(".")[0]}_{raw_name}'
            os.makedirs(result_save_path,exist_ok=True)
            print(f"Top-{top_k} chunks will be saved in: {result_save_path}")
            # video = cv2.VideoCapture(f'utils/videos/{video_path}')
            video = cv2.VideoCapture(f'{video_path}')
            # print(os.path.isfile(f'utils/videos/{video_path}'))
            print(os.path.isfile(f'{video_path}'))
            non_empty_frame=self.non_skipped_frames()
            if len(sorted_results[label])<top_k:
                print(f"Only {len(sorted_results[label])} for label {label}. Topk : {top_k}")
            top_k_chunks=sorted_results[label][:top_k]
            # self.plot_frame_scores(non_empty_frame,frame_results[label],result_save_path,connected_name,top_k_chunks)
            for i, chunk in enumerate(top_k_chunks):
                start, end=chunk
                #The frames are saved in memory
                edited_frames = []
                video.set(cv2.CAP_PROP_POS_FRAMES, start)
                ret, frame = video.read()
                # print(frame)
                # return
                for frame_index in range(start, end+1):
                    ret, frame = video.read()
                    #skip unprocessed frame
                    if self.frame_results[frame_index].skipped():
                        edited_frames.append(frame)
                        continue
                    if not ret:
                        print(f"Error: Unable to read frame {frame_index}")
                        continue
                    label_location=self.labels.index(label)
                    box_filtered=self.get_frame_result(frame_index).get_result_by_label(label_location)
                    try:
                        max_key_location,_=max(enumerate(box_filtered), key=(lambda x: x[1]))
                    except ValueError:
                        print(f"Warning: This object appeared in {i} frames, which is less than {top_k}.")
                        edited_frames.append(frame)
                        continue
                    max_key_location,_=max(enumerate(box_filtered), key=(lambda x: x[1]))
                    print(f"Max box location: {max_key_location}")
                    for j, box in enumerate(box_filtered):
                        point=box[2]
                        p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
                        if j==max_key_location:
                            print(f"Max box: {box}")
                            color=(0,215,255)
                        else:
                            color=(0,0,255)
                        cv2.rectangle(img=frame,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=color,thickness=3)
                        cv2.putText(frame, "{:.2f}".format(box[1]),(p1_x+5, p1_y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    edited_frames.append(frame)
                # print(edited_frames)
                fourcc = cv2.VideoWriter_fourcc(*'VP80')
                chunk_name = f'{result_save_path}/rank{i}.webm'
                chunk_writer = cv2.VideoWriter(chunk_name, fourcc, 30, (edited_frames[0].shape[1], edited_frames[0].shape[0]))
                for frame in edited_frames:
                    chunk_writer.write(frame)
                print(f'Chunk {i} saved to {chunk_name}')
                chunk_writer.release()
            video.release()
            final_result_dir.append(result_save_path)
        return final_result_dir

#plot the histogram for logit distribution
def visualize_logits(label_to_logits):
    for k in label_to_logits:
        print(f"bboxs count of {k}: {len(label_to_logits[k])}")
        plt.hist(label_to_logits[k], bins=50, color='b', edgecolor='black')
        plt.title(f'Scores Distribution of {len(label_to_logits[k])} {k}(s) ')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        hist_name=f"{k}_hist.jpg"
        print(f"Creating histogram {hist_name}")
        plt.savefig(hist_name)
        plt.clf()

def make_grid(image_dir,row_size=4):
    transform =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize(300)])
    dir_name=image_dir.split('/')[-1]
    images=[ transform(Image.open(os.path.join(image_dir,f'{i}.jpg'))) for i in range(row_size*row_size)]
    grid = torchvision.utils.make_grid(images, nrow=row_size,pad_value=4)
    torchvision.transforms.ToPILImage()(grid).save(f'{image_dir}/summary_{dir_name}_{row_size}x{row_size}.jpg')

if __name__=='__main__':
    #lang query
    #change to the name of the json output by lf.py
    parser=argparse.ArgumentParser()
    parser.add_argument('--video_results',type=str,required=True)
    parser.add_argument('--video_name',type=str,required=True)
    parser.add_argument('--dump_type',type=str,choices=['frame','chunk'],help='frame or chunk',required=True)
    parser.add_argument('--top_k',type=str,default=10)
    parser.add_argument('--chunk_size',type=str,default=90) # 3 seconds
    args=parser.parse_args()
    video_results=VideoResult()
    with open(args.video_results) as f:
        json_data=json.load(f)
        video_results.from_data_dict(json_data)
    top_k=int(args.top_k)
    print(f"Video results loaded Sorting top-{top_k}")
    if args.dump_type=='frame':
        save_dir=video_results.dump_top_k_frames(top_k,args.video_name)
        for dir in save_dir:
            make_grid(dir)
    elif args.dump_type=='chunk':
        sorted_result=video_results.sort_logits_chunks_ma(int(args.chunk_size))
        for k in sorted_result:
            sorted_result[k]=sorted_result[k]
        print(sorted_result)
        save_dir=video_results.dump_top_k_chunks(args.video_name,sorted_result,top_k)
    else:
        print("Dump result can only be chunk or frame")
    

    