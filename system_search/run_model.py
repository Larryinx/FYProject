import os
import datetime
import json
from typing import Union
import argparse

from outputs import VideoResult
from lf import process_video_owl, process_video_gdino
from proxy_model import ProxyAnalyzer
from frame_extract import extract_frames

from config import video_name, detect_object, query_type, interval, extract_precentage, model_name, query

def run_video(
            video_name:str,
            queries:list[str],
            run_type:str,
            model_name:str,
            interval:int=3,
            max_frame:Union[int,None]=None,
            visualize_all:bool=False,
            top_k:Union[int,None]=None,
            chunk_size:Union[int,None]=None,
            ):

    top_scores = {}
    total_frame_number = 0

    print(f"Extracting frames from {video_name} with interval {interval}")
    extract_frames(video_name, interval)

    print(f"Proxy model starts analyzing frames")
    analyzer = ProxyAnalyzer(
        semantic_search_phrase=detect_object,
        frame_rate=interval,
        video_path=video_name,
        top_percentage=extract_precentage
    )

    total_frame_number = analyzer.count_frames(analyzer.video_path)
    print(f"Number of frames in the video: {total_frame_number}")
    analyzer.analyze_frames()
    top_scores = analyzer.top_percentage_scores

    print(f"Using model {model_name}")
    os.makedirs('results',exist_ok=True)
    video_raw_name=video_name.split('/')[-1]
    str_max_frame=''
    if max_frame is not None:
        str_max_frame=f'_frame_{max_frame}'
    #append time stamp into name to avoid duplication
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M")
    if visualize_all:
        result_video_name=f'./results/{video_raw_name}{str_max_frame}_{formatted_time}_{run_type}.mp4'
    else:
        result_video_name=None
    result_json_name=f'./results/{video_raw_name}{str_max_frame}_{formatted_time}_{run_type}.json'
    print(f'Result Video Path: {result_video_name}')
    if run_type not in ['image','lang']:
        print("Invalid run_type: must be image or lang ")
        return
    process_video_method=processor_mapping[model_name]
    if run_type=='image':
        result=process_video_method(video_name,image_queries=queries,interval=interval,result_dir=None,max_frame=max_frame,result_video=result_video_name)
    elif run_type=='lang':
        result=process_video_method(top_scores, total_frame_number, video_name,text_queries=queries,interval=interval,result_dir=None,max_frame=max_frame,result_video=result_video_name)
    result={
            'query':queries,
            'type':run_type,
            'result':result,
        }
    with open(result_json_name,'w') as f:
        print(f"Result Json: {result_json_name}")
        json.dump(result,f)
    if top_k is not None:
        video_result=VideoResult()
        video_result.from_data_dict(result)
        sorted_chunks_ma=video_result.sort_logits_chunks_ma(chunk_size)
        result_dirs=video_result.dump_top_k_chunks(video_name,sorted_chunks_ma,top_k)
        return result_dirs
    else:
        return []

processor_mapping={
    'gdino':process_video_gdino,
    'owl':process_video_owl,
}

if __name__=='__main__':
    # parser=argparse.ArgumentParser()
    # parser.add_argument('--video_name',type=str)
    # parser.add_argument('--query_index',type=int)
    # parser.add_argument('--max_frame',type=int,default=None)
    # parser.add_argument('--interval',type=int,default=10,help="the number of frame between every model execution")
    # parser.add_argument('--visualize_all', action='store_true', default=False,help='visualize all bounding boxes of the video')
    # parser.add_argument('--top_k',type=int,default=None,help="top k chunks to output, if None, no chunk will be output")
    # parser.add_argument('--chunk_size',type=int,default=None,help="Number of frames in a chunk") # 2 seconds
    # parser.add_argument('--model_name',type=str,default=None,choices=['owl-vit','gdino'])
    # args=parser.parse_args()

    # print(f"Number of frames in the video: {analyzer.count_frames(analyzer.video_path)}")
    # analyzer.extract_and_analyze_frames()
    # top_scores = analyzer.top_percentage_scores

    results_dirs=run_video(video_name,query,query_type,model_name,interval,visualize_all=False,top_k=10,chunk_size=90)
    print(f"Results saved to {results_dirs}")