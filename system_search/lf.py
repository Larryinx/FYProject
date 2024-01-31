import cv2
import os
import time
import json
import numpy as np
import datetime
import argparse
import torch
from typing import Union
import matplotlib.pyplot as plt
import csv

from dino_processor import dino_processor
from owl_processor import owl_processor, visualize_results
from outputs import VideoResult


def process_video_owl(
        top_scores,
        total_frame_number,
        video_path:str,
        text_queries=None,
        image_queries=None,
        interval=6,
        result_dir=None,
        max_frame=None,
        result_video=None,
        ):
    if text_queries is None and image_queries is None:
        print('Suply either image query or lang query')
        return
    if text_queries is not None and image_queries is not None:
        print('Suply either image query or lang query')
        return
    if text_queries is not None:
        query_type='lang'
    else:
        query_type='image'
    print(f"Searching with {query_type} Queries: {text_queries if text_queries else image_queries} in video {video_path}")

    device='cuda' if torch.cuda.is_available() else 'cpu'
    all_frame_results=[]
    owl_processor.model.to(device)
    print("Using " + device)

    if query_type=='image':
        # convert BGR to RGB
        image_queries_cv2=[cv2.imread(name) for name in image_queries]
        image_queries_cv2=[cv2.cvtColor(i,cv2.COLOR_BGR2RGB) for i in image_queries_cv2]

    video = cv2.VideoCapture(video_path)

    if result_video is not None:
        #codec must be avc1 (h.264) to allowing playing in <video> element of html
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer=cv2.VideoWriter(result_video,fourcc,round(30.0/interval),(int(video.get(3)),int(video.get(4))))

    if result_dir is not None:
        os.makedirs(f'./results/{result_dir}',exist_ok=True)

    frame_count = 0
    image_folder = 'img_folder'  # Folder containing the images
    image_files = sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort the files numerically

    print(f"Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} Processing interval: {interval}")

    start_process = time.perf_counter()

    with open('results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame', 'Max Score'])

        for frame_count in range(total_frame_number):
            image_file = f"frame_{frame_count}.jpg"
            if image_file in image_files:
                image_path = os.path.join(image_folder, image_file)
                frame = cv2.imread(image_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if max_frame is not None and frame_count >= max_frame:
                break

            if frame_count in top_scores:
                start = time.perf_counter()
                if query_type=='lang':
                    result=owl_processor.process_image(frame,text_queries,device)
                else:
                    result=owl_processor.image_query(frame,image_queries_cv2,device)
                result['frame'] = frame_count
                all_frame_results.append(result)
                end = time.perf_counter()
                print(f"Results of frame {frame_count}: {result['scores']} Time:{end-start}s")

                if query_type=='lang':
                    class_string=", ".join([f"{index+1}->{c}" for index,c in enumerate(text_queries)])
                    format_string=f"Classes: [{class_string}]"
                    visualized_image=visualize_results(frame,result,text_queries,format_string)
                else:
                    format_string=f"Image: {', '.join(image_queries)}"
                    visualized_image=visualize_results(frame,result,image_queries,format_string)
                if result_dir is not None:
                    result_path=f'./results/{result_dir}/frame_{frame_count}.jpg'
                    cv2.imwrite(result_path,visualized_image)
                if result_video is not None:
                    video_writer.write(visualized_image)

                max_score = max(result['scores']) if result['scores'] else 0
                csvwriter.writerow([frame_count, max_score])
            elif frame_count % interval==0:
                all_frame_results.append({'scores':[], 'labels':[], 'logits':[], 'boxes':[], 'frame':frame_count})
            else:
                all_frame_results.append({'frame':frame_count})

    end_process = time.perf_counter()
    print(f"Total Time:{end_process-start_process}s")
    if result_video is not None:
        video_writer.release()
    video.release()
    return all_frame_results

def process_video_gdino(
        top_scores,
        total_frame_number,
        video_path:str,
        text_queries=None,
        image_queries=None,
        interval=6,
        result_dir=None,
        max_frame=None,
        result_video=None,
):
    if text_queries is None and image_queries is None:
        print('Suply either image query or lang query')
        return
    if text_queries is not None and image_queries is not None:
        print('Suply either image query or lang query')
        return
    if text_queries is not None:
        query_type='lang'
    else:
        query_type='image'
    print(f"Searching with {query_type} Queries: {text_queries if text_queries else image_queries} in video {video_path}")
    device='cuda' if torch.cuda.is_available() else 'cpu'
    all_frame_results=[]
    print("Using " + device)
    video = cv2.VideoCapture(video_path)
    if result_video is not None:
        #codec must be avc1 (h.264) to allowing playing in <video> element of html
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(result_video, fourcc, round(30.0/interval), (int(video.get(3)), int(video.get(4))))

    if result_dir is not None:
        os.makedirs(f'./results/{result_dir}',exist_ok=True)

    frame_count = 0
    image_folder = 'img_folder'  # Folder containing the images
    image_files = sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort the files numerically

    start_process = time.perf_counter()

    with open('results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame', 'Max Score'])

        for frame_count in range(total_frame_number):
            image_file = f"frame_{frame_count}.jpg"
            if image_file in image_files:
                image_path = os.path.join(image_folder, image_file)
                frame = cv2.imread(image_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if max_frame is not None and frame_count >= max_frame:
                break

            if frame_count in top_scores:
                start = time.perf_counter()
                if query_type == 'lang':
                    result, visualized_image = dino_processor.process_image(frame, text_queries, visualize=result_video is not None)
                else:
                    result, visualized_image = dino_processor.image_query(frame, image_queries_cv2)

                result['frame'] = frame_count
                all_frame_results.append(result)
                end = time.perf_counter()
                print(f"Results of frame {frame_count}: {result['scores']} Time:{end-start}s")

                if result_dir is not None:
                    result_path = f'./results/{result_dir}/frame_{frame_count}.jpg'
                    cv2.imwrite(result_path, visualized_image)

                if result_video is not None:
                    video_writer.write(visualized_image)

                max_score = max(result['scores']) if result['scores'] else 0
                csvwriter.writerow([frame_count, max_score])
            elif frame_count % interval==0:
                all_frame_results.append({'scores':[], 'labels':[], 'logits':[], 'boxes':[], 'frame':frame_count})
            else:
                all_frame_results.append({'frame':frame_count})

    end_process = time.perf_counter()
    print(f"Total Time:{end_process-start_process}s")

    if result_video is not None:
        video_writer.release()

    return all_frame_results
