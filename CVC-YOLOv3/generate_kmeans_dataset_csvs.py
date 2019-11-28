import os
import sys
import csv
import tempfile
import argparse
import cv2
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

visualization_tmp_path = "outputs/visualization/"

def assignment(boxes, centroids):
    for i in centroids:
        boxes['distance_from_{}'.format(i)] = (np.sqrt((boxes['h'] - centroids[i][0])**2 + (boxes['w'] - centroids[i][1])**2))
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    boxes['closest'] = boxes.loc[:, centroid_distance_cols].idxmin(axis=1)
    boxes['closest'] = boxes['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    return boxes   
    
def update(boxes, centroids):
    for i in centroids:
        centroids[i][0] = np.mean(boxes[boxes['closest'] == i]['h'])
        centroids[i][1] = np.mean(boxes[boxes['closest'] == i]['w'])
    return centroids

def main(csv_uri,dataset_path,output_path,num_clst,max_cone,min_cone,if_plot,split_up):
    box_dict = {} #dictionary with key=tuple of image size, value=list of bounding boxes in image of that size
    img_w = 0
    img_h = 0
    updated_rows = []
    final_rows = []
    in_csv_tempfile = csv_uri
    length = 0

    ##### getting csv length for progress bar #####
    with open(in_csv_tempfile) as lines:
        next(lines) #skip first line
        lines = [line for line in lines]
    length = len(lines)
    #############################


    with open(in_csv_tempfile) as f:
        next(f) #skip first line
        csv_reader = csv.reader(f)

        print("getting images' width and height")
        for i, row in enumerate(tqdm(csv_reader,total=length,desc='Reading Images')): 
            if i < 1:
                continue           
            ##### getting image width and height #####
            img_path = os.path.join(dataset_path,row[0])
            if not os.path.isfile(img_path):
                raise Exception("could not find image: {image_uri}".format(image_uri=os.path.join(dataset_path,row[0])))
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape            
            #############################

            ##### writing updated rows #####
            begin_part = row[:2]
            end_part = row[2:]
            begin_part.append(img_w)
            begin_part.append(img_h)
            rows = begin_part + end_part
            updated_rows.append(rows)
            #############################

            ##### preparing box dictionary for k-means #####
            h = int(row[2])
            w = int(row[3])
            box_dict[(img_h,img_w)] = box_dict.get((img_h, img_w), []) + [(h,w)]
            #############################

    ##### plot original #####
    if if_plot:
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(box_dict)))
        i=0
        fig = plt.figure()
        labels = []
        for key in box_dict:
            labels.append(key)
            h = [points[0] for points in box_dict[key]]
            w = [points[1] for points in box_dict[key]]
            plt.scatter(w, h, color=colors[i])
            i+=1
        fig.suptitle('Original Sizes', fontsize=20)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((0,450,0,450))
        plt.xlabel('Width', fontsize=18)
        plt.ylabel('Height', fontsize=16)
        plt.legend(labels)
        fig.savefig(os.path.join(visualization_tmp_path,'original_boxes.png'))

    #############################

    ##### calculating scale #####
    max_sizes = {}
    min_sizes = {}
    for h,w in box_dict:
        boxes = sorted(box_dict[(h,w)], key=lambda x: x[0])
        max_sizes[(h,w)] = boxes[int(.95*len(boxes))-1]
        min_sizes[(h,w)] = boxes[int(0.05*(len(boxes)))]

    scaled_heights = []
    scaled_widths = []

    i=0
    scaled_plot = {}
    scale_dict = {}
    for h,w in box_dict:
        plot_heights = []
        plot_widths = []
        max_h, max_w = max_sizes[(h,w)]
        min_h, min_w = min_sizes[(h,w)]

        h_ratio = (max_cone-min_cone)/(max_h-min_h)
        print("{height}x{width} images are scaled by {scale}".format(height=h, width=w, scale=h_ratio))
        scale_dict[(h,w)] = scale_dict.get((h, w), 0) + h_ratio

        for box_h, box_w in box_dict[(h,w)]:
            scaled_heights.append((box_h-min_h)*h_ratio + min_cone)
            scaled_widths.append((box_w-min_w)*h_ratio + min_cone)
            if if_plot:
                plot_heights.append((box_h-min_h)*h_ratio + min_cone)
                plot_widths.append((box_w-min_w)*h_ratio + min_cone)
        if if_plot: 
            scaled_plot[i] = [plot_widths, plot_heights]
            i += 1

    scaled_boxes = pd.DataFrame({'h': scaled_heights, 'w': scaled_widths})
    #############################

    ##### calculating k-means #####

    centroids = {} 
    for i in range(num_clst): #start with random boxes as centroids
        rand_index = np.random.randint(0, scaled_boxes.shape[0])
        centroids[i] = [scaled_boxes['h'][rand_index], scaled_boxes['w'][rand_index]]
    scaled_boxes = assignment(scaled_boxes, centroids)
    while True:
        closest_centroids = scaled_boxes['closest'].copy(deep=True)
        centroids = update(scaled_boxes, centroids)
        scaled_boxes = assignment(scaled_boxes, centroids)
        if closest_centroids.equals(scaled_boxes['closest']):
            break
    #############################

    ##### plot afterwards #####
    if if_plot:
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(scaled_plot)))
        #plot scaled height
        figure = plt.figure()
        for i in scaled_plot:
            plt.scatter(scaled_plot[i][0], scaled_plot[i][1], color=colors[i])
        figure.suptitle('Scaled Sizes', fontsize=20)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((0,450,0,450))
        plt.legend(labels)
        plt.xlabel('Width', fontsize=18)
        plt.ylabel('Height', fontsize=16)
        figure.savefig('scaled_boxes.png')
        #############################

        #plot centroids on top
        h = []
        w = []
        for key in centroids:
            h.append(centroids[key][0])
            w.append(centroids[key][1])
        plt.scatter(w,h, color='k')
        figure.suptitle('Centroids and Scaled Boxes', fontsize=20)
        figure.savefig(os.path.join(visualization_tmp_path,'centroids_scaled.png'))
        #############################

        #plot centroids separately
        figure2=plt.figure()
        h = []
        w = []
        for key in centroids:
            h.append(centroids[key][0])
            w.append(centroids[key][1])
        plt.scatter(w,h)
        figure2.suptitle('Centroids', fontsize=20)
        figure2.savefig(os.path.join(visualization_tmp_path,'centroids.png'))
        #############################
    #############################
    
    ##### uploading anchor boxes file #####
    text_file  = open('anchors.txt','w')
    print('Anchors = ', centroids)
    for key in centroids:
        text_file .write('%0.2f,%0.2f \n'%(centroids[key][0], centroids[key][1]))
    text_file.close()
    #############################

    scale = None
    flag_row = None
    with open(in_csv_tempfile) as f:
        next(f) #skip first line
        csv_reader = csv.reader(f)

        print("writing updated rows into csv file")
        for i, row in enumerate(tqdm(updated_rows,desc='Writing Files')):            

            ##### writing updated rows #####
            begin_part = row[:4]
            end_part = row[4:]
            img_w = row[2]
            img_h = row[3]
            begin_part.append(scale_dict[(img_h,img_w)])
            flag_row = begin_part + end_part
            final_rows.append(flag_row)
            #############################

    new_train_uri = os.path.join(output_path, "train.csv")
    train_rows = []
    new_test_uri = os.path.join(output_path, "test.csv")
    test_rows = []
    new_validate_uri = os.path.join(output_path, "validate.csv")
    validate_rows = []
    new_train_validate_uri = os.path.join(output_path, "train-validate.csv")
    train_validate_rows = []
    all_uri = os.path.join(output_path, "all.csv")
    all_rows = []
    empty_imgs = []
    compensate_rows = []

    print("spliting up datasets")
    for i, row in enumerate(tqdm(final_rows,desc='Spliting Datasets')):
        all_rows.append(row)
        remainder = i % 100
        if remainder < int(split_up[0]):
            train_rows.append(row)
            train_validate_rows.append(row)
            continue
        if remainder < int(split_up[0]) + int(split_up[1]):
            validate_rows.append(row)
            train_validate_rows.append(row)
            continue
        test_rows.append(row)
    
    ##############for 0 label images trading##############

    ###getting all 0 labeled images in validation set
    for i,row in enumerate(validate_rows):
        if "" == "".join(row[5:]):
            empty_imgs.append(row)
    #############################


    ###remove all those 0 labeled images in validation set
    for i,row in enumerate(empty_imgs):
        validate_rows.remove(row)
    #############################
    

    ###get compensation from training set
    counter = 0
    for i,row in enumerate(train_rows):
        if not "" == "".join(row[5:]):
            compensate_rows.append(row)
            counter +=1
            if counter == len(empty_imgs):
                break
    #############################

    ###remove compensation from training set
    for i,row in enumerate(compensate_rows):
        train_rows.remove(row)
    #############################

    ###adding 0 labeled images back to training set        
    for i,row in enumerate(empty_imgs):
        train_rows.append(row)
    #############################

    ###add compensation back to validation set
    for i,row in enumerate(compensate_rows):
        validate_rows.append(row)
    #############################

    

    ######################################################

    print(str(len(empty_imgs))+" '0 label images' got traded from validation set to training set.")

    ##### getting anchor values in order #####
    anchors = []
    anchors_prime = ""
    for key in centroids:
        anchors.append([centroids[key][0], centroids[key][1]])
    anchors.sort(key=lambda x: x[0]*x[1])
    for anchor in anchors:
        anchors_prime += str(anchor)[1:-1]
        anchors_prime += "|"
    anchors_prime = anchors_prime[:-1]
    first_row = anchors_prime
    notes = "please see k-means anchor boxes in train.csv"
    #############################

    second_row = ['Name', 'URL', 'Width', 'Height', 'Scale','X0, Y0, H0, W0', 'X1, Y1, H1, W1', 'etc', '\n']

    for (list_rows, list_uri) in ((train_rows, new_train_uri), (test_rows, new_test_uri),
                                  (validate_rows, new_validate_uri), (train_validate_rows, new_train_validate_uri),
                                  (all_rows, all_uri)):
        with tempfile.NamedTemporaryFile() as out_csv_tempfile:
            with open(out_csv_tempfile.name, 'w+') as out_csv_file:
                csv_writer = csv.writer(out_csv_file)
                if list_uri != new_train_uri:
                    csv_writer.writerow([notes])
                else:
                    csv_writer.writerow([first_row])
                csv_writer.writerow(second_row)
                for row in list_rows:
                    csv_writer.writerow(row)
            print("Saving {list_uri} ...")
            os.rename(out_csv_tempfile.name, list_uri)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})

    parser.add_argument("--input_csvs", help="csv file to split", default = 'dataset/all.csv')
    parser.add_argument('--dataset_path', type=str, help='path to image dataset',default="dataset/YOLO_Dataset/")
    parser.add_argument('--output_path', type=str, help='path to output csv files',default="dataset/")
    parser.add_argument('--num_clst', type=int, default=9, help='number of anchor boxes wish to be generated')
    parser.add_argument('--max_cone_height', default = 83, type = int, help='height of maximum sized cone to scale to\n')
    parser.add_argument('--min_cone_height', default = 10, type = int, help='height of minimum sized cone to scale to\n')
    parser.add_argument("--split_up",  type=str, default = '75-15-0', help="train/validate/test split")
    
    add_bool_arg('if_plot', default=True, help='whether to get anchor boxes plotted, plots saved as original_boxes.png, scaled_boxes.png, centroids.png in output uri')

    opt = parser.parse_args()

    split_up = [int(x) for x in opt.split_up.split('-')]

    main(csv_uri=opt.input_csvs,
    dataset_path=opt.dataset_path,
    output_path=opt.output_path,
    num_clst=opt.num_clst,
    max_cone=opt.max_cone_height,
    min_cone=opt.min_cone_height,
    if_plot=opt.if_plot,
    split_up=split_up)
    
