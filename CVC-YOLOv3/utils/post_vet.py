import csv
import os
import shutil
from shutil import copy
import argparse
from tqdm import tqdm

def main(csv_path,image_data_path):
    yellow_folder = './yellow/'
    if os.path.exists(yellow_folder):
        shutil.rmtree(yellow_folder)  # delete output folder
    os.makedirs(yellow_folder)  # make new output folder

    blue_folder = './blue/'
    if os.path.exists(blue_folder):
        shutil.rmtree(blue_folder)  # delete output folder
    os.makedirs(blue_folder)  # make new output folder

    small_orange_folder = './small_orange/'
    if os.path.exists(small_orange_folder):
        shutil.rmtree(small_orange_folder)  # delete output folder
    os.makedirs(small_orange_folder)  # make new output folder

    large_orange_folder = './large_orange/'
    if os.path.exists(large_orange_folder):
        shutil.rmtree(large_orange_folder)  # delete output folder
    os.makedirs(large_orange_folder)  # make new output folder

    inconclusive_folder = './inconclusive/'
    if os.path.exists(inconclusive_folder):
        shutil.rmtree(inconclusive_folder)  # delete output folder
    os.makedirs(inconclusive_folder)  # make new output folder


    with open(csv_path) as csv_file, open('yellow.csv','w') as yellow, open('blue.csv','w') as blue, open('small_orange.csv','w') as small_orange, open('large_orange.csv','w') as large_orange, open('inconclusive.csv','w') as inconclusive:
        yellow_counter = 0
        yellow_array = []
        yellow_label = csv.writer(yellow,lineterminator = '\n')
        blue_counter = 0
        blue_array = []
        blue_label = csv.writer(blue,lineterminator = '\n')
        small_orange_counter = 0
        small_orange_array = []
        small_orange_label = csv.writer(small_orange,lineterminator = '\n')
        large_orange_counter = 0
        large_orange_array = []
        large_orange_label = csv.writer(small_orange,lineterminator = '\n')
        inconclusive_counter = 0
        inconclusive_array = []
        inconclusive_label = csv.writer(inconclusive,lineterminator = '\n')
        
        csv_reader = csv.reader(csv_file)
        for i, row in enumerate(csv_reader):
            if row[2] == '["yellow"]':
                yellow_label.writerow(row)
                yellow_array.append(row[1].split('/')[-1])
                yellow_counter += 1
            if row[2] == '["blue"]':
                blue_label.writerow(row)
                blue_array.append(row[1].split('/')[-1])
                blue_counter += 1
            if row[2] == '["small_orange"]':
                small_orange_label.writerow(row)
                small_orange_array.append(row[1].split('/')[-1])
                small_orange_counter += 1
            if row[2] == '["large_orange"]':
                large_orange_label.writerow(row)
                large_orange_array.append(row[1].split('/')[-1])
                large_orange_counter += 1
            if row[2] == '["inconclusive"]':
                inconclusive_label.writerow(row)
                inconclusive_array.append(row[1].split('/')[-1])
                inconclusive_counter += 1
            # print(files[0])
            # print(yellow_label[0])

    print(f"We have {yellow_counter+blue_counter+small_orange_counter+inconclusive_counter} images in total")
    print(f"Found {yellow_counter} yellow labelled images.")
    print(f"Found {blue_counter} blue labelled images.")
    print(f"Found {small_orange_counter} small orange labelled images.")
    print(f"Found {large_orange_counter} large orange labelled images.")
    print(f"Found {inconclusive_counter} inconclusive labelled images.")

    files = [f for f in os.listdir(image_data_path)]
    for i,row in enumerate(tqdm(files)):
        if row in yellow_array:
            copy(os.path.join(image_data_path,row), os.path.join(yellow_folder, row))
        if row in blue_array:
            copy(os.path.join(image_data_path,row), os.path.join(blue_folder, row))
        if row in small_orange_array:
            copy(os.path.join(image_data_path,row), os.path.join(small_orange_folder, row))
        if row in large_orange_array:
            copy(os.path.join(image_data_path,row), os.path.join(large_orange_folder, row))
        if row in inconclusive_array:
            copy(os.path.join(image_data_path,row), os.path.join(inconclusive_folder, row))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, help="target csv file that contains all the images with label on", default='./data-labels_colour_ConeColor_final.csv')
    parser.add_argument("--image_data_path", type=str, help="folder that stores all the images downloaded from gcp", default='./round0/')
    arg = parser.parse_args()
    
    main(
        csv_path=arg.csv_path,
        image_data_path=arg.image_data_path
        )
