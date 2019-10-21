import pandas as pd 
import argparse
import csv
import sys
import os
sys.path.insert(1, os.path.realpath(os.path.pardir+'/vectorized_yolov3/utils'))
import storage_client

def download_file(file_uri):
    os_filepath = storage_client.get_file(file_uri)
    if not os.path.isfile(os_filepath):
        raise Exception("could not download file: {file_uri}".format(file_uri=file_uri))
    return os_filepath

def main(csv_uri,apple_uri,output_uri):

    csv_filepath = download_file(csv_uri)

    csv_file_name = csv_uri.split('/')[-1].split('.')[0]

    tmp_csv_path = csv_file_name + "_filtered" + ".csv" 

    apple_list = []

    with open(apple_uri,'r') as apple_file:
        apple = csv.reader(apple_file)
        for row in apple:
            apple_list.append(row[0])

    print(str(len(apple_list))+" good apples provided")

    total_list = []

    with open(csv_filepath) as csv_file:
        data = csv.reader(csv_file)
        for i,row in enumerate(data):
            if i < 1:
                continue
            total_list.append(row)

    print(str(len(total_list))+" total data provided")

    counter = 0

    with open(tmp_csv_path,'w') as file_out:
        writer = csv.writer(file_out,lineterminator = '\n')
        first_row = ['', '', 'top', 'mid_R_top', 'mid_R_bot','bot_R', 'bot_L', 'mid_L_bot', 'mid_L_top', '\n']
        writer.writerow(first_row)
        for i in apple_list:
            for j in total_list:
                if i == j[0]:
                    counter += 1
                    writer.writerow(j)

    print(str(counter) + " good apples matched in total dataset")

    storage_client.upload_file(tmp_csv_path, output_uri + tmp_csv_path)
    os.remove(tmp_csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_uri", help="keypoints label csv file that needs to be filtered", default = 'gs://mit-dut-driverless-internal/data-labels/keypoints/Round01_revised_parsed.csv')
    parser.add_argument("--apple_uri", help="csv that contains all the good labeled image's names", required=True)
    parser.add_argument("--output_uri", type=str, help="Folder name to upload the parsed csv", default = 'gs://mit-dut-driverless-internal/data-labels/keypoints/')
    opt = parser.parse_args()
    
    main(csv_uri=opt.csv_uri, apple_uri=opt.apple_uri, output_uri=opt.output_uri)
