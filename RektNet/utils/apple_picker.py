import os
import csv
import cv2
import shutil
import argparse

def main(target_path,output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  # delete output folder
    os.makedirs(output_path)  # make new output folder

    with open(os.path.join(output_path,'good_label.csv'),'w') as good_file, open(os.path.join(output_path,'bad_label.csv'),'w') as bad_file, open(os.path.join(output_path,'shitty_label.csv'),'w') as shitty_file:
        good_label = csv.writer(good_file,lineterminator = '\n')
        good_counter = 0
        bad_label = csv.writer(bad_file,lineterminator = '\n')
        bad_counter = 0
        shitty_label = csv.writer(shitty_file,lineterminator = '\n')
        shitty_counter = 0
            
        for i, images in enumerate(os.listdir(target_path)):
            if images.endswith("jpg"):
                rawImage = cv2.imread(os.path.join(target_path,images))
                height, width, _ = rawImage.shape
                windowName = "{num}/{ttl}: {img_name}".format(num=i, ttl=len(os.listdir(target_path)),img_name=str(images))
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(windowName, height, width)
                cv2.moveWindow(windowName, 0, 0)
                cv2.imshow(windowName, rawImage)
                key = cv2.waitKey(0)

                cv2.waitKey(1) & 0xFF
                cv2.destroyWindow(windowName)
                cv2.waitKey(1)
                ##### ESC for quit #####
                if key == 27:
                    print("Sorting job early stoopped. {good} good cones, {bad} bad cones, {shitty} shitty cones were sorted.".format(good=good_counter,bad=bad_counter,shitty=shitty_counter))
                    quit()
                ##### key 'j','k','l','u','i','o' for good labels #####
                elif key == 106 or key == 107 or key == 108 or key == 117 or key == 105 or key == 111:
                    good_label.writerow([str(images)])
                    good_counter += 1
                ##### key 'a','s','d','q','w','e' for bad labels #####
                elif key == 97 or key == 115 or key == 100 or key == 113 or key == 199 or key == 101:
                    bad_label.writerow([str(images)])
                    bad_counter += 1
                ##### key 'space', means this label is either gay or totally fucked-up cone labels#####
                elif key == 32:
                    shitty_label.writerow([str(images)])
                    shitty_counter += 1
                ##### to get key id of keyboard. 255 is what the console returns when there is no key press #####
                # elif key != 255:
                #     print("Uknown Key Pressed, please check")
                #     print(key)
                #################################################################################################
        print("Sorting job finished. {good} good cones, {bad} bad cones, {shitty} shitty cones were sorted.".format(good=good_counter,bad=bad_counter,shitty=shitty_counter))
    temp = cv2.imread('logs/cv_ritual.jpg')
    height, width, _ = temp.shape
    windowName = "Sorting job finished. {good} good cones, {bad} bad cones, {shitty} shitty cones were sorted.".format(good=good_counter,bad=bad_counter,shitty=shitty_counter)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, height, width)
    cv2.moveWindow(windowName, 0, 0)
    cv2.imshow(windowName,temp)
    key = cv2.waitKey(0)

    cv2.waitKey(1) & 0xFF
    cv2.destroyWindow(windowName)
    cv2.waitKey(1)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--target_path", type=str, help="target folder that contains all the images with label on", default='./vis_r01/')
	parser.add_argument("--output_path", type=str, help="folder to output the sorted image csv", default='./picked_apple/')
	arg = parser.parse_args()
	
	main(target_path=arg.target_path,output_path=arg.output_path)
