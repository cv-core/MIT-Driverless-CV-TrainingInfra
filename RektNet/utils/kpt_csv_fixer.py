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

def main(csv_uri,output_uri):

	csv_filepath = download_file(csv_uri)

	csv_file_name = csv_uri.split('/')[-1].split('.')[0]

	tmp_csv_path = csv_file_name + "_parsed" + ".csv" 

	with open(csv_filepath) as csv_file, open(tmp_csv_path,'w') as file_out:
		data = csv.reader(csv_file)
		writer = csv.writer(file_out,lineterminator = '\n')
		c = 0
		first_row = ['', '', 'top', 'mid_R_top', 'mid_R_bot','bot_R', 'bot_L', 'mid_L_bot', 'mid_L_top', '\n']
		writer.writerow(first_row)
		for i in data:
			if '' not in i and '[0, 0]' not in i:
				top = None
				mid_L_top = None
				mid_L_bot = None
				mid_R_top = None
				mid_R_bot = None
				bot_L = None
				bot_R = None

				flagger = None

				flag_list = i[2:]

				y_loc = []
				x_loc = []
				valid_x = []
				valid_y = []
				for j in range(len(flag_list)):
					flag_list[j] = flag_list[j][1:-1].split(', ')
					x_loc.append(int(flag_list[j][0]))
					y_loc.append(int(flag_list[j][1]))
				if i[0] == flagger:
					print(flag_list)

				#######getting top position########
				for k in flag_list:
					if min(y_loc) == int(k[1]):
						top = [int(k[0]),int(k[1])]
						flag_list.remove(k)
				if i[0] == flagger:
					print("the top location is: " + str(top))

				#######getting bot_L position########
				temp = []
				x_loc.sort()
				for k in flag_list:
					if x_loc[0] == int(k[0]) or x_loc[1] == int(k[0]) or x_loc[2] == int(k[0]) or x_loc[3] == int(k[0]):
						valid_x.append(k)
						temp.append(int(k[1]))
				for q in valid_x:
					if str(max(temp)) == q[1]:
						bot_L = [int(q[0]),int(q[1])]
						flag_list.remove(q)
				
				valid_x = []
				if i[0] == flagger:
					print("the bot_L location is: " + str(bot_L))

				#######getting bot_R position########
				temp = []
				x_loc.sort()
				for k in flag_list:
					if x_loc[-1] == int(k[0]) or x_loc[-2] == int(k[0]) or x_loc[-3] == int(k[0]) or x_loc[-4] == int(k[0]):
						valid_x.append(k)
						temp.append(int(k[1]))
				for q in valid_x:
					if str(max(temp)) == q[1]:
						bot_R = [int(q[0]),int(q[1])]
						flag_list.remove(q)
									
				valid_x = []
				if i[0] == flagger:
					print("the bot_R location is: " + str(bot_R))

				arena = []
				for w in flag_list:
					arena.append(int(w[0])**2 + int(w[1])**2)

				#######getting mid_L_top position########
				for e in flag_list:
					if (int(e[0])**2 + int(e[1])**2) == min(arena):
						mid_L_top = [int(e[0]),int(e[1])]
						flag_list.remove(e)
				if i[0] == flagger:
					print("the mid_L_top location is: " + str(mid_L_top))

				#######getting mid_R_bot position########
				for r in flag_list:
					if (int(r[0])**2 + int(r[1])**2) == max(arena):
						mid_R_bot = [int(r[0]),int(r[1])]
						flag_list.remove(r)
				if i[0] == flagger:
					print("the mid_R_bot location is: " + str(mid_R_bot))

				if i[0] == flagger:
					print(flag_list)
					# print(len(flag_list))

				#######getting mid_L_bot position########
				temp = []
				if len(flag_list) > 1:
					temp_a = flag_list[0]
					temp_a = [int(temp_a[0]),int(temp_a[1])]
					
					temp_b = flag_list[1]
					temp_b = [int(temp_b[0]),int(temp_b[1])]

					if temp_a[0] < temp_b[0]:
						mid_L_bot = temp_a
						flag_list.remove(flag_list[0])
					else:
						mid_L_bot = temp_b
						flag_list.remove(flag_list[1])
				else:
					print(str(i[0])+" isn't straight cone, skipped it but please check later")
					continue

				if i[0] == flagger:
					print("the mid_L_bot location is: " + str(mid_L_bot))

				#######getting mid_R_top position########
				for n in flag_list:
					mid_R_top = [int(n[0]),int(n[1])]
				
				valid_x = []
				if i[0] == flagger:
					print("the mid_R_top location is: " + str(mid_R_top))

				###################################

				i[2] = top
				i[3] = mid_R_top
				i[4] = mid_R_bot
				i[5] = bot_R
				i[6] = bot_L
				i[7] = mid_L_bot
				i[8] = mid_L_top
				writer.writerow(i)

	storage_client.upload_file(tmp_csv_path, output_uri + tmp_csv_path)
	os.remove(tmp_csv_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--csv_uri", help="original keypoints label csv file that need to be parsed", default = 'gs://mit-dut-driverless-internal/data-labels/keypoints/Round1_revised.csv')
	parser.add_argument("--output_uri", type=str, help="Folder name to upload the parsed csv", default = 'gs://mit-dut-driverless-internal/data-labels/keypoints/')
	opt = parser.parse_args()
	
	main(csv_uri=opt.csv_uri, output_uri=opt.output_uri)
