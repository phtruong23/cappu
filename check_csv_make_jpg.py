import csv
import Grasp_csv_Loader
import numpy as np

csv_path = '../../../Dataset/Gopro'
# csv_path = '..\\grasp_dataset\\Xsens'

csv_filename = 'SDATA1700291_annotated_data.csv'

save_folder = 'save_frames'

grasp_loader = Grasp_csv_Loader.csv_loader(data_path=csv_path,
										   csv_filename=csv_filename,
										   save_folder=save_folder)

print(len(grasp_loader.all_annotations))

# print(grasp_loader.all_annotations[0])

# print(np.shape(grasp_loader.read_frames_and_save_from_mp4(0,
# 												 'subject_1_gopro_seg_1.mp4')))

grasp_loader.total_save_from_mp4()
