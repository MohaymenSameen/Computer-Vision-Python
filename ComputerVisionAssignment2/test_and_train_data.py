import os
import splitfolders
input_folder = os.path.dirname(r"G:\programming\Github\Computer-Vision-Python\ComputerVisionAssignment2\all_test_tubes\Venosafe\Dset")
output = "output_path" #where you want the split datasets saved. one will be created if none is set

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .1, .1)) # ratio of split are in order of train/val/test. You can change to whatever you want. For train/val sets only, you could do .75, .25 for example.