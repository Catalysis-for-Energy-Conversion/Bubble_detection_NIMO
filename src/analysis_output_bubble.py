import csv
import time
import sys
import datetime
import os


class ORIGINAL():
    """Class of ORIGINAL

    This class can perform analysis of outputs from robot.

    """

    def __init__(self, input_file, output_file, num_objectives, output_folder, target_flame):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for proposals from MI algorithm
            output_file (str): the file for candidates which will be updated in this script
            num_objectives (int): the number of objectives
            output_folder (str): the folder where the output files are stored by robot

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.output_folder = output_folder
        self.target_flame = target_flame



    def perform(self):
        """perfroming analysis of output from robots

        This function do not depend on robot.
    
        Returns:
            res (str): True for success, False otherwise.

        """

        self.dt_now = datetime.datetime.now()

        print("Start analysis output!")

        #res = self.recieve_exit_message(self.output_folder)

        #if res == False:
        #    print("ErrorCode: error in recieve_exit_message function")
        #    sys.exit()
        
        res, p_List = self.load_data(self.input_file)

        if res == False:
            print("ErrorCode: error in load_data function")
            sys.exit()

        res, o_List = self.extract_objectives(self.num_objectives, self.output_folder, p_List)

        if res == False:
            print("ErrorCode: error in extract_objectives function")
            sys.exit()

        res = self.update_candidate_file(self.num_objectives, self.output_file, o_List, p_List)

        if res == False:
            print("ErrorCode: error in update_candidate_file function")
            sys.exit()


        print("Finish analysis output!")

        return "True"



    def load_data(self, input_file):
        """Loading proposals

        This function do not depend on robot.

        Args:
            input_file (str): the file for proposals from MI algorithm

        Returns:
            res (bool): True for success, False otherwise.
            p_List (list[float]): list of proposals

        """

        p_List = []

        try:
            with open(input_file) as inf:
                reader = csv.reader(inf)
                p_List = [row for row in reader]
                
            res = True

        except:
            res = False

        return res, p_List 



    def recieve_exit_message(self,output_folder):
        """Recieving exit message from machine

        This function DEPENDS on robot.

        Args:
            output_folder (str): the folder where the results by machine are stored

        Returns:
            res (bool): True for success, False otherwise.

        """

        try:
            filepath = output_folder + "/outputend.txt"

            while not(os.path.isfile(filepath)):
                time.sleep(10)

            os.remove(filepath)

            res = True


        except:
            res = False

        return res



    def extract_objectives(self, num_objectives, output_folder, p_List):    
        """Extracting objective values from output files by robot

        This function DEPENDS on robot.

        Args:
            num_objectives (int): the number of objectives
            output_folder (str): the folder where the results by machine are stored
            p_List (list[float]): the list of proposals

        Returns:
            res (bool): True for success, False otherwise.
            o_List (list[float]): the list of objectives

        """

        try:
            from bubblytics import Bubblytics
            import glob
            import numpy as np
            
            o_List = []

            for i in range(len(p_List)-1):
            
                filenames = glob.glob(output_folder + "/Exp_" + str(i+1)  + "/*.avi")

                print("Start video analysis!")

                print(filenames[0])

                if "\\" in filenames[0]:
                    input_filepath = filenames[0].replace("\\", "/")

                else:
                    input_filepath = filenames[0]

                bt = Bubblytics(input_filepath)

                bt.set_parameters(show_progress=True)

                bt.set_detection_parameters()

                bt.execute(output_folder + "/Exp_" + str(i+1))

                print(f'  whole_area = {bt.whole_area}')

                print(f'  black_area = {bt.mesh_black_area}')

                results_csv = input_filepath.replace('.avi', '') + ".csv"
            
                with open(results_csv) as f:
                    header = next(csv.reader(f))
                    reader = csv.reader(f)
                    data = [row for row in reader]

                size_data = []

                for ii in range(len(data)):
                    if int(data[ii][0]) in self.target_flame:
                        size_data.append(float(data[ii][3]))
                
                if len(size_data)==0:
                    o_List.append([p_List[i+1][0], ["E", "E"]])

                else:
                    o_List.append([p_List[i+1][0], [np.mean(size_data), np.std(size_data)]])

            res = True

        except:
            res = False

        return res, o_List


    def update_candidate_file(self, num_objectives, output_file, o_List, p_List):
        """Updating candidates

        This function do not depend on robot.

        Args:
            num_objectives (int): the number of objectives
            output_file (str): the file for candidates
            o_List (list[float]): the list of objectives
            p_List (list[float]): the list of proposals

        Returns:
            res (bool): True for success, False otherwise.

        """

        try:
            with open(output_file) as inf:
                reader = csv.reader(inf)
                c_List = [row for row in reader]

            for i in range(len(o_List)):
                combi_list = c_List[int(o_List[i][0])+1][0:-num_objectives] + o_List[i][1]
                c_List[int(o_List[i][0])+1] = combi_list

            with open(output_file, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(c_List)

            res = True

        except:
            res = False

        
        ##################
        # add at version 2

        resfolder = "./Res"

        if not os.path.exists(resfolder):
            os.makedirs(resfolder)

        resFile = self.dt_now.strftime('%Y%m%d%H%M%S') + '_res.csv'


        with open("./Res/" + resFile, mode = "w",newline = "") as outf:

            for i in range(len(p_List) - 1):
                wdata = p_List[i+1] + o_List[i][1]
                writer = csv.writer(outf)
                writer.writerow(wdata)

        ##################


        return res

