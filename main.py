import matplotlib.pyplot as plt
import numpy as np
import statistics
import glob, os
from mpl_toolkits.mplot3d import Axes3D



def getRawData3(number_marker,toolNum):
    if(number_marker==6):
        allData = analyseTxtFiles(number_marker, None, True, False, False)
    else:
        allData = analyseTxtFiles(number_marker, None, False, False, False)

    print("allData Size", len(allData))

    x = np.array(allData[toolNum][0][0:60])
    y = np.array(allData[toolNum][1][0:60])
    z = np.array(allData[toolNum][2][0:60])

    return x,y,z

def plot_3d_scatter(x, y, z, title, setAxLim,axRangeVec):
    fig = plt.figure(figsize=(12, 6))  # Increased width for side-by-side plots
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=z, cmap='viridis')

    ax.set_xlabel('Pitch')
    ax.set_ylabel('Yaw')
    ax.set_zlabel('Roll')
    ax.set_title(title)

    if(setAxLim):
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)

        xRange = [x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1]
        yRange = [y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1]
        zRange = [z_min - (z_max - z_min) * 0.1, z_max + (z_max - z_min) * 0.1]
        ax.set_xlim(xRange)
        ax.set_ylim(yRange)
        ax.set_zlim(zRange)
        rangeVec = [xRange,yRange,zRange]
    else:
        ax.set_xlim(axRangeVec[0])
        ax.set_ylim(axRangeVec[1])
        ax.set_zlim(axRangeVec[2])

    if(setAxLim):
        return ax,rangeVec
    else:
        return ax
def scatter3d():
    fig3d = plt.figure(figsize=(12, 6))

    x,  y,  z = getRawData3(4,10)
    x1, y1, z1 = getRawData3(5,15)

    ax1 = fig3d.add_subplot(121, projection='3d')
    ax2 = fig3d.add_subplot(122, projection='3d')

    ax1.scatter(x, y, z)
    ax1.set_title("Plot 1")
    ax1.set_xlabel('Pitch')
    ax1.set_ylabel('Yaw')
    ax1.set_zlabel('Roll')
    ax1.set_xlim([statistics.mean(x)-0.3, statistics.mean(x)+0.3])
    ax1.set_ylim([statistics.mean(y)-0.3, statistics.mean(y)+0.3])
    ax1.set_zlim([statistics.mean(z)-0.3, statistics.mean(z)+0.3])

    print(statistics.mean(x))
    ax2.scatter(x1, y1, z1)
    ax2.set_title("Plot 2")
    ax2.set_xlabel('Pitch')
    ax2.set_ylabel('Yaw')
    ax2.set_zlabel('Roll')
    ax2.set_xlim([statistics.mean(x1) - 0.3, statistics.mean(x1) + 0.3])
    ax2.set_ylim([statistics.mean(y1) - 0.3, statistics.mean(y1) + 0.3])
    ax2.set_zlim([statistics.mean(z1) - 0.3, statistics.mean(z1) + 0.3])

    plt.tight_layout()
    plt.show()


def find_txt_files(path):
    os.chdir(path)
    txt_files = []
    for file in glob.glob("*.txt"):  # Search for .txt files
        txt_files.append(file)
    return txt_files


def find_txtFiles(path):
    txt_val = []
    os.chdir(path)
    equal_six = False
    txt_files = []
    for file in glob.glob("*.txt"):  # Search for .txt files
        txt_val.append(file[len(data_path_name) + 1:])
        txt_files.append(file)
        if int(file[len(data_path_name) + 1]) == 6:
            equal_six = True
        elif int(file[len(data_path_name) + 1]) == 4:
            txt_val = [vals.replace("4_", "") for vals in txt_val]

    txt_val = [vals.replace("0Pos", "") for vals in txt_val]
    txt_val = [vals.replace("1Pos", "") for vals in txt_val]
    txt_val = [vals.replace("2Pos", "") for vals in txt_val]
    txt_val = [vals.replace("3Pos", "") for vals in txt_val]
    txt_val = [vals.replace("4Pos", "") for vals in txt_val]
    txt_val = [vals.replace("SLen", "") for vals in txt_val]
    txt_val = [vals.replace("0D", "") for vals in txt_val]
    txt_val = [vals.replace("45D", "") for vals in txt_val]
    txt_val = [vals.replace("70D", "") for vals in txt_val]
    txt_val = [vals.replace(txt_val[0][0] + "_", "") for vals in txt_val]
    txt_val = [vals.replace("Val_50000.txt", "") for vals in txt_val]
    txt_val = [vals.replace("3.5mm", "") for vals in txt_val]
    txt_val = [vals.replace("_", "") for vals in txt_val]
    txt_val = [vals.replace("-", "") for vals in txt_val]

    if equal_six:
        txt_val_int = list(map(float, txt_val))
    else:
        txt_val_int = list(map(int, txt_val))

    txt_val_int.sort()

    txt_val_str = list(map(str, txt_val_int))
    return [txt_val_int, txt_val_str, txt_files]


def plot_stdVec(x_vals, vec, ax, num_marker,maxYval):
    pitch = []
    yaw = []
    roll = []
    vec_sum = []
    for val in vec:
        vec_sum.append(sum(val))
        pitch.append(val[0])
        yaw.append(val[1])
        roll.append(val[2])

    ax.set_ylabel('Standardabweichung [°]')
    ax.set_xlabel('Tool Segmentlänge [mm]')
    ax.plot(x_vals, vec_sum, marker='o', linestyle='-', color='red', markersize=5)
    ax.plot(x_vals, pitch, color='blue')
    ax.plot(x_vals, yaw, color='magenta', linestyle='--')
    ax.plot(x_vals, roll, color='black')
    ax.set_ylim([0,maxYval])
    if len(data_path_name) == 3:
        ax.legend(['sum', 'pitch', 'yaw', 'roll'])
    else:
        ax.legend(['sum', 'X', 'Y', 'Z'])
    ax.set_title(str(num_marker) + ' Marker', fontsize=14, loc='left')
 # fontweight='bold',


def plot_sum(xvals, vec, ax):
    vec_sum = []
    for val in vec:
        vec_sum.append(sum(val))
    if xvals == None:
        xvals = []
        # [xvals.append(n) for n in range(0, len(vec))]
        for n in range(0, len(vec)): xvals.append(n)
    if ax == None:
        plt.plot(xvals, vec_sum, marker='o')
    else:
        # ax.plot(xvals, vec_sum, marker='o', linestyle='-', color='cyan', markersize=5)
        ax.plot(xvals, vec_sum)
        ax.legend(['sum', 'pitch', 'yaw', 'roll', '3.5mm'])


def dataAnalysis(Data, AdjustSize,getStdData):
    Data = Data.split("\n")
    Data.remove(Data[-1])
    std_data = []
    raw_data = []
    for dataSet in Data:
        dataSet = dataSet.split(" ")
        dataSet.remove('')
        dataSet = list(map(float, dataSet))
        if AdjustSize:
            dataSet = dataSet[0:50000]
        std_data.append(statistics.stdev(dataSet))

        raw_data.append(dataSet)

    if(getStdData):
        return std_data
    else:
        return raw_data



def analyseTxtFiles(num_marker, ax, mm_isTrue, justV_isTrue,getStd):
    path = tooldata_path + str(num_marker) + "MarkerData\\"

    if mm_isTrue:
        further_path = "Data\\3.5mm\\"
        end_of_txtFile = "_3.5mm-Val_50000.txt"
        txt_names = find_txtFiles(path + data_path_name + "Data\\3.5mm")
    else:
        further_path = "Data\\"
        end_of_txtFile = "-Val_50000.txt"
        txt_names = find_txtFiles(path + data_path_name + "Data")
    stdVec = []
    rawVec = []

    if(getStd):
        for value in txt_names[1]:
            with open(path + data_path_name + further_path + data_path_name + '-' + str(
                    num_marker) + "_" + value + end_of_txtFile) as file:
                stdVec.append(dataAnalysis(file.read(), False, True))
                # rawVec.append(dataAnalysis(file.read(), False, False))
        if justV_isTrue:
            return [txt_names[0], stdVec]
        else:
            plot_stdVec(txt_names[0], stdVec, ax, num_marker,0.16)
    else:
        for value in txt_names[1]:
            with open(path + data_path_name + further_path + data_path_name + '-' + str(
                    num_marker) + "_" + value + end_of_txtFile) as file:
                rawVec.append(dataAnalysis(file.read(), False, False))

        return rawVec


def txtFiles2int(txtFiles):
    txtFiles = [vals.replace(data_path_name, "") for vals in txtFiles]
    txtFiles = [vals.replace(vals[0:3], "") for vals in txtFiles]
    txtFiles = [vals.replace(vals[4:], "") for vals in txtFiles]
    txtFiles = [vals.replace("_", "") for vals in txtFiles]
    txtFiles = [vals.replace("-", "") for vals in txtFiles]
    return list(map(int, txtFiles))


def analyseDeg():
    path = tooldata_path + "Degrees\\" + data_path_name + "Data"
    degrees = ["0", "45", "70"]
    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 5))
    ax_vec = [ax2[0][0], ax2[0][1], ax2[1][0], ax2[1][1]]
    for i, degree in enumerate(degrees):
        txtFiles = find_txtFiles(path + "\\" + degree)
        stdVec = []
        for txtFile in txtFiles[2]:
            with open(path + "\\" + degree + "\\" + txtFile) as file:
                stdVec.append(dataAnalysis(file.read(), False,True))
        plot_stdVec(txtFiles2int(txtFiles[2]), stdVec, ax_vec[i], 4,0.19)
        ax_vec[i].set_title(degree + '°', fontsize=12, loc='left')
        plot_sum(txtFiles2int(txtFiles[2]), stdVec, ax_vec[3])
        stdVec.clear()
    ax_vec[3].legend(degrees)
    ax_vec[3].set_xlabel('Tool Segmentlänge [mm]')
    ax_vec[3].set_ylabel('Standardabweichung [°]')
    ax_vec[3].set_title('Vergleich der Summen', fontsize=12, loc='left')

def plot_resorted_5Marker(vec, vec_mm, ax):
    vec[0].insert(0, vec_mm[0][3])
    vec[0].insert(0, vec_mm[0][2])
    vec[0].insert(0, vec_mm[0][1])
    vec[0].insert(0, vec_mm[0][0])
    vec[0].append(vec_mm[0][4])
    vec[0].append(vec_mm[0][5])
    vec[1].insert(0, vec_mm[1][3])
    vec[1].insert(0, vec_mm[1][2])
    vec[1].insert(0, vec_mm[1][1])
    vec[1].insert(0, vec_mm[1][0])
    vec[1].append(vec_mm[1][4])
    vec[1].append(vec_mm[1][5])
    plot_stdVec(vec[0], vec[1], ax, 5, 0.16)


def analyseDistance():
    path = tooldata_path + "Distance\\"
    stdVec = []
    legend_list = ['950mm', '1200mm', '1650mm', '1850mm', '2200mm']
    # legend_list = []
    plt.figure(3)
    order_list = [0, 1, 4, 2, 3]
    for num in order_list:
        # legend_list.append(str(num) + "Pos")
        print(num)
        txtFiles = find_txtFiles(path + str(num) + "Pos\\" + data_path_name + "Data\\")
        for txtFile in txtFiles[2]:
            with open(path + str(num) + "Pos\\" + data_path_name + "Data\\" + txtFile) as file:
                stdVec.append(dataAnalysis(file.read(), False, True))
        plot_sum(txtFiles2int(txtFiles[2]), stdVec, None)
        stdVec.clear()

    plt.legend(legend_list)
    plt.title('Distanz Untersuchung')
    plt.xlabel('Tool Segmentlänge [mm]')
    plt.ylabel('Summe Standardabweichung [°]')

def analyseSameLen():
    path = tooldata_path + "SameLength\\"
    stdVec = []
    legend_str = []
    plt.figure(4)

    for markerNum in range(3, 6):
        txtFiles = find_txt_files(path + str(markerNum) + "Marker\\" + data_path_name + "Data")
        legend_str.append(
            str(markerNum) + "Marker" + "_" + txtFiles[1][len(data_path_name) + 6:len(data_path_name) + 10])
        for txtFile in txtFiles:
            with open(path + str(markerNum) + "Marker\\" + data_path_name + "Data\\" + txtFile) as file:
                stdVec.append(dataAnalysis(file.read(), False,True))
        plot_sum(None, stdVec, None)
        stdVec.clear()

    legend_str = [vals.replace("_", " ") for vals in legend_str]
    plt.legend(legend_str)
    plt.title("Geometrie Untersuchung")
    plt.xlabel("Tool Index")
    plt.ylabel("Summe Standardabweichung [°]")

tooldata_path = "C:\\Users\\wolfd\\Desktop\\ToolDataAnalysis\\"
data_path_name = "Rot"
# data_path_name = "Trans"
#--------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(10, 5))  
analyseTxtFiles(4, ax[0][1], False, False,True)

vec1 = analyseTxtFiles(5, None, False, True,True)
vec2 = analyseTxtFiles(5, None, True, True,True)
plot_resorted_5Marker(vec1, vec2, ax[1][0])

analyseTxtFiles(6, ax[1][1], True, False,True)
analyseTxtFiles(3, ax[0][0], False, False,True)

mm_4 = analyseTxtFiles(4, None, True, True,True)

analyseDeg()
analyseDistance()
analyseSameLen()
# plt.show()
#---------------------------------------------

vec = analyseTxtFiles(3, None, False, False,False)
print(len(vec))
print(len(vec[0]))

print(len(vec[0][0]))

print(vec[2][0][0:10])
print(vec[2][2][0:10])

scatter3d()
