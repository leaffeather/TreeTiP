#include<iostream>
#include<vector>
#include<time.h>
#include<algorithm>
#include<string>

#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/console/parse.h>


void PrintUsage(const char* progName) {
    printf("Usage: %s -i FILEPATH \n\
Options:\n\
-i FILEPATH\t\tInput point cloud file(.pcd);\n\
Result will save to *_trans.pcd in the same directory of input point cloud.\n", progName);
}

void OpenPCD(std::string strFilePath, pcl::PointCloud<pcl::PointXYZ>& pcCloud) {
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(strFilePath, pcCloud) == -1) {
        PCL_ERROR("[ERROR]Cloud not read file %s\n", strFilePath);
        exit(-1);
    }
}

void SubstrFromPath(std::string path, std::string& dir, std::string& filename)
{
    for (int i = path.size() - 1; i > 0; i--)
    {
        if (path[i] == '\\' || path[i] == '/')
        {
            filename = path.substr(i + 1);
            dir = path.substr(0, i + 1);
            return;
        }
    }
    filename = path;
    dir = "";
}
void SavePCD(std::string strFilePath, pcl::PointCloud<pcl::PointXYZ>& pcCloud) {
    if (pcl::io::savePCDFileASCII(strFilePath, pcCloud) == -1) {
        PCL_ERROR("[ERROR]Cloud not save file %s\n", strFilePath);
        exit(-1);
    }
}
int main(int argc, char** argv) {
	std::string strInputFilePath;
    if (pcl::console::parse(argc, argv, "-i", strInputFilePath) < 0) {
        PrintUsage(argv[0]);
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ> pcCloud;
    OpenPCD(strInputFilePath, pcCloud);

    float x_min = INT_MAX;
    float y_min = INT_MAX;
    float z_min = INT_MAX;
    float x_max = INT_MIN;
    float y_max = INT_MIN;
    float z_max = INT_MIN;

    for (int i = 0; i < pcCloud.size(); i++) {
        float x = pcCloud[i].x;
        float y = pcCloud[i].y;
        float z = pcCloud[i].z;
        x_min = fmin(x_min, x);
        y_min = fmin(y_min, y);
        z_min = fmin(z_min, z);
        x_max = fmax(x_max, x);
        y_max = fmax(y_max, y);
        z_max = fmax(z_max, z);
    }

    for (int i = 0; i < pcCloud.size(); i++) {
        pcCloud[i] = {
            (float)(pcCloud[i].x - x_min - 0.5 * (x_max - x_min)),
            (float)(pcCloud[i].y - y_min - 0.5 * (y_max - y_min)),
            (float)(pcCloud[i].z - z_min)
        };
    }

    std::string strInputFileDir, strInputFilename;
    SubstrFromPath(strInputFilePath, strInputFileDir, strInputFilename);

    std::string strOutputTranslatedFilename = strInputFilename.substr(0, strInputFilename.rfind(".")) + "_trans.pcd";

    SavePCD(strInputFileDir + strOutputTranslatedFilename, pcCloud);
    PCL_INFO("[INFO]Saved branch point cloud to %s\n", (strInputFileDir + strOutputTranslatedFilename).c_str());
}