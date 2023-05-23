#include<iostream>
#include<vector>
#include<time.h>
#include<algorithm>
#include<string>

#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/visualization/cloud_viewer.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<pcl/search/kdtree.h>
#include<pcl/console/parse.h>
#include<pcl/filters/voxel_grid.h>
#include<pcl/segmentation/impl/extract_clusters.hpp>
#include <pcl/filters/extract_indices.h>

#include <boost/thread/thread.hpp>

//Environment parameter
//Camera parameter
static double pos_x = 0;
static double pos_y = 0;
static double pos_z = 0;
static double view_x = 0;
static double view_y = 0;
static double view_z = 1;
static double up_x = 0;
static double up_y = 1;
static double up_z = 0;

//Function
//Common
void SubstrFromPath(std::string, std::string&, std::string&);

//Basic info & public
void PrintUsage(const char*);
void OpenPCD(std::string, pcl::PointCloud<pcl::PointXYZ>&);
void SavePCD(std::string, pcl::PointCloud<pcl::PointXYZ>&);
void SetCameraParameters(double, double, double, double, double, double, double, double, double);
int OpenCameraFile(std::string);
int SaveCameraFile(std::string);
void SaveCurrentCameraParameters(boost::shared_ptr<pcl::visualization::PCLVisualizer>);

//Support function
double ComputeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::Ptr);

//Program function
void RemoveClosePoint(pcl::PointCloud<pcl::PointXYZ>&, double);
void SeparateBranchAndLeaves(pcl::PointCloud<pcl::PointXYZ>, double,int, pcl::PointCloud<pcl::PointXYZ>&, pcl::PointCloud<pcl::PointXYZ>&);
void ViewTreePointCloud(pcl::PointCloud<pcl::PointXYZ>);
void ViewBranchAndLeavesPointCloud(pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointXYZ>);
int DetectRootNode(pcl::PointCloud<pcl::PointXYZ>, int, double);

int main(int argc, char** argv) {
    std::string strInputFilePath;
    double dMagnification = 10.0;
    double dEpsilon = 0.000001;
    double dFixedLen = -1;
    bool bViewInputPC = false;
    bool bViewOutputPC = false;
    float fSubsampleLeafSize = -1;
    int iMinPts = 3;
    
    if (pcl::console::parse(argc, argv, "-i", strInputFilePath) < 0) {
        PrintUsage(argv[0]);
        return -1;
    }
    if (pcl::console::parse(argc, argv, "-e", dEpsilon) >= 0) {
        if (dEpsilon <= 0) {
            dEpsilon = 0.000001;
            PCL_WARN("[WARNING]Invalid epsilon, use default 0.000001.\n");
        }
    }
    if (pcl::console::parse(argc, argv, "-m", dMagnification) >= 0) {
        if (dMagnification <= 0) {
            dMagnification = 10.0;
            PCL_WARN("[WARNING]Invalid magnification, use default 10.0.\n");
        }
    }
    if (pcl::console::parse(argc, argv, "-f", dFixedLen) >= 0) {
        if (dFixedLen <= 0) {
            dFixedLen = -1;
            PCL_WARN("[WARNING]Invalid fixed length, use magnification(%f) * resolution as search radius.\n", dMagnification);
        }
    }
    if (pcl::console::parse(argc, argv, "-p", iMinPts) >= 0) {
        if (iMinPts <= 0) {
            iMinPts = 1;
            PCL_WARN("[WARNING]Invalid minimum points of a cluster, changed to %d\n", iMinPts);
        }
    }
    if (pcl::console::find_argument(argc, argv, "-vin") >= 0) {
        bViewInputPC = true;
    }
    if (pcl::console::find_argument(argc, argv, "-vout") >= 0) {
        bViewOutputPC = true;
    }

    pcl::PointCloud<pcl::PointXYZ> pcTree;
    OpenPCD(strInputFilePath, pcTree);

    if (pcl::console::parse(argc, argv, "-s", fSubsampleLeafSize) >= 0) {
        if (fSubsampleLeafSize <= 0) {
            PCL_ERROR("[ERROR] Invalid or unspecified subsample leaf size.\n");
            return -1;
        }
        
        DWORD dwTimeStart_sor, dwTimeEnd_sor;
        PCL_INFO("[INFO]Subsampling tree point cloud with leaf size %f. Original size %d.\n",fSubsampleLeafSize,pcTree.size());
        dwTimeStart_sor = GetTickCount64();
        
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(pcTree.makeShared());
        sor.setLeafSize(fSubsampleLeafSize, fSubsampleLeafSize, fSubsampleLeafSize);
        sor.filter(pcTree);

        dwTimeEnd_sor = GetTickCount64();
        PCL_INFO("[INFO]Subsampling completed in %f sec(s). Current size %d.\n", (float)(dwTimeEnd_sor - dwTimeStart_sor)/1000, pcTree.size());
    }
    
    DWORD dwTimeStart_remove, dwTimeEnd_remove;
    dwTimeStart_remove = GetTickCount64();
    PCL_INFO("[INFO]Removing close points with epsilon %f. Original size %d.\n", dEpsilon, pcTree.size());

    RemoveClosePoint(pcTree, dEpsilon);

    dwTimeEnd_remove = GetTickCount64();
    PCL_INFO("[INFO]Removing completed in %f sec(s). Current size %d.\n", (float)(dwTimeEnd_remove - dwTimeStart_remove) / 1000, pcTree.size());

    double dResolution = ComputeCloudResolution(pcTree.makeShared());

    DWORD dwTimeStart_separate, dwTimeEnd_separate;
    dwTimeStart_separate = GetTickCount64();

    pcl::PointCloud<pcl::PointXYZ> pcBranch;
    pcl::PointCloud<pcl::PointXYZ> pcLeaves;
    if (dFixedLen <= 0) {
        PCL_INFO("[INFO]Detecting the root node with MinPts = %d\n", iMinPts);
        int iRootNode = DetectRootNode(pcTree, iMinPts, dMagnification * dResolution);
        PCL_INFO("[INFO]Separating branch and leaves with radius %f(%fx point cloud resolution).\n", dMagnification * dResolution, dMagnification);
        SeparateBranchAndLeaves(pcTree, dMagnification * dResolution, iRootNode, pcBranch, pcLeaves);
    }
    else {
        PCL_INFO("[INFO]Detecting the root node with MinPts = %d\n", iMinPts);
        int iRootNode = DetectRootNode(pcTree, iMinPts, dFixedLen);
        PCL_INFO("[INFO]Separating branch and leaves with radius %f(fixed length).\n", dFixedLen);
        SeparateBranchAndLeaves(pcTree, dFixedLen, iRootNode, pcBranch, pcLeaves);
    }

    dwTimeEnd_separate = GetTickCount64();
    PCL_INFO("[INFO]Separating completed in %f sec(s). Branch point cloud size %d, leaves point cloud size %d.\n", (float)(dwTimeEnd_separate - dwTimeStart_separate) / 1000, pcBranch.size(), pcLeaves.size());

    std::string strInputFileDir, strInputFilename;
    SubstrFromPath(strInputFilePath, strInputFileDir, strInputFilename);

    std::string strOutputBranchFilename = strInputFilename.substr(0, strInputFilename.rfind(".")) + "_branch.pcd";
    std::string strOutputLeavesFilename = strInputFilename.substr(0, strInputFilename.rfind(".")) + "_leaves.pcd";

    SavePCD(strInputFileDir + strOutputBranchFilename, pcBranch);
    PCL_INFO("[INFO]Saved branch point cloud to %s\n", (strInputFileDir + strOutputBranchFilename).c_str());

    if (pcLeaves.size() != 0) {
        SavePCD(strInputFileDir + strOutputLeavesFilename, pcLeaves);
        PCL_INFO("[INFO]Saved leaves point cloud to %s\n", (strInputFileDir + strOutputLeavesFilename).c_str());
    }
    else {
        PCL_WARN("[WARNING]%s is a void leaves point cloud, and will not save. If it existed, it would be deleted.\n", (strInputFileDir + strOutputLeavesFilename).c_str());
        remove((strInputFileDir + strOutputLeavesFilename).c_str());
    }
    

    //Visualization
    if (bViewInputPC == false && bViewOutputPC == false) {
        return 0;
    }
    std::string strCameraFilePath;
    bool bNeedWritingCameraParas = false;
    if (pcl::console::find_argument(argc, argv, "-va") >= 0) {
        bNeedWritingCameraParas = true;
        PCL_INFO("[INFO]Enabled camera parameters writing to file, you can stop it by pressing CTRL+C in Terminal not PCLVisualizer before the program ends.\n");
    }
    if (pcl::console::parse(argc, argv, "-vp", strCameraFilePath) >= 0) {
        if (OpenCameraFile(strCameraFilePath) == -1) {
            //Invalid camera parameters' file, create a new file
            PCL_WARN("[WARNING]Invalid camera parameters' file, will use default parameters.\n");
        }
    }
    if (bViewInputPC == true) {
        ViewTreePointCloud(pcTree);
    }
    if (bViewOutputPC == true) {
        ViewBranchAndLeavesPointCloud(pcBranch, pcLeaves);
    }
    if (bNeedWritingCameraParas == true) {
        if (SaveCameraFile(strCameraFilePath) == -1) {
            PCL_ERROR("[ERROR]Cannot save file. Please check whether -va FILENAME is valid or not.\n");
        }
        else {
            PCL_INFO("[INFO]Camera parameter's file %s has been saved.", strCameraFilePath.c_str());
        }
    }

    return 0;
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

void PrintUsage(const char* progName) {
    printf("Usage: %s -i FILEPATH [-s SUBSAMPLE_SIZE] [-e FILTING_RADIUS] [-p MINPTS] [-m MAGNIFICATION | -f FIXED_LEN] [-vin] [-vout] [-vp FILEPATH [-va]]\n\
Options:\n\
-i FILEPATH\t\tInput TREE point cloud file(.pcd);\n\
-s SUBSAMPLE_SIZE\tVoxel side length to subsample;\n\
-e FILTING_RADIUS\tRemoval radius of repetitive points(default: 0.000001);\n\
-p MINPTS\t\tMin Points of a cluster to detect the main branch(default:3);\n\
-m MAGNIFICATION\tMAGNIFICATION * point_cloud_resolution as radius of separating branch and leaves(default:10)\n\
-f FIXED_LEN\t\tFIXED_LEN as radius of separating branch and leaves(default not work and -f is prior to -m)\n\
-vin\t\t\tView tree point cloud;\n\
-vout\t\t\tView separation result;\n\
-vp FILEPATH\t\tImport a camera parameters' file(.txt)(if not exists, use default parameters);\n\
-va\t\t\tAuto writing camera parameters to file.\n\
Result will save to *_branch.pcd and *_leaves.pcd in the same directory of input TREE point cloud.\n", progName);
}

void OpenPCD(std::string strFilePath, pcl::PointCloud<pcl::PointXYZ>& pcCloud) {
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(strFilePath, pcCloud) == -1) {
        PCL_ERROR("[ERROR]Cloud not read file %s\n", strFilePath);
        exit(-1);
    }
}

void SavePCD(std::string strFilePath, pcl::PointCloud<pcl::PointXYZ>& pcCloud) {
    if (pcl::io::savePCDFileASCII(strFilePath,pcCloud) == -1) {
        PCL_ERROR("[ERROR]Cloud not save file %s\n", strFilePath);
        exit(-1);
    }
}

int OpenCameraFile(std::string strFilePath) {
    fstream file(strFilePath, ios::in);
    if (file.fail() == true)
        return -1;

    std::vector<double> vParas;
    try {
        while (!file.eof()) {
            char buf[128];
            file.getline(buf, 128);
            std::string buf_str = buf;
            if (std::all_of(buf_str.begin(), buf_str.end(), isspace) == true) {
                // Pass a blank line
                continue;
            }

            /*
            // Remove begin and end blanks
            std::string blanks("\f\v\r\t\n ");
            buf_str.erase(0, buf_str.find_first_not_of(blanks));
            buf_str.erase(buf_str.find_last_not_of(blanks) + 1);
            */

            double dPara = std::stod(buf_str); //If not a number or out_of_range will raise exception
            vParas.push_back(dPara);

            if (vParas.size() > 9)break;
        }
    }
    catch (const std::exception&) {
        file.close();
        return -1;
    }
    file.close();
    if (vParas.size() != 9) {
        return -1;
    }

    SetCameraParameters(vParas[0], vParas[1], vParas[2], vParas[3], vParas[4], vParas[5], vParas[6], vParas[7], vParas[8]);

    return 0;
}


int SaveCameraFile(std::string strFilePath) {
    ofstream ofs;
    ofs.open(strFilePath, ios::out);
    if (ofs.is_open() == false) {
        return -1;
    }
    ofs << pos_x << std::endl << pos_y << std::endl << pos_z << std::endl\
        << view_x << std::endl << view_y << std::endl << view_z << std::endl\
        << up_x << std::endl << up_y << std::endl << up_z;
    ofs.close();
    return 0;
};

double ComputeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    double res = 0.0;
    int n_points = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> sqr_distances(2);
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i)
    {
        if (!pcl_isfinite((*cloud)[i].x))
        {
            continue;
        }
        //Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
        if (nres == 2)
        {
            res += sqrt(sqr_distances[1]);
            ++n_points;
        }
    }
    if (n_points != 0)
    {
        res /= n_points;
    }
    return res;
}

void SetCameraParameters(double current_pos_x,
    double current_pos_y,
    double current_pos_z,
    double current_view_x,
    double current_view_y,
    double current_view_z,
    double current_up_x,
    double current_up_y,
    double current_up_z) {
    pos_x = current_pos_x;
    pos_y = current_pos_y;
    pos_z = current_pos_z;
    view_x = current_view_x;
    view_y = current_view_y;
    view_z = current_view_z;
    up_x = current_up_x;
    up_y = current_up_y;
    up_z = current_up_z;
}

void RemoveClosePoint(pcl::PointCloud<pcl::PointXYZ>& cloud, double epsilon) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree; 
    kdtree.setInputCloud(cloud.makeShared());
    bool* mask = (bool*)calloc(cloud.points.size(), sizeof(bool)); 
    memset(mask, 0, cloud.points.size() * sizeof(bool));

    //std::vector<size_t> duplicatePtsNo;
    pcl::PointIndices::Ptr duplicatePtsNo(new pcl::PointIndices());

    for (size_t i = 0; i < cloud.points.size(); i++) {
        mask[i] = true;
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance; 
        if (kdtree.radiusSearch(cloud.points[i], epsilon, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
            for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++) {
                if (mask[pointIdxRadiusSearch[j]] == false)
                {
                    mask[pointIdxRadiusSearch[j]] = true;
                    //duplicatePtsNo.push_back(pointIdxRadiusSearch[j]);
                    duplicatePtsNo->indices.push_back(pointIdxRadiusSearch[j]);
                }
            }
        }
    }


    std::sort(duplicatePtsNo->indices.begin(), duplicatePtsNo->indices.end());
    /*
    for (size_t i = 0; i < duplicatePtsNo.size(); i++) {
        cloud.erase(cloud.begin() + duplicatePtsNo[i] - i);
    }*/

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud.makeShared());
    extract.setIndices(duplicatePtsNo);
    extract.setNegative(true);
    extract.filter(cloud);

}

class ECE {
private:
    std::vector<pcl::PointIndices> inlier;
public:
    ECE(pcl::PointCloud<pcl::PointXYZ> cloud, double dSearchRadius, int iMinClusterSize) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud(new pcl::PointCloud<pcl::PointXYZ>);
        _cloud = cloud.makeShared();
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
        ece.setInputCloud(_cloud);
        ece.setClusterTolerance(dSearchRadius);
        ece.setMinClusterSize(iMinClusterSize);
        ece.setSearchMethod(tree);
        ece.extract(inlier);
    }
    std::vector<pcl::PointIndices> GetIndices() {
        return inlier;
    }
};

int DetectRootNode(pcl::PointCloud<pcl::PointXYZ> pcTree, int iMinPts, double dSearchRadius) {
    //Cluster all the points lower than 1.3 m(if tree height is lower than 1.3 m, zoom it) and find the max cluster
    float z_min = INT_MAX;
    float z_max = INT_MIN;
    for (int i = 0; i < pcTree.size(); i++) {
        z_min = fmin(z_min, pcTree[i].z);
        z_max = fmax(z_max, pcTree[i].z);
    }

    float z_upbound = fmin(1.3, 1.3 / 5 * (z_max - z_min));

    std::vector<int> idx_from_pcTree;
    pcl::PointCloud<pcl::PointXYZ> pcMainBranchPart;

    for (int i = 0; i < pcTree.size(); i++) {
        if (pcTree[i].z >= z_min && pcTree[i].z <= z_min + z_upbound) {
            pcMainBranchPart.push_back(pcTree[i]);
            idx_from_pcTree.push_back(i);
        }
    }

    ECE* ece(new ECE(pcMainBranchPart, dSearchRadius, iMinPts));
    //Main branch must be the most points
    std::vector<pcl::PointIndices> inlier = ece->GetIndices();
    std::vector<int> inlier_size;
    for (int i = 0; i < inlier.size(); i++) {
        inlier_size.push_back(inlier[i].indices.size());
    }
    delete ece;
    int max_cluster_idx = std::distance(inlier_size.begin(), std::max_element(inlier_size.begin(), inlier_size.end()));
    
    float iMainBranch_z_min = INT_MAX;
    int iMainBranch_z_min_idx = -1;
    for (int i = 0; i < inlier[max_cluster_idx].indices.size(); i++) {
        if (pcMainBranchPart[inlier[max_cluster_idx].indices[i]].z < iMainBranch_z_min) {
            iMainBranch_z_min = pcMainBranchPart[inlier[max_cluster_idx].indices[i]].z;
            iMainBranch_z_min_idx = inlier[max_cluster_idx].indices[i];
        }
    }

    return idx_from_pcTree[iMainBranch_z_min_idx];
}

void SeparateBranchAndLeaves(pcl::PointCloud<pcl::PointXYZ> pcTree, double dSearchRadius, int iRootNode, pcl::PointCloud<pcl::PointXYZ>& pcBranch, pcl::PointCloud<pcl::PointXYZ>& pcLeaves) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcTreePtr = pcTree.makeShared();

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree; 
    kdtree.setInputCloud(pcTreePtr);

    bool* mask = (bool*)calloc(pcTreePtr->points.size(), sizeof(bool)); 
    memset(mask, 0, pcTreePtr->points.size() * sizeof(bool));

    /*
    size_t m_rootIndex = 0; 
    double minz = pcTreePtr->points[0].z;
    for (size_t i = 0; i < pcTreePtr->points.size(); i++) {
        if (pcTreePtr->points[i].z < minz) {
            minz = pcTreePtr->points[i].z;
            m_rootIndex = i;
        }
    }*/

    size_t m_rootIndex = iRootNode;

    mask[m_rootIndex] = true; 
    pcBranch.push_back(pcTreePtr->points[m_rootIndex]);
    for (size_t i = 0; i < pcBranch.points.size(); i++) {
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance; 

        if (kdtree.radiusSearch(pcBranch.points[i], dSearchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
            for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++) {
                if (mask[pointIdxRadiusSearch[j]] == false)
                {
                    mask[pointIdxRadiusSearch[j]] = true;
                    pcBranch.push_back(pcTreePtr->points[pointIdxRadiusSearch[j]]);
                }
            }
        }
    }
    for (size_t i = 0; i < pcTreePtr->points.size(); i++) {
        if (mask[i] == false) {
            pcLeaves.push_back(pcTreePtr->points[i]);
        }
    }
}

void ViewTreePointCloud(pcl::PointCloud<pcl::PointXYZ> pcTree) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcTreePtr = pcTree.makeShared();

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> clrTree(pcTreePtr, 0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(pcTreePtr, clrTree, "tree");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "tree");

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPosition(pos_x, pos_y, pos_z, view_x, view_y, view_z, up_x, up_y, up_z);
    
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));

        SaveCurrentCameraParameters(viewer);
    }
}

void ViewBranchAndLeavesPointCloud(pcl::PointCloud<pcl::PointXYZ> pcBranch, pcl::PointCloud<pcl::PointXYZ> pcLeaves) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcBranchPtr = pcBranch.makeShared();
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> clrBranch(pcBranchPtr, 140, 81, 25);
    viewer->addPointCloud<pcl::PointXYZ>(pcBranchPtr, clrBranch, "branch");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "branch");

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcLeavesPtr = pcLeaves.makeShared();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> clrLeaves(pcLeavesPtr, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(pcLeavesPtr, clrLeaves, "leaves");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "leaves");

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPosition(pos_x, pos_y, pos_z, view_x, view_y, view_z, up_x, up_y, up_z);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));

        SaveCurrentCameraParameters(viewer);
    }
}


void SaveCurrentCameraParameters(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer) {
    pcl::visualization::Camera camera;
    viewer->getCameraParameters(camera);
    double* pos = camera.pos;
    double* view = camera.focal;
    double* up = camera.view;

    double current_pos_x = *pos;
    double current_pos_y = *(pos + 1);
    double current_pos_z = *(pos + 2);
    double current_view_x = *view;
    double current_view_y = *(view + 1);
    double current_view_z = *(view + 2);
    double current_up_x = *up;
    double current_up_y = *(up + 1);
    double current_up_z = *(up + 2);

    if (current_pos_x == pos_x && current_pos_y == pos_y && current_pos_z == pos_z \
        && current_view_x == view_x && current_view_y == view_y && current_view_z == view_z \
        && current_up_x == up_x && current_up_y == up_y && current_up_z == up_z) {
        return;
    }

    SetCameraParameters(current_pos_x, current_pos_y, current_pos_z, current_view_x, current_view_y, current_view_z, current_up_x, current_up_y, current_up_z);
}