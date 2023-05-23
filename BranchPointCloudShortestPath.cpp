#include<iostream>
#include<vector>
#include<time.h>
#include<algorithm>
#include<string>
#include<queue>

#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<pcl/search/kdtree.h>
#include<pcl/console/parse.h>
#include<pcl/segmentation/impl/extract_clusters.hpp>


//Function
//Common
void SubstrFromPath(std::string, std::string&, std::string&);

//Basic info & public
void PrintUsage(const char*);
void OpenPCD(std::string, pcl::PointCloud<pcl::PointXYZ>&);

//Support function
float GetDistanceBetween2pts(pcl::PointXYZ p1, pcl::PointXYZ p2);

//Program function
bool IsPointCloudHavingDuplicatePoint(pcl::PointCloud<pcl::PointXYZ>::Ptr);
int GetRootPointIndex(pcl::PointCloud<pcl::PointXYZ>::Ptr);
float ComputeMinInterval(pcl::PointCloud<pcl::PointXYZ>::Ptr);
int GetAdjacentTable(pcl::PointCloud<pcl::PointXYZ>::Ptr, float, std::vector<std::vector<int>>&);
std::vector<float> DijkstraPriorityQueueImpl(pcl::PointCloud<pcl::PointXYZ>::Ptr, int, std::vector<std::vector<int>>);
std::vector<float> SPFA(pcl::PointCloud<pcl::PointXYZ>::Ptr, int, std::vector<std::vector<int>>);

int main(int argc, char** argv) {
    std::string strInputFilePath;
    bool bUseDijkstra = false;
    float fSearchRadius = -1;

    if (pcl::console::parse(argc, argv, "-i", strInputFilePath) < 0) {
        PrintUsage(argv[0]);
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    OpenPCD(strInputFilePath, *cloud);

    if (IsPointCloudHavingDuplicatePoint(cloud) == true) {
        PCL_ERROR("[ERROR] There are duplicate points in branch point cloud.\n");
        return -1;
    }

    if (pcl::console::find_argument(argc, argv, "-d") >= 0) {
        bUseDijkstra = true;
    }
    if (pcl::console::parse(argc, argv, "-f", fSearchRadius) >= 0) {
        if (fSearchRadius <= 0) {
            fSearchRadius = -1;
            PCL_WARN("[WARNING]Invalid fixed length, use default minimum interval distance that can make graph connected as search radius.\n");
        }
    }

    //int iSrcNode = GetRootPointIndex(cloud);
    int iSrcNode = 0;

    if (fSearchRadius <= 0) {
        fSearchRadius = ComputeMinInterval(cloud);
        PCL_INFO("[INFO]Searching neighbors with default radius = %f to create graph.\n",fSearchRadius);
    }
    else {
        PCL_INFO("[INFO]Searching neighbors with specific radius = %f to create graph.\n", fSearchRadius);
    }

    std::vector<std::vector<int>> adjacentTable;

    DWORD time_begin, time_end;
    time_begin = GetTickCount64();
    if (GetAdjacentTable(cloud,fSearchRadius, adjacentTable) < 0) {
        PCL_ERROR("[ERROR]Unreasonable search radius passed in. Graph is not connected.\n");
        return -1;
    }
    time_end = GetTickCount64();

    int iCntEdges = 0;
    for (int i = 0; i < adjacentTable.size(); i++) {
        for (int j = 0; j < adjacentTable[i].size(); j++) {
            iCntEdges++;
        }
    }
    
    PCL_INFO("[INFO]Graph was created in %f sec(s). Vertices %d, edges %d(x2).\n" , (float)(time_end - time_begin) / 1000, cloud->size(), iCntEdges/2);

    std::vector<float> vDist;

    if (bUseDijkstra == true) {
        time_begin = GetTickCount64();
        vDist = DijkstraPriorityQueueImpl(cloud, iSrcNode, adjacentTable);
        time_end = GetTickCount64();
        PCL_INFO("[INFO]Shortest-path distances were gotten in %f sec(s) with Dijkstra Algorithm using priority queue.\n", (float)(time_end - time_begin) / 1000);
    }
    else {

        time_begin = GetTickCount64();
        vDist = SPFA(cloud, iSrcNode, adjacentTable);
        time_end = GetTickCount64();
        PCL_INFO("[INFO]Shortest-path distances were gotten in %f sec(s) with SPFA.\n", (float)(time_end - time_begin) / 1000);
    }

    std::string strInputFileDir, strInputFilename;
    SubstrFromPath(strInputFilePath, strInputFileDir, strInputFilename);
    std::string strOutputDistFilename = strInputFilename.substr(0, strInputFilename.rfind(".")) + "_dist.csv";

    std::ofstream ofs;
    ofs.open(strInputFileDir + strOutputDistFilename, std::ios::out);
    if (ofs.is_open() == false) {
        return -1;
    }
    for (int i = 0; i < vDist.size(); i++) {
        ofs << vDist[i] << std::endl;
    }
    ofs.close();

    PCL_INFO("[INFO]Shortest-path distances were written into %s.\n", (strInputFileDir + strOutputDistFilename).c_str());

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
    printf("Usage: %s -i FILEPATH [-f FIXED_LEN] [-d]\n\
Options:\n\
-i FILEPATH\t\tInput tree BRANCH point cloud file(.pcd);\n\
-f FIXED_LEN\t\tFIXED_LEN as search radius to connect nodes with their neighbor nodes to create graph (Default use minimum interval distance that can make graph connected);\n\
-d\t\t\tCompute shortest-path distances with Dijkstra Algorithm using priority queue(Default use SPFA);\n\
Result will save to *_dist.csv in the same directory of input tree BRANCH point cloud. Line numbers correspond to point indices.\n\
[Attention] Please ensure the Node 0 is the root node.\n", progName);
}

void OpenPCD(std::string strFilePath, pcl::PointCloud<pcl::PointXYZ>& pcCloud) {
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(strFilePath, pcCloud) == -1) {
        PCL_ERROR("[ERROR]Cloud not read file %s\n", strFilePath);
        exit(-1);
    }
}

float GetDistanceBetween2pts(pcl::PointXYZ p1, pcl::PointXYZ p2) {
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
}

bool IsPointCloudHavingDuplicatePoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);
    for (int i = 0; i < cloud->size(); ++i)
    {
        //Considering the second neighbor since the first is the point itself.
        std::vector<int> indices(2);
        std::vector<float> sqr_distances(2);

        int nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
        pcl::PointXYZ p0 = cloud->points[i];
        pcl::PointXYZ p1 = cloud->points[indices[1]];
        if (p0.x == p1.x && p0.y == p1.y && p0.z == p1.z) {
            return true;
        }
    }
    return false;
}

int GetRootPointIndex(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    int m_rootIndex = 0; 
    double minz = cloud->points[0].z;
    for (int i = 0; i < cloud->points.size(); i++) {
        if (cloud->points[i].z < minz) {
            minz = cloud->points[i].z;
            m_rootIndex = i;
        }
    }
    return m_rootIndex;
}

class MST_with_AdjacentTable_Kruskal {
private:
    std::vector<int> father;
    std::vector<int> son;
    std::vector<std::tuple<int, int>> mst;
    std::vector<double> weights;

    typedef struct Edge {
        int start;
        int end;
        double dis;
    };
    static bool gt(const Edge& e1, const Edge& e2) {
        return e1.dis < e2.dis;
    }
    int unionsearch(int x) {//查找根节点+压缩路径
        return x == father[x] ? x : unionsearch(father[x]);
    }
    bool join(int x, int y) {//边的合并
        int root1, root2;
        root1 = unionsearch(x);
        root2 = unionsearch(y);
        if (root1 == root2)//根节点相同，故为环路
            return false;
        else if (son[root1] >= son[root2]) {
            father[root2] = root1;
            son[root1] += son[root2];
        }
        else {
            father[root1] = root2;
            son[root2] += son[root1];
        }
        return true;
    }

public:
    MST_with_AdjacentTable_Kruskal(std::vector<std::vector<std::tuple<int, double>>> vecAdjacentTable) {
        for (int i = 0; i < vecAdjacentTable.size(); i++) {
            father.push_back(i);
            son.push_back(0);
        }

        int E_total = 0;
        for (int i = 0; i < vecAdjacentTable.size(); i++) {
            for (auto j : vecAdjacentTable[i]) {
                int u = i;
                int v = std::get<0>(j);
                if (u > v)continue;
                E_total++;
            }
        }

        Edge* edge = (Edge*)malloc(E_total * sizeof(Edge));
        int count = 0;

        for (int i = 0; i < vecAdjacentTable.size(); i++) {
            for (auto j : vecAdjacentTable[i]) {
                int u = i;
                int v = std::get<0>(j);
                double w = std::get<1>(j);
                if (u > v)continue;
                edge[count] = { u,v,w };
                count++;
            }
        }
        std::sort(edge, edge + E_total, gt);
        int etotal = 0;
        bool flag = false;
        double sum = 0;
        for (int i = 0; i < E_total; i++) {
            if (join(edge[i].start, edge[i].end)) {
                etotal++;
                sum += edge[i].dis;
                mst.push_back(std::make_tuple(edge[i].start, edge[i].end));
                weights.push_back(edge[i].dis);
            }
            if (etotal == vecAdjacentTable.size() - 1) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            PCL_ERROR("[ERROR] MST Failed! Current MST only has %d edge(s)(should have %d edge(s)).\n " , etotal, vecAdjacentTable.size() - 1);
        }
    }
    std::vector<std::tuple<int, int>> GetMST() {
        return mst;
    }
    std::vector<double> GetWeights() {
        return weights;
    }
};

float ComputeMinInterval(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);
    float fMaxResolution = 0.0;
    for (int i = 0; i < cloud->size(); i++) {
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        kdtree.nearestKSearch(i, 2, pointIdxRadiusSearch, pointRadiusSquaredDistance);

        fMaxResolution = fmax(fMaxResolution, sqrt(pointRadiusSquaredDistance[1]));
    }

    float r = fMaxResolution;

    /* fMaxResolution might be less than previous radius of separating branch and leaves */
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> inlier;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
    ece.setInputCloud(cloud);
    ece.setClusterTolerance(fMaxResolution);
    ece.setMinClusterSize(1); // Least can be 1 point
    ece.setSearchMethod(tree);
    ece.extract(inlier);

    if (inlier.size() != 1) {
        std::vector<std::vector<std::tuple<int, double>>> adjacentTable(inlier.size());

        for (int s = 0; s < inlier.size(); s++) {
            for (int t = s + 1; t < inlier.size(); t++) {
                float fMinClusterInterval = INT_MAX;
                for (auto i : inlier[s].indices) {
                    for (auto j : inlier[t].indices) {
                        float dist = GetDistanceBetween2pts(cloud->points[i], cloud->points[j]);
                        fMinClusterInterval = fmin(fMinClusterInterval, dist);
                    }
                }
                adjacentTable[s].push_back(std::make_tuple(t, fMinClusterInterval));
            }
        }

        MST_with_AdjacentTable_Kruskal* mst(new MST_with_AdjacentTable_Kruskal(adjacentTable));
        std::vector<double> weights_in_mst = mst->GetWeights();

        float fMinConnectiveClusterInterval = (float)*std::max_element(weights_in_mst.begin(), weights_in_mst.end());

        r = fMinConnectiveClusterInterval;
    }
    return r;
}

int GetAdjacentTable(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float fSearchRadius, std::vector<std::vector<int>>& adjacentTable) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    adjacentTable.clear();
    adjacentTable.resize(cloud->size());

    for (int i = 0; i < cloud->size(); i++) {
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        kdtree.radiusSearch(i, fSearchRadius+0.000001, pointIdxRadiusSearch, pointRadiusSquaredDistance);

        if (pointIdxRadiusSearch.size() <= 1) {
            return -1;
        }

        for (auto j : pointIdxRadiusSearch) {
            if (i == j)
                continue;
            adjacentTable[i].push_back(j);
        }
    }
    return 0;
}

std::vector<float> DijkstraPriorityQueueImpl(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int iSrcNode, std::vector<std::vector<int>> adjacentTable ) {
    std::vector<float> vDist(cloud->size(), INT_MAX);
    vDist[iSrcNode] = 0.0;

    std::priority_queue<std::pair<float,int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> pque; //place_1 stores shortest-path distance, place_2 stores vertex

    pque.push(std::pair<float, int>(0.0, iSrcNode));

    while (!pque.empty()) {
        std::pair<float, int> p = pque.top();
        pque.pop();
        float d_i = p.first;
        int i = p.second;
        if (vDist[i] < d_i)
            continue;
        for (auto j : adjacentTable[i]) {
            float w = GetDistanceBetween2pts(cloud->points[i], cloud->points[j]);
            if (vDist[j] > vDist[i] + w) {
                vDist[j] = vDist[i] + w;
                pque.push(std::pair<float, int>(vDist[j], j));
            }
        }
    }
    return vDist;
}
std::vector<float> SPFA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int iSrcNode, std::vector<std::vector<int>> adjacentTable) {
    std::vector<float> vDist(cloud->size(), INT_MAX);
    vDist[iSrcNode] = 0.0;

    std::vector<bool> visited(cloud->size(), false);
    visited[iSrcNode] = true;

    std::queue<int> que;
    que.push(iSrcNode);

    while (!que.empty()) {
        int i = que.front();
        que.pop();

        visited[i] = false;
        for (auto j : adjacentTable[i]) {
            float w = GetDistanceBetween2pts(cloud->points[i], cloud->points[j]);
            if (vDist[j] > vDist[i] + w) {
                vDist[j] = vDist[i] + w;
                if (visited[j] == false) {
                    que.push(j);
                    visited[j] = true;
                }
            }
        }
    }
    return vDist;
}