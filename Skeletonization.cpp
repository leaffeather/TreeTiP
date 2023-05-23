#include<iostream>
#include<vector>
#include<time.h>
#include<algorithm>
#include<string>
#include<thread>

#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<pcl/search/kdtree.h>
#include<pcl/console/parse.h>
#include<pcl/segmentation/impl/extract_clusters.hpp>
#include<pcl/common/centroid.h>
#include<pcl/visualization/cloud_viewer.h>
#include<boost/thread/thread.hpp>
#include <pcl/filters/extract_indices.h>

#include <Eigen/Eigen>

#include <boost/polygon/polygon.hpp>
#include <boost/geometry.hpp>

#include<vtkLine.h>
#include<vtkOBJWriter.h>
#include<vtkPLYWriter.h>

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
int OpenDist(std::string, std::vector<float>&);
void SetCameraParameters(double, double, double, double, double, double, double, double, double);
int OpenCameraFile(std::string);
int SaveCameraFile(std::string);
void SaveCurrentCameraParameters(boost::shared_ptr<pcl::visualization::PCLVisualizer>);

//Support function
float GetDistanceBetween2pts(pcl::PointXYZ, pcl::PointXYZ);
bool IsSamePoint(pcl::PointXYZ, pcl::PointXYZ);

//Program function
float ComputeMinInterval(pcl::PointCloud<pcl::PointXYZ>::Ptr);

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

class ISTTWN {
private:
    static pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    static std::vector<float> vDist;
    static float fSearchRadius;
    static int iMinPts;
    static float fStepLen;
    pcl::PointCloud<pcl::PointXYZ> skeletonPoints;
    std::vector<std::tuple<int, int>> skeletonEdges;
    std::vector<pcl::PointCloud<pcl::PointXYZ>> bin;

    static void RecursionImpl(pcl::PointIndices indices, float fGrownDist, float fMaxDist, pcl::PointXYZ centroid_prev, std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ>>& skeletonLines, std::vector<pcl::PointCloud<pcl::PointXYZ>>& bin_seq, bool bFirstTurn = true) {
        if (fGrownDist > fMaxDist || indices.indices.size() == 0) {
            return;
        }
        std::vector<pcl::PointIndices> inlier;

        pcl::PointCloud<pcl::PointXYZ>* _cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*cloud, indices, *_cloud);
        ECE* ece(new ECE(*_cloud, fSearchRadius, iMinPts));
        inlier = ece->GetIndices();
        delete(ece);
        delete _cloud;

        for (int s = 0; s < inlier.size(); s++) {
            for (int t = 0; t < inlier[s].indices.size(); t++) {
                //Convert _cloud index to cloud index
                inlier[s].indices[t] = indices.indices[inlier[s].indices[t]];
            }
        }


        while (inlier.size() != 0) {
            pcl::PointCloud<pcl::PointXYZ>* _bin(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointIndices sub_indices;

            if (bFirstTurn == true) {
                /* If it is the first turn, the point equals to 0.0 should be included in, or the following turn will produce two clusters. */
                for (auto i : inlier.back().indices) {
                    if (vDist[i] >= fGrownDist && vDist[i] <= fGrownDist + fStepLen) {
                        _bin->push_back(cloud->points[i]);
                    }
                    else {
                        sub_indices.indices.push_back(i);
                    }
                }
            }
            else {
                for (auto i : inlier.back().indices) {
                    if (vDist[i] > fGrownDist && vDist[i] <= fGrownDist + fStepLen) {
                        _bin->push_back(cloud->points[i]);
                    }
                    else {
                        sub_indices.indices.push_back(i);
                    }
                }
            }
            inlier.pop_back();

            Eigen::Vector4f vCentroid;
            pcl::compute3DCentroid(*_bin, vCentroid);
            pcl::PointXYZ centroid(vCentroid(0), vCentroid(1), vCentroid(2));

            if (bFirstTurn == true) { // The first turn needn't produce the skeleton line
                bin_seq.push_back(*_bin);
            }
            else {
                bin_seq.push_back(*_bin);
                skeletonLines.push_back(std::make_tuple(centroid_prev, centroid));
            }

            delete _bin;

            RecursionImpl(sub_indices, fGrownDist + fStepLen, fMaxDist, centroid, skeletonLines, bin_seq, false);
        }
    }

    static void IterativeImpl(float fGrownDist, float fMaxDist, int iSteps, std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ>>& skeletonLines, std::vector<pcl::PointCloud<pcl::PointXYZ>>& binSeq, int& iFlag) {
        while (fGrownDist < fMaxDist) {
            pcl::PointIndices indices_in_interval;
            
            float fMaxGrownDistInInterval = fGrownDist + (iSteps + 1) * fStepLen;//In order to connect intervals, plus one is to include indices of the first step in next interval

            for (int s = 0; s < vDist.size(); s++) {
                if (fGrownDist == 0.0) {
                    if (vDist[s] >= fGrownDist && vDist[s] <= fMaxGrownDistInInterval) {
                        indices_in_interval.indices.push_back(s);
                    }
                }
                else {
                    if (vDist[s] > fGrownDist && vDist[s] <= fMaxGrownDistInInterval) {
                        indices_in_interval.indices.push_back(s);
                    }
                }
            }

            std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ>> skeletonLinesInInterval;
            std::vector<pcl::PointCloud<pcl::PointXYZ>> binSeqInInterval;

            RecursionImpl(indices_in_interval, fGrownDist, fMaxGrownDistInInterval, { INT_MIN,INT_MIN,INT_MIN }, skeletonLinesInInterval, binSeqInInterval); // If the grown distance is more than 0.0, there are no distances in the interval will be the same as the grown distance

            skeletonLines.insert(skeletonLines.end(), skeletonLinesInInterval.begin(), skeletonLinesInInterval.end());
            binSeq.insert(binSeq.end(), binSeqInInterval.begin(), binSeqInInterval.end());
            
            fGrownDist = fMaxGrownDistInInterval - fStepLen;
        }
        iFlag++;
    }

    void IterativeMultithreading(int iStepsEachInterval, std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ>>& skeletonLines, std::vector<pcl::PointCloud<pcl::PointXYZ>>& binSeq) {

        float fMaxDist = *std::max_element(vDist.begin(), vDist.end());
        int iIntervals = (int)ceilf(fMaxDist / fStepLen); // Max dist can be divided by step length into such number of intervals

        UINT uiCntThreads = std::thread::hardware_concurrency();
        int iIntervalsEachThread = (int)ceilf(iIntervals / uiCntThreads);

        int iCntFinishedThreads = 0;

        std::vector<std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ>>> skeletonLines_t(uiCntThreads);
        std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>>> binSeq_t(uiCntThreads);


        for (int s = 0; s < uiCntThreads; s++) {
            std::thread t(IterativeImpl, (float)s* iIntervalsEachThread *fStepLen, (float)((s+1)*iIntervalsEachThread)*fStepLen, iStepsEachInterval,std::ref(skeletonLines_t[s]), std::ref(binSeq_t[s]), std::ref(iCntFinishedThreads));

            t.detach();
        }

        while (true) {
            if (iCntFinishedThreads == uiCntThreads) {
                break;
            }
            Sleep(20);
        }

        for (int s = 0; s < uiCntThreads; s++) {
            skeletonLines.insert(skeletonLines.end(), skeletonLines_t[s].begin(), skeletonLines_t[s].end());
            binSeq.insert(binSeq.end(), binSeq_t[s].begin(), binSeq_t[s].end());
        }

    }

    void SkeletonLines2SkeletonPointsAndSkeletonEdges(std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ>>& skeletonLines, pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges) {
        while (!skeletonLines.empty()) {
            std::tuple<pcl::PointXYZ, pcl::PointXYZ> skeletonLine = skeletonLines[0];

            int iStartNode = -1;
            int iEndNode = -1;

            pcl::PointXYZ ptStart = std::get<0>(skeletonLine);
            pcl::PointXYZ ptEnd = std::get<1>(skeletonLine);

            for (int t = 0; t < skeletonPoints.size(); t++) {
                if (iStartNode == -1) {
                    if (IsSamePoint(skeletonPoints[t], ptStart)) {
                        iStartNode = t;
                    }
                }
                if (iEndNode == -1) {
                    if (IsSamePoint(skeletonPoints[t], ptEnd)) {
                        iEndNode = t;
                    }
                }
                if (iStartNode != -1 && iEndNode != -1){
                    break;
                }
            }

            if (iStartNode == -1) {
                iStartNode = skeletonPoints.size();
                skeletonPoints.push_back(ptStart);
            }
            if (iEndNode == -1) {
                iEndNode = skeletonPoints.size();
                skeletonPoints.push_back(ptEnd);
            }
            if (iStartNode != iEndNode) {
                bool bHasDuplicateLines = false;
                for (int s = 0; s < skeletonEdges.size(); s++) {
                    // Duplicate lines will happen in multithreading, as the chosen interval of interation it calls will include margin part
                    if (std::get<0>(skeletonEdges[s]) == iStartNode && std::get<1>(skeletonEdges[s]) == iEndNode) {
                        bHasDuplicateLines = true;
                        break;
                    }
                }
                if(bHasDuplicateLines== false)
                    skeletonEdges.push_back(std::make_tuple(iStartNode, iEndNode));
            }
            skeletonLines.erase(skeletonLines.begin());
        }
    }

    void IsOnTheSuboffshoot(int begin_node, int test_node, bool& result) {
        std::vector<int> end_nodes;
        for (auto edge : skeletonEdges) {
            if (std::get<0>(edge) == begin_node) {
                end_nodes.push_back(std::get<1>(edge));
            }
        }
        for (auto next_begin_node : end_nodes) {
            if (next_begin_node == test_node) {
                result = true;
                return;
            }
        }
        for (auto next_begin_node : end_nodes) {
            IsOnTheSuboffshoot(next_begin_node, test_node, result);
        }
    }

    int ConnectBreakpoint(std::vector<int>& indegree, std::vector<int>& outdegree) {
        std::vector<int> breakpoints_node;
        for (int s = 1; s < indegree.size(); s++) { // Index 0 is the root. 
            if (indegree[s] == 0) {
                breakpoints_node.push_back(s);
            }
        }
        if (breakpoints_node.size() == 0)
            return 0;
        int breakpoints_cnt = breakpoints_node.size();

        while (!breakpoints_node.empty()) {
            int breakpoint_node = breakpoints_node[0]; // Patch from front to end
            breakpoints_node.erase(breakpoints_node.begin());

            std::vector<int> breakpoint_node_3neighbors; // Exclude every neighbors on the sub-offshoot beginning with this breakpoint 

            int k = 1; //Explore flag;

            while (true) {
                k++;
                if (k > skeletonPoints.size() || breakpoint_node_3neighbors.size() >= 3) {
                    break;
                }

                pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
                kdtree.setInputCloud(skeletonPoints.makeShared());

                std::vector<int> pointIdxRadiusSearch;
                std::vector<float> pointRadiusSquaredDistance;
                kdtree.nearestKSearch(breakpoint_node, k, pointIdxRadiusSearch, pointRadiusSquaredDistance);

                bool bBelongToBreakNodeOffshoot = false;
                IsOnTheSuboffshoot(breakpoint_node, pointIdxRadiusSearch[k - 1], bBelongToBreakNodeOffshoot);

                if (bBelongToBreakNodeOffshoot == false) {
                    breakpoint_node_3neighbors.push_back(pointIdxRadiusSearch[k - 1]);
                }
            }

            if (breakpoint_node_3neighbors.size() == 0) {
                continue;
            }

            /* Judge whether bin of breakpoints and bin of one neighbor have partial common points, if so, they should be the same point */
            
            std::vector<int> breakpoint_node_3neighbors_bin_intersect_points(breakpoint_node_3neighbors.size(),0);
            for (int t = 0; t < breakpoint_node_3neighbors.size(); t++) {
                int neighbor_node = breakpoint_node_3neighbors[t];
                int count_intersect_points = 0;
                for (auto pi : bin[breakpoint_node]) {
                    for (auto pj : bin[neighbor_node]) {
                        if (IsSamePoint(pi, pj) == true) {
                            count_intersect_points++;
                            break;
                        }
                    }
                }
                breakpoint_node_3neighbors_bin_intersect_points[t] = count_intersect_points;
            }

            if (*std::max_element(breakpoint_node_3neighbors_bin_intersect_points.begin(), breakpoint_node_3neighbors_bin_intersect_points.end()) == 0) {
                //If every neighbors are not the same point as breakpoints, directly connect the nearest neighbor to the breakpoint
                skeletonEdges.push_back(std::make_tuple(breakpoint_node_3neighbors[0], breakpoint_node));
                outdegree[breakpoint_node_3neighbors[0]]++;
                indegree[breakpoint_node]++;
            }
            else {
                //If there are some common points, combine two node.
                int neighbor_node = breakpoint_node_3neighbors[std::distance(breakpoint_node_3neighbors_bin_intersect_points.begin(),std::max_element(breakpoint_node_3neighbors_bin_intersect_points.begin(), breakpoint_node_3neighbors_bin_intersect_points.end()))];

                // Use the left bin and its corresponding node to replace the right bin and its corresponding node
                // Involved indices in skeleton edges need to be modified
                // Even possible two bin size are the same, points in them should not be always same. Need to union two bin.

                int left_node = (neighbor_node < breakpoint_node) ? neighbor_node : breakpoint_node;
                int right_node = (neighbor_node < breakpoint_node) ? breakpoint_node : neighbor_node;
                
                for (int s = 0; s < bin[left_node].size(); s++) {
                    for (int t = 0; t < bin[right_node].size(); t++) {
                        if (IsSamePoint(bin[left_node][s], bin[right_node][t]) == true) {
                            bin[right_node].erase(bin[right_node].begin() + t);
                            break;
                        }
                    }
                }

                bin[left_node] += bin[right_node]; // The combined bin probably contains points in other bin_s. 
                indegree[left_node] += indegree[right_node];
                outdegree[left_node] += outdegree[right_node];
                Eigen::Vector4f new_centroid_v;
                pcl::compute3DCentroid(bin[left_node], new_centroid_v);
                skeletonPoints[left_node] = { new_centroid_v(0),new_centroid_v(1), new_centroid_v(2) };

                bin.erase(bin.begin() + right_node);
                indegree.erase(indegree.begin() + right_node);
                outdegree.erase(outdegree.begin() + right_node);
                skeletonPoints.erase(skeletonPoints.begin() + right_node);

                for (int s = 0; s < skeletonEdges.size(); s++) {
                    int v1 = std::get<0>(skeletonEdges[s]);
                    int v2 = std::get<1>(skeletonEdges[s]);

                    if (right_node == v1) {
                        v1 = left_node;
                    }
                    else {
                        if (right_node < v1) {
                            v1--;
                        }
                    }
                    if (right_node == v2) {
                        v2 = left_node;
                    }
                    else {
                        if (right_node < v2) {
                            v2--;
                        }
                    }
                    if (v1 != std::get<0>(skeletonEdges[s]) || v2 != std::get<1>(skeletonEdges[s])) {
                        skeletonEdges[s] = std::make_tuple(v1, v2);
                    }
                }

                //As indices changed, breakpoint node should adjust as well.
                for (int s = 0; s < breakpoints_node.size(); s++) {
                    if (right_node == breakpoints_node[s]) {
                        breakpoints_node[s] = left_node;
                    }
                    else {
                        if (right_node < breakpoints_node[s]) {
                            breakpoints_node[s] = breakpoints_node[s] - 1;
                        }
                    }
                }
            }
        }
        return breakpoints_cnt;
    }

    int BreakCircle(std::vector<int>& indegree, std::vector<int>& outdegree) {
        /* As the feature of single source shortest-path graph of branch point cloud and producing process of ISTTWN, 
        so-called circles only happen at the node that several directed edges point to at the same time. 
        The solution is to choose which father node is more probable to connect to by shortest-path distances.
        As different father node's bin must have differences in distances */
        std::vector<int> circle_nodes;
        for (int s = 0; s < indegree.size(); s++) {
            if (indegree[s] >= 2) {
                circle_nodes.push_back(s);
            }
        }
        if(circle_nodes.size() != 0)
            return 0;

        int circle_cnt = circle_nodes.size();

        while (!circle_nodes.empty()) {
            int circle_node = circle_nodes[0]; // Solve from front to end too
            circle_nodes.erase(circle_nodes.begin());

            std::vector<int> father_edges;
            for (int s = 0; s < skeletonEdges.size(); s++) {
                if (std::get<1>(skeletonEdges[s]) == circle_node) {
                    father_edges.push_back(s);
                }
            }

            std::vector<float> distance_difference(father_edges.size());
            for (int s = 0; s < father_edges.size(); s++) {
                int father_node = std::get<0>(skeletonEdges[father_edges[s]]);

                pcl::PointXYZ p_in_circle_node;
                pcl::PointXYZ p_in_father_node;

                float min_distance_interval = INT_MAX;
                
                pcl::search::KdTree<pcl::PointXYZ> kdtree;
                kdtree.setInputCloud(bin[father_node].makeShared());

                for (int u = 0; u < bin[circle_node].size(); u++) {
                    std::vector<int> k_indices;
                    std::vector<float> sqr_distances;
                    kdtree.nearestKSearch(bin[circle_node][u], 1, k_indices, sqr_distances);// Find point in father bin which is closest to current bin

                    if (sqrt(sqr_distances[0]) < min_distance_interval) {
                        min_distance_interval = sqrt(sqr_distances[0]);
                        p_in_circle_node = bin[circle_node][u];
                        p_in_father_node = bin[father_node][k_indices[0]];
                    }
                }

                float dist_p_in_circle_node = -1;
                float dist_p_in_father_node = -1;

                for (int i = 0; i < cloud->size(); i++) {
                    if (cloud->points[i].x == p_in_circle_node.x && cloud->points[i].y == p_in_circle_node.y && cloud->points[i].z == p_in_circle_node.z) {
                        dist_p_in_circle_node = vDist[i];
                    }
                    if (cloud->points[i].x == p_in_father_node.x && cloud->points[i].y == p_in_father_node.y && cloud->points[i].z == p_in_father_node.z) {
                        dist_p_in_father_node = vDist[i];
                    }
                    if (dist_p_in_circle_node >= 0 && dist_p_in_father_node >= 0) {
                        break;
                    }
                }

                distance_difference[s] = fabs(dist_p_in_circle_node - dist_p_in_father_node);
            }

            int true_father_idx = std::distance(distance_difference.begin(), std::max_element(distance_difference.begin(), distance_difference.end()));
            father_edges.erase(father_edges.begin() + true_father_idx); // The rest are erroneous edges

            std::sort(father_edges.begin(), father_edges.end());
            for (int s = 0; s < father_edges.size(); s++) {
                skeletonEdges.erase(skeletonEdges.begin() + father_edges[s] - s); //When removed a element, subsequent indices will entirely move forward. It is necessary to keep the deleting sequence in  sequential order.
            }
            
            indegree[circle_node] = 1;
        }
        return circle_cnt;
    }

    void EssentialProcedure() {
        /* Only remain one of same edges. It could happen in coincidence */
        for (int i = 0; i < skeletonEdges.size(); i++) {
            int v1 = std::get<0>(skeletonEdges[i]);
            int v2 = std::get<1>(skeletonEdges[i]);
            for (int j = i + 1; j < skeletonEdges.size(); j++) {
                int u1 = std::get<0>(skeletonEdges[j]);
                int u2 = std::get<1>(skeletonEdges[j]);
                if (u1 == v1 && u2 == v2) {
                    skeletonEdges.erase(skeletonEdges.begin() + j);
                    j--;
                }
            }
        }

        /* Change the node with indegree = 0 if it is not the first element */
        std::vector<int> indegree(skeletonPoints.size(), 0);
        std::vector<int> outdegree(skeletonPoints.size(), 0);
        for (int s = 0; s < skeletonEdges.size(); s++) {
            int v1 = std::get<0>(skeletonEdges[s]);
            int v2 = std::get<1>(skeletonEdges[s]);
            outdegree[v1]++;
            indegree[v2]++;
        }
        if (indegree[0] == 0)
            return;
        //Find the first node with indegree = 0 and consider it as the root node;
        int root_node = -1;
        for (int i = 1; i < indegree.size(); i++) {
            if (indegree[i] == 0) {
                root_node = i;
                break;
            }
        }
        if (root_node == -1) {
            PCL_ERROR("[ERROR] FATAL ERROR: No root node was produced.\n");
            return;
        }

        pcl::PointXYZ root_point = skeletonPoints[root_node];
        pcl::PointCloud<pcl::PointXYZ> root_bin = bin[root_node];
        skeletonPoints.erase(skeletonPoints.begin() + root_node);
        bin.erase(bin.begin() + root_node);
        skeletonPoints.insert(skeletonPoints.begin(), root_point);
        bin.insert(bin.begin(), root_bin);
        for (int i = 0; i < skeletonEdges.size(); i++) {
            int v1 = std::get<0>(skeletonEdges[i]);
            int v2 = std::get<1>(skeletonEdges[i]);
            if (v1 == root_node) {
                v1 = 0;
            }
            else {
                if (v1 < root_node) {
                    v1++;
                }
            }
            if (v2 == root_node) {
                v2 = 0;
            }
            else {
                if (v2 < root_node) {
                    v2++;
                }
            }
            skeletonEdges[i] = std::make_tuple(v1, v2);
        }
    }

    /* Iterative implementation and its multithreading improvement need this procedure */
    int fixed_breakpoint_count = 0;
    int fixed_circle_count = 0;

    void PostProcessing() {
        std::vector<int> indegree(skeletonPoints.size(), 0);
        std::vector<int> outdegree(skeletonPoints.size(), 0);

        for (int s = 0; s < skeletonEdges.size(); s++) {
            int v1 = std::get<0>(skeletonEdges[s]);
            int v2 = std::get<1>(skeletonEdges[s]);
            outdegree[v1]++;
            indegree[v2]++;
        }

        fixed_breakpoint_count = ConnectBreakpoint(indegree, outdegree);
        fixed_circle_count = BreakCircle(indegree, outdegree);
    }

public:
    ISTTWN(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<float> vDist, float fSearchRadius, float fStepLen, int iSteps, int iSpeed = 2, int iMinPts = 1) {
        this->cloud = cloud;
        this->vDist = vDist;
        this->fSearchRadius = fSearchRadius;
        this->fStepLen = fStepLen;
        this->iMinPts = iMinPts;

        std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ>> skeletonLines;
        std::vector<pcl::PointCloud<pcl::PointXYZ>> vBin;

        switch (iSpeed) {
        case 0: {
            pcl::PointIndices init_indices;
            for (int s = 0; s < cloud->size(); s++) {
                init_indices.indices.push_back(s);
            }
            float fMaxDist = *std::max_element(vDist.begin(), vDist.end());
            RecursionImpl(init_indices, 0.0, fMaxDist, { INT_MIN,INT_MIN,INT_MIN }, skeletonLines, vBin);
            break;
        }
        case 1: {
            float fMaxDist = *std::max_element(vDist.begin(), vDist.end());
            int iFlag = 0;
            IterativeImpl(0.0, fMaxDist, iSteps, skeletonLines, vBin, iFlag);
            break;
        }
        default: {
            IterativeMultithreading(iSteps, skeletonLines, vBin);
            break;
        }
        }

        SkeletonLines2SkeletonPointsAndSkeletonEdges(skeletonLines, this->skeletonPoints, this->skeletonEdges);

        bin.resize(skeletonPoints.size());

        // Original bin_s are very confused as vertices do not always equal to edges + 1. Although it seems that it needs too much computation when recalculating centroids and comparing, totally it still keeps a very short time cost
        while (!vBin.empty()) {
            pcl::PointCloud<pcl::PointXYZ> node_bin = vBin.back();
            if (node_bin.size() == 0) {
                vBin.pop_back();
                continue;
            }
            Eigen::Vector4f centroid_v;
            pcl::compute3DCentroid(node_bin, centroid_v);
            pcl::PointXYZ centroid = { centroid_v.x(), centroid_v.y(), centroid_v.z() };
            int idx = -1;
            for (int t = 0; t < skeletonPoints.size(); t++) {
                if (IsSamePoint(centroid, skeletonPoints[t])) {
                    idx = t;
                }
            }
            if (idx == -1) { //If not found, omit
                vBin.pop_back();
                continue;
            }
            bin[idx] = node_bin;
            vBin.pop_back();
        }
        //Check that if a bin is void, use the centroid instead.
        for (int i = 0; i < bin.size(); i++) {
            if (bin[i].size() == 0) {
                bin[i].push_back(skeletonPoints[i]);
            }
        }


        if (iSpeed != 0) {
            PostProcessing();
        }
        EssentialProcedure();
    }

    int getCntFixedBreakpoints(){
        return fixed_breakpoint_count;
    }
    int getCntFixedCircles() {
        return fixed_circle_count;
    }

    pcl::PointCloud<pcl::PointXYZ> getSkeletonPoints(){
        return this->skeletonPoints;
    }

    std::vector<std::tuple<int, int>> getSkeletonEdges() {
        return this->skeletonEdges;
    };

    std::vector<pcl::PointCloud<pcl::PointXYZ>> getBin() {
        return this->bin;
    }
};
pcl::PointCloud<pcl::PointXYZ>::Ptr ISTTWN::cloud(new pcl::PointCloud<pcl::PointXYZ>);
std::vector<float> ISTTWN::vDist;
float ISTTWN::fSearchRadius = 0.0;
float ISTTWN::fStepLen = 0.0;
int ISTTWN::iMinPts = 0;

// Skeleton refinement
int RootAdjustment(float, float, pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>&, std::vector<std::tuple<int, int>>&, std::vector<pcl::PointCloud<pcl::PointXYZ>>&, int);
void MainTrunkExtraction(std::vector<int> , pcl::PointCloud<pcl::PointXYZ>& , std::vector<std::tuple<int, int>>& , std::vector<std::vector<int>>& , std::vector<int>& , int =0, int=0);
void RemoveSmallOffshoots(int, float, int, pcl::PointCloud<pcl::PointXYZ>&, std::vector<std::tuple<int, int>>&, std::vector<pcl::PointCloud<pcl::PointXYZ>>&, std::vector<std::vector<int>>&, std::vector<int>&); //This procedure will automatically extract main trunk again at the end
void OptimizeBifurcations(int, float, float, int, float, pcl::PointCloud<pcl::PointXYZ>&, std::vector<std::tuple<int, int>>&, std::vector<pcl::PointCloud<pcl::PointXYZ>>&);
void CombineEdgesOfSameDirections(pcl::PointCloud<pcl::PointXYZ>&, std::vector<std::tuple<int, int>>&, std::vector<pcl::PointCloud<pcl::PointXYZ>>&);
void SmoothSkeleton(pcl::PointCloud<pcl::PointXYZ>&, std::vector<std::tuple<int, int>>&, std::vector<std::vector<int>>&, float, int);
int WriteSkeletonToOBJ(pcl::PointCloud<pcl::PointXYZ>& , std::vector<std::tuple<int, int>>&, std::string);

//Visualizer
void ViewBranchCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr);
void ViewSkeleton(pcl::PointCloud<pcl::PointXYZ>::Ptr , pcl::PointCloud<pcl::PointXYZ>&, std::vector<std::tuple<int, int>>&, std::vector<pcl::PointCloud<pcl::PointXYZ>>&, bool,bool);
void ViewOffshoots(pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>&, std::vector<std::tuple<int, int>>&, std::vector<pcl::PointCloud<pcl::PointXYZ>>&, std::vector<std::vector<int>>&, std::vector<int>&, int, bool);

int main(int argc, char** argv) {
    printf("Incomplete Simulation of Tree Transmitting Water and Nutrients\nA fast tool for reconstructing the tree skeleton from a tree branch point cloud.\nAuthor: J. Yang; Current version: 1.0 [Release]\n[Attention] It is recommended to use a point cloud with uniform point distribution\n\n");

    std::string strBranchCloudFilePath;
    std::string strDistFilePath;
    float fStepLen = -1;
    float fStepLenMagnification = 1.0;
    if (pcl::console::parse(argc, argv, "-i", strBranchCloudFilePath) < 0) {
        PrintUsage(argv[0]);
        return -1;
    }
    if (pcl::console::parse(argc, argv, "-d", strDistFilePath) < 0) {
        PrintUsage(argv[0]);
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<float> vDist;

    OpenPCD(strBranchCloudFilePath, *cloud);
    OpenDist(strDistFilePath, vDist);

    if (vDist.size() != cloud->size()) {
        PCL_ERROR("[ERROR]Invalid file that saves shortest-path distances.\n");
        return(-1);
    }

    PCL_INFO("[INFO] Initializing... Please wait.\n");
    float fMinInterval = ComputeMinInterval(cloud);
    float fSearchRadius = fMinInterval + 0.000001;

    if (pcl::console::parse(argc, argv, "-sm", fStepLenMagnification) >= 0) {
        if (fStepLenMagnification <= 1) {
            PCL_WARN("[WARNING]Invalid magnification of step length, use 1.0x minimum connected interval+epsilon.\n");
            fStepLenMagnification = -1;
        }
    }
    if (pcl::console::parse(argc, argv, "-sf", fStepLen) >= 0) {
        if (fStepLen <= fMinInterval + 0.000001) {
            PCL_WARN("[WARNING] Too short step length. Change to 1.0x minimum connected interval by default\n");
            fStepLen = -1;
        }
    }

    fStepLen = fmax(fStepLen, fmax(fStepLenMagnification * fMinInterval + 0.000001, fMinInterval + 0.000001));

    int iStep = 1;

    DWORD time_start, time_end;
    PCL_INFO("[INFO] Producing the skeleton by ISTTWN iteration edition with multithreading.\n\tParameters: Search radius = %f, step length = %f, %d step(s) each interval segment(NOT INCLUDE NEXT INTERVAL HEADER),  %d thread(s).\n", fSearchRadius, fStepLen, iStep, std::thread::hardware_concurrency());
    time_start = GetTickCount64();
    ISTTWN* isttwn(new ISTTWN(cloud, vDist, fSearchRadius, fStepLen,iStep,2));
    time_end = GetTickCount64();
    PCL_INFO("[INFO] The initial skeleton was produced in %f sec(s).\n", (float)(time_end - time_start) / 1000);

    std::pair<int, int> post_processing_result = std::pair<int,int>(isttwn->getCntFixedBreakpoints(),isttwn->getCntFixedCircles());
    PCL_INFO("[INFO] Postprocessing succeed in fixing %d breakpoint(s) and %d circle(s).\n", post_processing_result.first, post_processing_result.second);
    pcl::PointCloud<pcl::PointXYZ> skeletonPoints = isttwn->getSkeletonPoints();
    std::vector<std::tuple<int, int>> skeletonEdges = isttwn->getSkeletonEdges();
    std::vector<pcl::PointCloud<pcl::PointXYZ>> bin = isttwn->getBin();
    delete isttwn;

    std::vector<std::vector<int>> edgesInOffshoots;
    std::vector<int> offShootsLevel;
    MainTrunkExtraction(std::vector<int>(), skeletonPoints, skeletonEdges, edgesInOffshoots, offShootsLevel);
    PCL_INFO("[INFO] Current skeleton has %d skeleton point(s) and %d skeleton edge(s). Totally %d offshoots.\n", skeletonPoints.size(), skeletonEdges.size(), edgesInOffshoots.size());

    int iMinPts = 1;
    if (pcl::console::parse(argc, argv, "-m", iMinPts) >= 0) {
        if (iMinPts < 1) {
            PCL_WARN("[WARNING]Invalid MinPts. Use 1 by default.\n");
            iMinPts = 1;
        }
    }
    if (iMinPts >= 2) {
        PCL_INFO("[INFO] Removing the skeleton points with the bin size less than %d\n",iMinPts);
        time_start = GetTickCount64();
        RemoveSmallOffshoots(iMinPts, 0, 0, skeletonPoints, skeletonEdges, bin, edgesInOffshoots, offShootsLevel);
        time_end = GetTickCount64();
        PCL_INFO("[INFO] Finished removal in %f sec(s)\n", (float)(time_end - time_start)/1000);
        PCL_INFO("[INFO] Current skeleton has %d skeleton point(s) and %d skeleton edge(s). Totally %d offshoots.\n", skeletonPoints.size(), skeletonEdges.size(), edgesInOffshoots.size());
    }

    bool bEnableOpt = false;
    if (pcl::console::find_argument(argc, argv, "-o") >= 0) {
        bEnableOpt = true;
    }

    int iMinEdges = 1;
    if (pcl::console::parse(argc, argv, "-b", iMinEdges) >= 0) {
        if (iMinEdges >= 2 && bEnableOpt == false) {
            PCL_INFO("[INFO] Removing the offshoot with less than %d lines.\n", iMinEdges);
            time_start = GetTickCount64();
            RemoveSmallOffshoots(0, 0, iMinEdges, skeletonPoints, skeletonEdges, bin, edgesInOffshoots, offShootsLevel);
            time_end = GetTickCount64();
            PCL_INFO("[INFO] Finished removal in %f sec(s).\n", (float)(time_end - time_start) / 1000);
            PCL_INFO("[INFO] Current skeleton has %d skeleton point(s) and %d skeleton edge(s). Totally %d offshoots.\n", skeletonPoints.size(), skeletonEdges.size(), edgesInOffshoots.size());
        }
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
    bool bOverlayVisualization = false;
    if (pcl::console::find_argument(argc, argv, "-vo") >= 0) {
        bOverlayVisualization = true;
    }
    bool bEnableDisplay = false;
    if (pcl::console::find_argument(argc, argv, "-ve") >= 0) {
        bEnableDisplay = true;
    }
    bool bOnlyViewSkeleton = false;
    if (pcl::console::find_argument(argc, argv, "-vs") >= 0) {
        bOnlyViewSkeleton = true;
    }

    if (bEnableDisplay == true) {
        ViewBranchCloud(cloud);
        ViewSkeleton(cloud, skeletonPoints, skeletonEdges, bin, bOverlayVisualization, false);
        if (bOnlyViewSkeleton == false || bOverlayVisualization == true) {
            ViewSkeleton(cloud, skeletonPoints, skeletonEdges, bin, true, bOverlayVisualization);
            PCL_INFO("[INFO] Because of the postprocessing, some bin_s will not display normally.\n");
            ViewOffshoots(cloud, skeletonPoints, skeletonEdges, bin, edgesInOffshoots, offShootsLevel, 2, bOverlayVisualization);
            ViewOffshoots(cloud, skeletonPoints, skeletonEdges, bin, edgesInOffshoots, offShootsLevel, 1, bOverlayVisualization);
            ViewOffshoots(cloud, skeletonPoints, skeletonEdges, bin, edgesInOffshoots, offShootsLevel, 0, bOverlayVisualization);
        }
    }


    if (bEnableOpt == true) {
        float fRootStepLen = 0.0;
        for (int i = 0; i < skeletonEdges.size(); i++) {
            float dist = GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[i])], skeletonPoints[std::get<1>(skeletonEdges[i])]);
            fRootStepLen += dist;
        }
        fRootStepLen /= skeletonEdges.size(); //Get average skeleton length;

        float root_magnification = 1.0;
        if (pcl::console::parse(argc, argv, "-rm", root_magnification) >= 0) {
            if (root_magnification * fRootStepLen < fMinInterval + 0.000001) {
                PCL_WARN("[WARN] Magnification is too small. Use default config.\n");
            }
        }
        fRootStepLen = fmax(fRootStepLen * root_magnification, fMinInterval + 0.000001);

        PCL_INFO("[INFO] Adjusting the skeleton near the root with interval length %f...\n", fRootStepLen);
        time_start = GetTickCount64();
        int iRootAdjustmentStatus = RootAdjustment(fRootStepLen, fSearchRadius, cloud, skeletonPoints, skeletonEdges, bin, iMinPts);
        time_end = GetTickCount64();
        if (iRootAdjustmentStatus == -1) {
            PCL_WARN("[WARNING] Reconstructing the skeleton at root failed. The skeleton will remain the initial appearance.\n");
        }
        else {
            PCL_INFO("[INFO] Finished adjustment in %f sec(s).\n", (float)(time_end - time_start) / 1000);
        }
        edgesInOffshoots.clear();
        offShootsLevel.clear();
        MainTrunkExtraction(std::vector<int>(), skeletonPoints, skeletonEdges, edgesInOffshoots, offShootsLevel);
        PCL_INFO("[INFO] Current skeleton has %d skeleton point(s) and %d skeleton edge(s). Totally %d offshoots.\n", skeletonPoints.size(), skeletonEdges.size(), edgesInOffshoots.size());


        PCL_INFO("[INFO] Optimizing bifurcations...\n");
        time_start = GetTickCount64();
        
        float fAffectRange = 3 * fStepLen;
        if (fAffectRange > 0.5) {
            fAffectRange = 0.5;
        }

        OptimizeBifurcations(0, 1.0 / 5 * fStepLen, fAffectRange, iMinPts, 0, skeletonPoints, skeletonEdges, bin);
        PCL_INFO("[INFO] Combining...\n");
        CombineEdgesOfSameDirections(skeletonPoints, skeletonEdges, bin);
        time_end = GetTickCount64();
        PCL_INFO("[INFO] Finished optimization in %f sec(s).\n", (float)(time_end - time_start) / 1000);

        edgesInOffshoots.clear();
        offShootsLevel.clear();
        MainTrunkExtraction(std::vector<int>(), skeletonPoints, skeletonEdges, edgesInOffshoots, offShootsLevel);
        PCL_INFO("[INFO] Current skeleton has %d skeleton point(s) and %d skeleton edge(s). Totally %d offshoots.\n", skeletonPoints.size(), skeletonEdges.size(), edgesInOffshoots.size());

        if (iMinEdges >= 2) {
            PCL_INFO("[INFO] Removing the offshoot with less than %d lines.\n", iMinEdges);
            time_start = GetTickCount64();
            
            RemoveSmallOffshoots(0, 0, iMinEdges, skeletonPoints, skeletonEdges, bin, edgesInOffshoots, offShootsLevel);
            time_end = GetTickCount64();
            PCL_INFO("[INFO] Finished removal in %f sec(s).\n", (float)(time_end - time_start) / 1000);
            PCL_INFO("[INFO] Current skeleton has %d skeleton point(s) and %d skeleton edge(s). Totally %d offshoots.\n", skeletonPoints.size(), skeletonEdges.size(), edgesInOffshoots.size());
        }

        int iTurn = 1;
        float fLambda = -1;
        if (pcl::console::parse(argc, argv, "-t", iTurn) >= 0) {
            if (iTurn < 1) {
                iTurn = 1;
            }
        }
        if (pcl::console::parse(argc, argv, "-l", fLambda) >= 0) {
            if (!(fLambda > 0.0 && fLambda <= 1.0)) {
                fLambda = -1;
            }
        }
        PCL_INFO("[INFO] Smoothing the skeleton");
        if (fLambda > 0.0 && fLambda <= 1.0) {
            PCL_INFO("by coefficient smoothing. Lambda:%f, Turn:%d\n", fLambda, iTurn);
        }
        else {
            PCL_INFO("by three-point average. Turn:%d\n", iTurn);
        }

        time_start = GetTickCount64();
        SmoothSkeleton(skeletonPoints, skeletonEdges, edgesInOffshoots, fLambda, iTurn);
        time_end = GetTickCount64();
        PCL_INFO("[INFO] Finished smoothness in %f sec(s).\n", (float)(time_end - time_start) / 1000);

        if (bEnableDisplay == true) {
            ViewSkeleton(cloud, skeletonPoints, skeletonEdges, bin, bOverlayVisualization, false);
        }
    }

    std::string strInputFileDir, strInputFilename;
    SubstrFromPath(strBranchCloudFilePath, strInputFileDir, strInputFilename);
    std::string strOutputObj = strInputFilename.substr(0, strInputFilename.rfind(".")) + "_ske.obj";

    int result = WriteSkeletonToOBJ(skeletonPoints, skeletonEdges, strInputFileDir + strOutputObj);
    if (result == 1) {
        PCL_INFO("[INFO] Succeed in saving the skeleton to %s", (strInputFileDir + strOutputObj).c_str());
    }
    else {
        PCL_ERROR("[ERROR] Failed in saving the skeleton.\n");
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

void PrintUsage(const char* progName) {
    printf("Usage: %s -i FILEPATH -d FILEPATH [-sm STEPLEN_MAGNIFICATION | -sf FIXED_STEPLEN] [-o [-m MINPTS] [-rm STEPLEN_MAGNIFICATION] [-b CNT] [-l LAMBDA] [-t TURN]] [-vo] [-ve | -vs] [-vp FILEPATH [-va]]\n\
Options:\n\
-i FILEPATH\t\tInput tree BRANCH point cloud file(.pcd);\n\
-d FILEPATH\t\tInput file that saves SHORTEST-PATH distances gotten by the graph fromed by tree branch cloud\n\
-sm STEPLEN_MAGNIFICATION\tMagnification of step length for extracting the skeleton(default: 1.0x minimum connected interval);\n\
-sf FIXED_STEPLEN\tSpecify a fixed step length;\n\
-o\t\t\tEnable skeleton optimization (refinement);\n\
-m MINPTS\t\tCluster tolerance(default: 1)\n\
-rm STEPLEN_MAGNIFICATION\tMagnification of average skeleton line length for adjusting the skeleton at the root(If not specified, the procedure will not work. This function is applicable to the branch cloud which has removed the part of the stump of root / root collar / root crown);\n\
-b CNT\t\t\tThe minimum lines in the terminal offshoots;\n\
-l LAMBDA\t\tSmoothing coefficient(default use the average of three points)(If used, it is better to set it to 0.1);\n\
-t TURN\t\t\tTurn of Smoothing iteration(default 1)(If used, it is better to set it to 5~6);\n\
-vo\t\t\tDisplay overlay results;\n\
-ve\t\t\tEnable viewer(display every effect);\n\
-vs\t\t\tOnly view skeletons; \n\
-vp FILEPATH\t\tImport a camera parameters' file(.txt)(if not exists, use default parameters);\n\
-va\t\t\tAuto writing camera parameters to file.\n\
Output skeleton will save to *_ske.obj in the same directory of input tree BRANCH point cloud.\n", progName);
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
void OpenPCD(std::string strFilePath, pcl::PointCloud<pcl::PointXYZ>& pcCloud) {
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(strFilePath, pcCloud) == -1) {
        PCL_ERROR("[ERROR]Cloud not read file %s\n", strFilePath);
        exit(-1);
    }
}
int OpenDist(std::string strFilePath, std::vector<float>& vDist) {
    std::fstream file(strFilePath, std::ios::in);
    if (file.fail() == true)
        return -1;

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

            float fDist = std::stof(buf_str); //If not a number or out_of_range will raise exception

            vDist.push_back(fDist);
        }
    }
    catch (const std::exception&) {
        file.close();
        return -1;
    }
    file.close();
    return 0;
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
            PCL_ERROR("[ERROR] MST Failed! Current MST only has %d edge(s)(should have %d edge(s)).\n ", etotal, vecAdjacentTable.size() - 1);
        }
    }
    std::vector<std::tuple<int, int>> GetMST() {
        return mst;
    }
    std::vector<double> GetWeights() {
        return weights;
    }
};

float GetDistanceBetween2pts(pcl::PointXYZ p1, pcl::PointXYZ p2) {
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
}

bool IsSamePoint(pcl::PointXYZ p1, pcl::PointXYZ p2) {
    return (p1.x == p2.x) && (p1.y == p2.y) && (p1.z == p2.z);
}

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

int OpenCameraFile(std::string strFilePath) {
    std::fstream file(strFilePath, std::ios::in);
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
    std::ofstream ofs;
    ofs.open(strFilePath, std::ios::out);
    if (ofs.is_open() == false) {
        return -1;
    }
    ofs << pos_x << std::endl << pos_y << std::endl << pos_z << std::endl\
        << view_x << std::endl << view_y << std::endl << view_z << std::endl\
        << up_x << std::endl << up_y << std::endl << up_z;
    ofs.close();
    return 0;
};

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

class XuEtAl {
private:
    pcl::PointCloud<pcl::PointXYZ> skeletonPoints;
    std::vector<std::tuple<int, int>> skeletonEdges;
    std::vector<pcl::PointCloud<pcl::PointXYZ>> bin;

    void Correct(std::vector<bool>& flagEdgeChecked, int iBeginNode = 0) {
        std::vector<int> nextNodes;
        for (int i = 0; i < skeletonEdges.size(); i++) {
            if (flagEdgeChecked[i] == true)
                continue;

            if (std::get<0>(skeletonEdges[i]) == iBeginNode) {
                nextNodes.push_back(std::get<1>(skeletonEdges[i]));
                flagEdgeChecked[i] = true;
            }

            if (std::get<1>(skeletonEdges[i]) == iBeginNode) {
                skeletonEdges[i] = std::make_tuple(std::get<1>(skeletonEdges[i]), std::get<0>(skeletonEdges[i]));
                nextNodes.push_back(std::get<1>(skeletonEdges[i]));
                flagEdgeChecked[i] = true;
            }
        }
        for (auto node : nextNodes) {
            Correct(flagEdgeChecked, node);
        }
    }

public:
    XuEtAl(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<float> vLayerRefer, float fSearchRadius, int iLayers,int iMinClusterSize, float fLayerLength = -1, bool bEnableLayerRelation =false) {
        float fMinLayerRefer = *std::min_element(vLayerRefer.begin(), vLayerRefer.end());
        float fMaxLayerRefer = *std::max_element(vLayerRefer.begin(), vLayerRefer.end());

        if (fLayerLength <= 0.0) {
            fLayerLength = (fMaxLayerRefer - fMinLayerRefer) / iLayers;
        }
        else {
            iLayers = (int)ceil((fMaxLayerRefer - fMinLayerRefer) / fLayerLength);
        }

        std::vector<pcl::PointCloud<pcl::PointXYZ>> bins(iLayers);
        for (int i = 0; i < cloud->size(); i++) {
            int layerNo = (int)fmin((int)((vLayerRefer[i] - fMinLayerRefer) / fLayerLength),iLayers - 1); //The highest point should not be at a single layer
            bins[layerNo].push_back(cloud->points[i]);
        }

        std::vector<int> layerNoRecord;
        for (int i = 0; i < bins.size(); i++) {
            ECE* ece(new ECE(bins[i], fSearchRadius, iMinClusterSize));
            std::vector<pcl::PointIndices> inlier = ece->GetIndices();
            for (int j = 0; j < inlier.size(); j++) {
                pcl::PointCloud<pcl::PointXYZ> a_bin;
                pcl::copyPointCloud(bins[i], inlier[j], a_bin);
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(a_bin, centroid);
                skeletonPoints.push_back({ centroid(0), centroid(1), centroid(2) });
                bin.push_back(a_bin);
                layerNoRecord.push_back(i);
            }
        }

        std::vector<std::vector<std::tuple<int, double>>> adjacentTable(skeletonPoints.size());

        if (bEnableLayerRelation == false) {
            /* Find the farest interval */
            float fMaxResolution = ComputeMinInterval(skeletonPoints.makeShared());
            /* Create neighborhood graph */
            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            kdtree.setInputCloud(skeletonPoints.makeShared());
            for (int i = 0; i < skeletonPoints.size(); i++) {
                std::vector<int> pointIdxRadiusSearch;
                std::vector<float> pointRadiusSquaredDistance;
                kdtree.radiusSearch(i, fMaxResolution + 0.000001, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                for (int j = 0; j < pointIdxRadiusSearch.size(); j++) {
                    if (i < pointIdxRadiusSearch[j]) {
                        adjacentTable[i].push_back(std::make_tuple(pointIdxRadiusSearch[j], sqrt(pointRadiusSquaredDistance[j])));
                    }
                }
            }
        }
        else {
            //Two bin_s in adjacent layers within minimum interval less than search radius should connect
            for (int i = 0; i < skeletonPoints.size(); i++) {
                for (int j = 0; j < i; j++) {
                    if (layerNoRecord[j] != layerNoRecord[i] - 1)
                        continue;
                    float min_interval = INT_MAX;

                    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
                    kdtree.setInputCloud(bin[j].makeShared());

                    for (int k = 0; k < bin[i].size(); k++) {
                        std::vector<int> pointIdxNeighborSearch;
                        std::vector<float> pointNeighborSquaredDistance;
                        kdtree.nearestKSearch(bin[i][k], 1, pointIdxNeighborSearch, pointNeighborSquaredDistance);
                        min_interval = fmin(min_interval, sqrt(pointNeighborSquaredDistance[0]));
                    }
                    /*
                    for (int m = 0; m < bin[i].size(); m++) {
                        for (int n = 0; n < bin[j].size(); n++) {
                            min_interval = fmin(min_interval, GetDistanceBetween2pts(bin[i][m], bin[j][n]));
                        }
                    }*/
                    if (min_interval < fSearchRadius) {
                        adjacentTable[j].push_back(std::make_tuple(i, GetDistanceBetween2pts(skeletonPoints[i], skeletonPoints[j])));
                    }
                }
            }
        }

        /* MST */
        MST_with_AdjacentTable_Kruskal* mst(new MST_with_AdjacentTable_Kruskal(adjacentTable));
        skeletonEdges = mst->GetMST();

        /* Give direction */
        //As the first point must be at the real root in main branch(ECE is automatically sorted by size), use it as the root of graph
        std::vector<bool> flagEdgeChecked(skeletonEdges.size(), false);
        Correct(flagEdgeChecked);
    }

    pcl::PointCloud<pcl::PointXYZ> getSkeletonPoints() {
        return skeletonPoints;
    }
    std::vector<std::tuple<int, int>> getSkeletonEdges() {
        return skeletonEdges;
    }
    std::vector<pcl::PointCloud<pcl::PointXYZ>> getBin() {
        return bin;
    }
};

int RootAdjustment(float fStepLen, float fSearchRadius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges, std::vector<pcl::PointCloud<pcl::PointXYZ>>& bin, int iMinPts = 1) {
    float z_min = INT_MAX;
    float z_max = INT_MIN;

    for (int s = 0; s < cloud->size(); s++) { // First bin must exist the point with the lowest hight
        z_min = fmin(z_min, cloud->points[s].z);
        z_max = fmax(z_max, cloud->points[s].z);
    }

    int iMagnification = -1;
    if (1.3 / 5 * (z_max - z_min) < 1.3) {
        while ((iMagnification + 1) * fStepLen <= 1.3 / 5 * (z_max - z_min)) {
            iMagnification++;
        }
    }
    else {
        while ((iMagnification + 1) * fStepLen <= 1.3) {
            iMagnification++;
        }
    }

    float h_base = iMagnification * fStepLen + z_min;
    // Select a base height plane that there is the same number of clusters both in intervals obtained by adding and subtracting a step length of the height (or in other words, in two continuous intervals with same number of clusters), and the skeleton edges passing the plane equals it as well.
    while (true) {
        if (h_base <= fStepLen + z_min) {
            return -1; /* It may happen under 4 circumstances:
                       1) step length is too large, if step length is equal to the original, center deviation at the root is not obvious;
                       2) the target tree species could be a shrub that has many continuous bifurcations;
                       3) the quality of point cloud is bad;
                       4) No ground correction has been done */
        }

        pcl::PointCloud<pcl::PointXYZ> submaincloud;
        std::vector<int> need_delete_nodes;

        pcl::PointCloud<pcl::PointXYZ> subcloud;
        for (int s = 0; s < cloud->size(); s++) {
            if (cloud->points[s].z < h_base + fStepLen && cloud->points[s].z >= z_min) {
                subcloud.push_back(cloud->points[s]);
            }
        }

        /* Find main branch*/
        ECE* ece(new ECE(subcloud, fSearchRadius, 1));
        std::vector<pcl::PointIndices> inlier = ece->GetIndices();
        std::vector<int> vCntIntersectPoints(inlier.size(), 0);
        for (int s = 0; s < inlier.size(); s++) {
            for (int t = 0; t < inlier[s].indices.size(); t++) {
                for (int u = 0; u < bin[0].size(); u++) {
                    if (IsSamePoint(bin[0][u], subcloud[inlier[s].indices[t]])) {
                        vCntIntersectPoints[s]++;
                        break;
                    }
                }
            }
        }
        pcl::copyPointCloud(subcloud, inlier[std::distance(vCntIntersectPoints.begin(), std::max_element(vCntIntersectPoints.begin(), vCntIntersectPoints.end()))], submaincloud);


        pcl::PointCloud<pcl::PointXYZ> h_base_beyond_1;
        pcl::PointCloud<pcl::PointXYZ> h_base_below_1;
        for (int s = 0; s < submaincloud.size(); s++) {
            if (submaincloud.points[s].z >= h_base - fStepLen && submaincloud.points[s].z < h_base) {
                h_base_below_1.push_back(submaincloud.points[s]);
            }
            if (submaincloud.points[s].z >= h_base && submaincloud.points[s].z < h_base + fStepLen) {
                h_base_beyond_1.push_back(submaincloud.points[s]);
            }
        }
        ECE* ece1(new ECE(h_base_below_1, fSearchRadius, 1));
        int iCntClustersBelow1 = ece1->GetIndices().size();
        delete ece1;

        ECE* ece2(new ECE(h_base_beyond_1, fSearchRadius, 1));
        int iCntClustersBeyond1 = ece2->GetIndices().size();
        delete ece2;

        if (iCntClustersBelow1 != iCntClustersBeyond1) {
            h_base -= fStepLen;
            continue;
        }
        for (int s = 0; s < submaincloud.size(); s++) {
            if (submaincloud.points[s].z >= h_base && submaincloud.points[s].z < h_base + fStepLen) {
                submaincloud.erase(submaincloud.begin() + s);
                s--;
            }
        }
        for (int s = 0; s < bin.size(); s++) {
            bool bNeedDelete = false;
            //a skeleton node should be reconstruct if its bin has intersect part with the main branch and its height cannot exceed the height of base.
            for (int t = 0; t < bin[s].size(); t++) {
                for (int u = 0; u < submaincloud.size(); u++) {
                    if (IsSamePoint(bin[s][t], submaincloud[u]) == true && skeletonPoints[s].z < h_base) {
                        bNeedDelete = true;
                        break;
                    }
                }
                if (bNeedDelete == true) {
                    break;
                }
            }
            if (bNeedDelete == true) {
                need_delete_nodes.push_back(s);
            }
        }
        int iCntPassEdges = 0;
        bool bExistReverse = false;
        for (int s = 0; s < skeletonEdges.size(); s++) {
            int v1 = std::get<0>(skeletonEdges[s]);
            int v2 = std::get<1>(skeletonEdges[s]);
            if (std::find(need_delete_nodes.begin(), need_delete_nodes.end(), v1) != need_delete_nodes.end()
                && std::find(need_delete_nodes.begin(), need_delete_nodes.end(), v2) == need_delete_nodes.end()) {
                iCntPassEdges++;
            }
            if (std::find(need_delete_nodes.begin(), need_delete_nodes.end(), v1) == need_delete_nodes.end()
                && std::find(need_delete_nodes.begin(), need_delete_nodes.end(), v2) != need_delete_nodes.end()) {
                bExistReverse = true;
                break;
            }
        }
        if (iCntPassEdges != iCntClustersBelow1 || bExistReverse == true) {
            h_base -= fStepLen;
            continue;
        }

        std::vector<float> vHeightsCalibrated(submaincloud.size());
        for (int s = 0; s < submaincloud.size(); s++) {
            vHeightsCalibrated[s] = submaincloud[s].z - z_min;
        }
        pcl::PointCloud<pcl::PointXYZ> skeletonPointsByHeights;
        std::vector<std::tuple<int, int>> skeletonEdgesByHeights;
        std::vector<pcl::PointCloud<pcl::PointXYZ>> binByHeights;

        /*
        ISTTWN* isttwn_by_heights(new ISTTWN(submaincloud.makeShared(), vHeightsCalibrated, fSearchRadius, fStepLen, 1, 0, iMinPts));

        skeletonPointsByHeights = isttwn_by_heights->getSkeletonPoints();
        skeletonEdgesByHeights = isttwn_by_heights->getSkeletonEdges();
        binByHeights = isttwn_by_heights->getBin();*/

        XuEtAl* xuetal(new XuEtAl(submaincloud.makeShared(), vHeightsCalibrated, fSearchRadius, -1, iMinPts, fStepLen,true));
        skeletonPointsByHeights = xuetal->getSkeletonPoints();
        skeletonEdgesByHeights = xuetal->getSkeletonEdges();
        binByHeights = xuetal->getBin();

        if (skeletonPointsByHeights.size() == 0) {
            h_base -= fStepLen;
            continue;
        }

        std::vector<int> indegreeByHeights(skeletonPointsByHeights.size(), 0);
        std::vector<int> outdegreeByHeights(skeletonPointsByHeights.size(), 0);

        for (int s = 0; s < skeletonEdgesByHeights.size(); s++) {
            int v1 = std::get<0>(skeletonEdgesByHeights[s]);
            int v2 = std::get<1>(skeletonEdgesByHeights[s]);
            outdegreeByHeights[v1]++;
            indegreeByHeights[v2]++;
        }

        std::vector<int> nodeOutdegree0ByHeights;
        std::vector<std::tuple<int, float, int>> distToHBase;
        for (int i = 0; i < skeletonPointsByHeights.size(); i++) {
            float minDist = INT_MAX;
            for (int j = 0; j < binByHeights[i].size(); j++) {
                minDist = fmin(minDist, fabs(binByHeights[i][j].z - h_base));
            }
            distToHBase.push_back(std::make_tuple(i, minDist, outdegreeByHeights[i]));
        }
        std::sort(distToHBase.begin(), distToHBase.end(), [](const auto& e1, const auto& e2) {
            return std::get<1>(e1) < std::get<1>(e2);
            });
        for (int i = 0; i < distToHBase.size(); i++) {
            if (std::get<2>(distToHBase[i]) != 0) {
                continue;
            }

            nodeOutdegree0ByHeights.push_back(std::get<0>(distToHBase[i]));
            if (nodeOutdegree0ByHeights.size() >= iCntPassEdges) {
                break;
            }
        }

        pcl::PointCloud<pcl::PointXYZ> margin_cloud;

        while (!need_delete_nodes.empty()) {
            int node = need_delete_nodes[0];

            for (int t = 0; t < bin[node].size(); t++) {
                if (bin[node][t].z >= h_base) {
                    margin_cloud.push_back(bin[node][t]);
                }
            }

            need_delete_nodes.erase(need_delete_nodes.begin());
            skeletonPoints.erase(skeletonPoints.begin() + node);
            bin.erase(bin.begin() + node);
            for (int s = 0; s < skeletonEdges.size(); s++) {
                int v1 = std::get<0>(skeletonEdges[s]);
                int v2 = std::get<1>(skeletonEdges[s]);
                if (v1 == node || v2 == node) {
                    skeletonEdges.erase(skeletonEdges.begin() + s);
                    s--;
                    continue;
                }
                if (v1 > node) {
                    v1--;
                }
                if (v2 > node) {
                    v2--;
                }
                skeletonEdges[s] = std::make_tuple(v1, v2);
            }
            for (int s = 0; s < need_delete_nodes.size(); s++) {
                if (need_delete_nodes[s] > node) {
                    need_delete_nodes[s] --;
                }
            }
        }
        std::vector<int> indegree(skeletonPoints.size(), 0);
        std::vector<int> outdegree(skeletonPoints.size(), 0);
        for (int s = 0; s < skeletonEdges.size(); s++) {
            int v1 = std::get<0>(skeletonEdges[s]);
            int v2 = std::get<1>(skeletonEdges[s]);
            outdegree[v1]++;
            indegree[v2]++;
        }

        std::vector<int> nodeIndegree0;
        for (int i = 0; i < indegree.size(); i++) {
            if (indegree[i] == 0) {
                nodeIndegree0.push_back(i);
            }
        }

        for (int s = 0; s < bin.size(); s++) {
            for (int t = 0; t < bin[s].size(); t++) {
                if (bin[s][t].z < h_base) {
                    bin[s].erase(bin[s].begin() + t);
                    t--;
                }
            }
        }
        
        // Build mapping
        std::vector<std::tuple<int, int>> append;
        for (int s = 0; s < nodeOutdegree0ByHeights.size(); s++) {
            float min_interval = INT_MAX;
            int to = -1;
            for (int t = 0; t < binByHeights[nodeOutdegree0ByHeights[s]].size(); t++) {
                for (int u = 0; u < nodeIndegree0.size(); u++) {
                    for (int v = 0; v < bin[nodeIndegree0[u]].size(); v++) {
                        float dist = GetDistanceBetween2pts(binByHeights[nodeOutdegree0ByHeights[s]][t], bin[nodeIndegree0[u]][v]);
                        if (dist < min_interval) {
                            min_interval = dist;
                            to = u;
                        }
                    }
                }
            }
            append.push_back(std::make_tuple(nodeOutdegree0ByHeights[s], nodeIndegree0[to] + skeletonPointsByHeights.size()));// New nodes are inserted into head.
        }
        skeletonEdgesByHeights.insert(skeletonEdgesByHeights.end(), append.begin(), append.end());


        ECE* ece3(new ECE(margin_cloud, fSearchRadius, 1));
        std::vector<pcl::PointIndices> inlier3 = ece3->GetIndices();
        delete(ece3);
        while (!inlier3.empty()) {
            pcl::PointCloud<pcl::PointXYZ> inlier_cloud;
            pcl::copyPointCloud(margin_cloud, inlier3[0], inlier_cloud);
            inlier3.erase(inlier3.begin());

            std::vector<float> cluster_dist(nodeOutdegree0ByHeights.size());
            for (int s = 0; s < nodeOutdegree0ByHeights.size(); s++) {
                float min_dist = INT_MAX;
                for (int t = 0; t < binByHeights[nodeOutdegree0ByHeights[s]].size(); t++) {
                    for (int u = 0; u < inlier_cloud.size(); u++) {
                        float dist = GetDistanceBetween2pts(binByHeights[nodeOutdegree0ByHeights[s]][t], inlier_cloud[u]);
                        if (min_dist > dist) {
                            min_dist = dist;
                        }
                    }
                }
                cluster_dist[s] = min_dist;
            }

            int idx = std::distance(cluster_dist.begin(), std::min_element(cluster_dist.begin(), cluster_dist.end()));
            binByHeights[nodeOutdegree0ByHeights[idx]] += inlier_cloud;
        }

        for (int s = 0; s < skeletonEdges.size(); s++) {
            int v1 = std::get<0>(skeletonEdges[s]);
            int v2 = std::get<1>(skeletonEdges[s]);
            skeletonEdges[s] = std::make_tuple(v1 + skeletonPointsByHeights.size(), v2 + skeletonPointsByHeights.size());
        }

        skeletonPoints.insert(skeletonPoints.begin(), skeletonPointsByHeights.begin(), skeletonPointsByHeights.end());
        skeletonEdges.insert(skeletonEdges.begin(), skeletonEdgesByHeights.begin(), skeletonEdgesByHeights.end());
        bin.insert(bin.begin(), binByHeights.begin(), binByHeights.end());

        break;
    }
    return 0;
}

void ViewSkeleton(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges, std::vector<pcl::PointCloud<pcl::PointXYZ>>& bin, bool bDisplayCloud = false, bool bDisplayBin = false) {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);

    pcl::PointCloud<pcl::PointXYZ>::Ptr skeletonPointsPtr = skeletonPoints.makeShared();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> clrSkeletonPoints(skeletonPointsPtr, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(skeletonPointsPtr, clrSkeletonPoints, "skeletonpoint");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "skeletonpoint");

    if (bDisplayCloud == true) {
        if (bDisplayBin == false) {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> clrCloud(cloud, 128, 64, 0);
            viewer->addPointCloud<pcl::PointXYZ>(cloud, clrCloud, "branch");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "branch");
        }
        else {
            for (int i = 0; i < bin.size(); i++) {
                std::ostringstream ostr;
                ostr << "bin" << i;
                std::string id = ostr.str();
                pcl::PointCloud<pcl::PointXYZ>::Ptr a_bin(new pcl::PointCloud<pcl::PointXYZ>);
                a_bin = bin[i].makeShared();
                pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> random_color(a_bin);
                viewer->addPointCloud<pcl::PointXYZ>(a_bin, random_color, id);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, id);
            }
        }
    }

    for (int s = 0; s < skeletonEdges.size(); s++) {
        pcl::PointXYZ ptStart = skeletonPoints[std::get<0>(skeletonEdges[s])];
        pcl::PointXYZ ptEnd = skeletonPoints[std::get<1>(skeletonEdges[s])];
        std::ostringstream ostr;
        ostr << "line" << s;
        std::string id = ostr.str();
        viewer->addLine(ptStart, ptEnd, 0, 0, 0, id);
    }

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPosition(pos_x, pos_y, pos_z, view_x, view_y, view_z, up_x, up_y, up_z);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));

        SaveCurrentCameraParameters(viewer);

        std::ostringstream ostr;
        ostr << "Pos(" << pos_x << "," << pos_y << "," << pos_z << ") View(" << view_x << "," << view_y << "," << view_z << ") Up(" << up_x << "," << up_y << "," << up_z << ")";
        viewer->setWindowName(ostr.str());
    }
}


void MainTrunkExtraction(std::vector<int> edgesInCurrentOffshoot, pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges, std::vector<std::vector<int>>& edgesInOffshoots, std::vector<int>& offShootsLevel, int iBeginNode, int level) {

    std::vector<int> endEdges;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<0>(skeletonEdges[i]) == iBeginNode) {
            endEdges.push_back(i);
        }
    }
    switch (endEdges.size()) {
    case 0:
        if (edgesInCurrentOffshoot.size() != 0) { // In case of no skeletons
            edgesInOffshoots.push_back(edgesInCurrentOffshoot);
            offShootsLevel.push_back(level);
        }
        break;
    case 1:
        edgesInCurrentOffshoot.push_back(endEdges[0]);
        MainTrunkExtraction(edgesInCurrentOffshoot, skeletonPoints, skeletonEdges, edgesInOffshoots, offShootsLevel, std::get<1>(skeletonEdges[endEdges[0]]), level);
        break;
    default:
        if (edgesInCurrentOffshoot.size() != 0) { // In case of bifurcation at root
            edgesInOffshoots.push_back(edgesInCurrentOffshoot);
            offShootsLevel.push_back(level);
        }

        int iBeginEdge = -1;
        for (int i = 0; i < skeletonEdges.size(); i++) {
            if (std::get<1>(skeletonEdges[i]) == iBeginNode) {
                iBeginEdge = i;
                break;
            }
        }
        std::vector<float> angles(endEdges.size());
        if (iBeginEdge == -1) {
            // If separating at the root, use the angle between z
            Eigen::Vector3f vIn = { 0,0,1 };
            for (int s = 0; s < endEdges.size(); s++) {
                pcl::PointXYZ ptOutEdgeBegin = skeletonPoints[std::get<0>(skeletonEdges[endEdges[s]])];
                pcl::PointXYZ ptOutEdgeEnd = skeletonPoints[std::get<1>(skeletonEdges[endEdges[s]])];
                Eigen::Vector3f vOut = {
                    ptOutEdgeEnd.x - ptOutEdgeBegin.x,
                    ptOutEdgeEnd.y - ptOutEdgeBegin.y,
                    ptOutEdgeEnd.z - ptOutEdgeBegin.z
                };
                float angle = acos(vIn.dot(vOut) / vIn.norm() / vOut.norm());
                angles[s] = angle;
            }
        }
        else {
            pcl::PointXYZ ptInEdgeBegin = skeletonPoints[std::get<0>(skeletonEdges[iBeginEdge])];
            pcl::PointXYZ ptInEdgeEnd = skeletonPoints[std::get<1>(skeletonEdges[iBeginEdge])];
            Eigen::Vector3f vIn = {
                    ptInEdgeEnd.x - ptInEdgeBegin.x,
                    ptInEdgeEnd.y - ptInEdgeBegin.y,
                    ptInEdgeEnd.z - ptInEdgeBegin.z
            };
            /* The main sub-offshoot is regarded as an son offshoot with the minimum angle among angles between the direction of the father offshoot and son offshoots at the same layer */
            for (int s = 0; s < endEdges.size(); s++) {
                pcl::PointXYZ ptOutEdgeBegin = skeletonPoints[std::get<0>(skeletonEdges[endEdges[s]])];
                pcl::PointXYZ ptOutEdgeEnd = skeletonPoints[std::get<1>(skeletonEdges[endEdges[s]])];
                Eigen::Vector3f vOut = {
                    ptOutEdgeEnd.x - ptOutEdgeBegin.x,
                    ptOutEdgeEnd.y - ptOutEdgeBegin.y,
                    ptOutEdgeEnd.z - ptOutEdgeBegin.z
                };
                float angle = acos(vIn.dot(vOut) / vIn.norm() / vOut.norm());
                angles[s] = angle;
            }
        }
        
        // Combine fields and sort
        std::vector<std::tuple<int, float>> edgesWithAngles;
        for (int i = 0; i < endEdges.size(); i++) {
            edgesWithAngles.push_back(std::make_tuple(endEdges[i], angles[i]));
        }

        std::sort(edgesWithAngles.begin(), edgesWithAngles.end(), [](auto& e1, auto& e2) {
            return std::get<1>(e1) < std::get<1>(e2);
            });
        for (int i = 0; i < endEdges.size(); i++) {
            endEdges[i] = std::get<0>(edgesWithAngles[i]);
        }
        std::vector<int> newOffshoot0;
        newOffshoot0.push_back(endEdges[0]);
        MainTrunkExtraction(newOffshoot0, skeletonPoints, skeletonEdges, edgesInOffshoots, offShootsLevel, std::get<1>(skeletonEdges[endEdges[0]]), level);
        for (int s = 1; s < endEdges.size(); s++) {
            std::vector<int> newOffshooti;
            newOffshooti.push_back(endEdges[s]);
            MainTrunkExtraction(newOffshooti, skeletonPoints, skeletonEdges, edgesInOffshoots, offShootsLevel, std::get<1>(skeletonEdges[endEdges[s]]), level + 1);
        }
        break;
    }
}

//void MainTrunkExtraction( std::vector<int> edgesInCurrentOffshoot, pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges, std::vector<std::vector<int>>& edgesInOffshoots, std::vector<int>& offShootsLevel, int iBeginEdge, int level) {
//
//    std::vector<int> endEdges;
//    for (int s = 0; s < skeletonEdges.size(); s++) {
//        if (std::get<1>(skeletonEdges[iBeginEdge]) == std::get<0>(skeletonEdges[s])) {
//            endEdges.push_back(s);
//        }
//    }
//    edgesInCurrentOffshoot.push_back(iBeginEdge);
//    switch (endEdges.size()) {
//    case 0: {
//        edgesInOffshoots.push_back(edgesInCurrentOffshoot);
//        offShootsLevel.push_back(level);
//        break;
//    }
//    case 1: {
//        MainTrunkExtraction( edgesInCurrentOffshoot, skeletonPoints, skeletonEdges, edgesInOffshoots, offShootsLevel, endEdges[0], level);
//        break;
//    }
//    default: {
//        std::vector<float> angles(endEdges.size());
//        pcl::PointXYZ ptInEdgeBegin = skeletonPoints[std::get<0>(skeletonEdges[iBeginEdge])];
//        pcl::PointXYZ ptInEdgeEnd = skeletonPoints[std::get<1>(skeletonEdges[iBeginEdge])];
//        Eigen::Vector3f vIn = {
//                ptInEdgeEnd.x - ptInEdgeBegin.x,
//                ptInEdgeEnd.y - ptInEdgeBegin.y,
//                ptInEdgeEnd.z - ptInEdgeBegin.z
//        };
//        /* The main sub-offshoot is regarded as an son offshoot with the minimum angle among angles between the direction of the father offshoot and son offshoots at the same layer */
//        for (int s = 0; s < endEdges.size(); s++) {
//            pcl::PointXYZ ptOutEdgeBegin = skeletonPoints[std::get<0>(skeletonEdges[endEdges[s]])];
//            pcl::PointXYZ ptOutEdgeEnd = skeletonPoints[std::get<1>(skeletonEdges[endEdges[s]])];
//            Eigen::Vector3f vOut = {
//                ptOutEdgeEnd.x - ptOutEdgeBegin.x,
//                ptOutEdgeEnd.y - ptOutEdgeBegin.y,
//                ptOutEdgeEnd.z - ptOutEdgeBegin.z
//            };
//            float angle = acos(vIn.dot(vOut) / vIn.norm() / vOut.norm());
//            angles[s] = angle;
//        }
//        edgesInOffshoots.push_back(edgesInCurrentOffshoot);
//        offShootsLevel.push_back(level);
//        // Combine fields and sort
//        std::vector<std::tuple<int, float>> edgesWithAngles;
//        for (int i = 0; i < endEdges.size(); i++) {
//            edgesWithAngles.push_back(std::make_tuple(endEdges[i], angles[i]));
//        }
//
//        std::sort(edgesWithAngles.begin(), edgesWithAngles.end(), [](auto& e1, auto& e2) {
//            return std::get<1>(e1) < std::get<1>(e2);
//            });
//        for (int i = 0; i < endEdges.size(); i++) {
//            endEdges[i] = std::get<0>(edgesWithAngles[i]);
//        }
//
//        MainTrunkExtraction(std::vector<int>(), skeletonPoints, skeletonEdges, edgesInOffshoots, offShootsLevel, endEdges[0], level);
//        for (int s = 1; s < endEdges.size(); s++) {
//            MainTrunkExtraction(std::vector<int>(), skeletonPoints, skeletonEdges, edgesInOffshoots, offShootsLevel, endEdges[s], level + 1);
//        }
//        break;
//    }
//    }
//}

void ViewOffshoots(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges, std::vector<pcl::PointCloud<pcl::PointXYZ>>& bin, std::vector<std::vector<int>>& edgesInOffshoots, std::vector<int>& offShootsLevel, int iDifference = 2, bool bDisplaySkeleton = false) {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);
    std::vector<bool> nodeAppended(skeletonPoints.size(), false);
    std::vector<bool> cloudAppened(cloud->size(), false);

    switch (iDifference) {
    case 0: { // Distinguish different offshoots according to levels 
        int iMaxLevel = *std::max_element(offShootsLevel.begin(), offShootsLevel.end());

        for (int i = 0; i <= iMaxLevel; i++) {
            std::vector<int> sameLevelOffshoots;
            pcl::PointCloud<pcl::PointXYZ> levelOffshoot_cloud;
            for (int j = 0; j < offShootsLevel.size(); j++) {
                if (offShootsLevel[j] == i) {
                    sameLevelOffshoots.push_back(j);
                }
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr level_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (int j = 0; j < sameLevelOffshoots.size(); j++) {
                for (int k = 0; k < edgesInOffshoots[sameLevelOffshoots[j]].size(); k++) {
                    int v1 = std::get<0>(skeletonEdges[edgesInOffshoots[sameLevelOffshoots[j]][k]]);
                    int v2 = std::get<1>(skeletonEdges[edgesInOffshoots[sameLevelOffshoots[j]][k]]);
                    if (nodeAppended[v1] == false) {
                        *level_cloud += bin[v1];
                        nodeAppended[v1] = true;
                    }
                    if (nodeAppended[v2] == false) {
                        *level_cloud += bin[v2];
                        nodeAppended[v2] = true;
                    }
                }
            }
            std::ostringstream ostr;
            ostr << "level" << i;
            std::string id = ostr.str();

            pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> random_color(level_cloud);
            viewer->addPointCloud<pcl::PointXYZ>(level_cloud, random_color, id);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, id);
        }
        break;
    }
    case 1: {// Distinguish different offshoots according to connectivity 

        std::vector<bool> treetop(edgesInOffshoots.size(), true);
        for (int i = 0; i < edgesInOffshoots.size(); i++) {
            int end_node = std::get<1>(skeletonEdges[edgesInOffshoots[i].back()]);

            for (int j = 0; j < edgesInOffshoots.size(); j++) {
                int begin_node = std::get<0>(skeletonEdges[edgesInOffshoots[j].front()]);
                if (end_node == begin_node) {
                    treetop[i] = false;
                    break;
                }
            }
        }

        int iMaxLevel = *std::max_element(offShootsLevel.begin(), offShootsLevel.end());
        int cnt = 0;
        for (int i = 0; i <= iMaxLevel; i++) {
            std::vector<int> level_treetops;
            for (int j = 0; j < offShootsLevel.size(); j++) {
                if (offShootsLevel[j] == i && treetop[j] == true) {
                    level_treetops.push_back(j);
                }
            }

            for (int j = 0; j < level_treetops.size(); j++) {
                std::vector<int> link;
                link.assign(edgesInOffshoots[level_treetops[j]].begin(), edgesInOffshoots[level_treetops[j]].end());
                int begin_node = std::get<0>(skeletonEdges[edgesInOffshoots[level_treetops[j]].front()]);

                for (int k = 0; k < edgesInOffshoots.size(); ) {
                    int end_node = std::get<1>(skeletonEdges[edgesInOffshoots[k].back()]);
                    if (end_node == begin_node && offShootsLevel[k] == i) {
                        link.insert(link.begin(), edgesInOffshoots[k].begin(), edgesInOffshoots[k].end());
                        begin_node = std::get<0>(skeletonEdges[edgesInOffshoots[k].front()]);
                        k = 0;
                    }
                    else {
                        k++;
                    }
                }
                pcl::PointCloud<pcl::PointXYZ>::Ptr link_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                for (int k = 0; k < link.size(); k++) {
                    int v1 = std::get<0>(skeletonEdges[link[k]]);
                    int v2 = std::get<1>(skeletonEdges[link[k]]);
                    if (nodeAppended[v1] == false) {
                        *link_cloud += bin[v1];
                        nodeAppended[v1] = true;
                    }
                    if (nodeAppended[v2] == false) {
                        *link_cloud += bin[v2];
                        nodeAppended[v2] = true;
                    }
                }
                std::ostringstream ostr;
                ostr << "link" << cnt;
                std::string id = ostr.str();

                pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> random_color(link_cloud);
                viewer->addPointCloud<pcl::PointXYZ>(link_cloud, random_color, id);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, id);
                cnt++;
            }
        }

        break;
    }
    default: {// Distinguish different offshoots according to segments.
        for (int s = 0; s < edgesInOffshoots.size(); s++) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr offshoot_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (int t = 0; t < edgesInOffshoots[s].size(); t++) {
                int v1 = std::get<0>(skeletonEdges[edgesInOffshoots[s][t]]);
                int v2 = std::get<1>(skeletonEdges[edgesInOffshoots[s][t]]);
                if (nodeAppended[v1] == false) {
                    *offshoot_cloud += bin[v1];
                    nodeAppended[v1] = true;
                }
                if (nodeAppended[v2] == false) {
                    *offshoot_cloud += bin[v2];
                    nodeAppended[v2] = true;
                }
            }
            std::ostringstream ostr;
            ostr << "offshoot" << s;
            std::string id = ostr.str();
            
            pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> random_color(offshoot_cloud);
            viewer->addPointCloud<pcl::PointXYZ>(offshoot_cloud, random_color, id);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, id);
        }
        break;
    }
    }

    if (bDisplaySkeleton == true) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr skeletonPointsPtr = skeletonPoints.makeShared();
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> clrSkeletonPoints(skeletonPointsPtr, 255, 0, 0);
        viewer->addPointCloud<pcl::PointXYZ>(skeletonPointsPtr, clrSkeletonPoints, "SkeletonPoint");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "SkeletonPoint");
        for (int s = 0; s < skeletonEdges.size(); s++) {
            pcl::PointXYZ ptStart = skeletonPoints[std::get<0>(skeletonEdges[s])];
            pcl::PointXYZ ptEnd = skeletonPoints[std::get<1>(skeletonEdges[s])];
            std::ostringstream ostr;
            ostr << "line" << s;
            std::string id = ostr.str();
            viewer->addLine(ptStart, ptEnd, 0, 0, 0, id);
        }
    }

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPosition(pos_x, pos_y, pos_z, view_x, view_y, view_z, up_x, up_y, up_z);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));

        SaveCurrentCameraParameters(viewer);

        std::ostringstream ostr;
        ostr << "Pos(" << pos_x << "," << pos_y << "," << pos_z << ") View(" << view_x << "," << view_y << "," << view_z << ") Up(" << up_x << "," << up_y << "," << up_z << ")";
        viewer->setWindowName(ostr.str());
    }
}

void RemoveSmallOffshoots(int iMinPts, float fMinLength, int iMinCnt, pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges, std::vector<pcl::PointCloud<pcl::PointXYZ>>& bin, std::vector<std::vector<int>>& edgesInOffshoots, std::vector<int>& offshootsLevel) {

    std::vector<bool> treetop(edgesInOffshoots.size(), true);
    for (int i = 0; i < edgesInOffshoots.size(); i++) {
        int end_node = std::get<1>(skeletonEdges[edgesInOffshoots[i].back()]);
        for (int j = 0; j < edgesInOffshoots.size(); j++) {
            int begin_node = std::get<0>(skeletonEdges[edgesInOffshoots[j].front()]);
            if (end_node == begin_node) {
                treetop[i] = false;
                break;
            }
        }
    }

    std::vector<float> offshootsLength(edgesInOffshoots.size());
    for (int i = 0; i < edgesInOffshoots.size(); i++) {
        float fLength = 0.0;
        for (int j = 0; j < edgesInOffshoots[i].size(); j++) {
            int idxEdge = edgesInOffshoots[i][j];
            fLength += GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[idxEdge])], skeletonPoints[std::get<1>(skeletonEdges[idxEdge])]);
        }
        offshootsLength[i] = fLength;
    }
    //Criterion of removal: only remove treetops.(Only save the end node of a directed edge.
    // Record all nodes

    std::set<int> need_delete_nodes_set;
    for (int i = 0; i < edgesInOffshoots.size(); i++) {
        if (treetop[i] == false) {
            continue;
        }
        if (offshootsLength[i] < fMinLength) {
            for (int j = 0; j < edgesInOffshoots[i].size(); j++) {
                int v2 = std::get<1>(skeletonEdges[edgesInOffshoots[i][j]]);
                need_delete_nodes_set.insert(v2);
            }
        }
        if (edgesInOffshoots[i].size() < iMinCnt) {
            for (int j = 0; j < edgesInOffshoots[i].size(); j++) {
                int v2 = std::get<1>(skeletonEdges[edgesInOffshoots[i][j]]);
                need_delete_nodes_set.insert(v2);
            }
        }
        int offshoot_treetop_node = std::get<1>(skeletonEdges[edgesInOffshoots[i].back()]);
        if (bin[offshoot_treetop_node].size() < iMinPts) {
            need_delete_nodes_set.insert(offshoot_treetop_node);
        }
    }
    std::vector<int> need_delete_nodes;
    need_delete_nodes.assign(need_delete_nodes_set.begin(), need_delete_nodes_set.end());
    std::sort(need_delete_nodes.begin(), need_delete_nodes.end());
    while (!need_delete_nodes.empty()) {
        int node = need_delete_nodes[0];
        need_delete_nodes.erase(need_delete_nodes.begin());

        skeletonPoints.erase(skeletonPoints.begin() + node);
        bin.erase(bin.begin() + node);

        for (int i = 0; i < skeletonEdges.size(); i++) {
            int v1 = std::get<0>(skeletonEdges[i]);
            int v2 = std::get<1>(skeletonEdges[i]);
            if (v1 == node || v2 == node) {
                skeletonEdges.erase(skeletonEdges.begin() + i);
                i--;
                continue;
            }
            if (v1 > node) {
                v1--;
            }
            if (v2 > node) {
                v2--;
            }
            skeletonEdges[i] = std::make_tuple(v1, v2);
        }

        for (int i = 0; i < need_delete_nodes.size(); i++) {
            if (need_delete_nodes[i] > node) {
                need_delete_nodes[i]--;
            }
        }
    }
    //Auto reconstruction
    edgesInOffshoots.clear();
    offshootsLevel.clear();
    MainTrunkExtraction(std::vector<int>(), skeletonPoints, skeletonEdges, edgesInOffshoots, offshootsLevel);
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


pcl::PointXYZ getFootPt(pcl::PointXYZ L1, pcl::PointXYZ L2, pcl::PointXYZ P) {
    pcl::PointXYZ N;
    float x0 = P.x;
    float y0 = P.y;
    float z0 = P.z;
    float x1 = L1.x;
    float y1 = L1.y;
    float z1 = L1.z;
    float x2 = L2.x;
    float y2 = L2.y;
    float z2 = L2.z;

    double k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) / (pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2));

    N = { (float)(k * (x2 - x1) + x1), (float)(k * (y2 - y1) + y1),(float)(k * (z2 - z1) + z1) };
    return N;
}
void PCAComputeCloudMainDirection(pcl::PointCloud<pcl::PointXYZ> cloud, Eigen::Vector3f& main_direction) {
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(cloud, pcaCentroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(cloud, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();

    int max_eigen_value_col = 0;
    float max_eigen_value = eigenValuesPCA(0);
    if (max_eigen_value < eigenValuesPCA(1)) {
        max_eigen_value = eigenValuesPCA(1);
        max_eigen_value_col = 1;
    }
    if (max_eigen_value < eigenValuesPCA(2)) {
        max_eigen_value = eigenValuesPCA(2);
        max_eigen_value_col = 2;
    }
    main_direction = eigenVectorsPCA.col(max_eigen_value_col);
}

void OptimizeBifurcations(int iBeginNodeNo, float fStepLen, float fAffectRange, int iBinMinPts, float fMinCorrelation, pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges, std::vector<pcl::PointCloud<pcl::PointXYZ>>& bin) {
    std::vector<int> sonNodes;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<0>(skeletonEdges[i]) == iBeginNodeNo) {
            int iSonNode = std::get<1>(skeletonEdges[i]);
            if (bin[iSonNode].size() < iBinMinPts) {// If a bin is too small, the result could be overfitting
                    continue;
            }
            sonNodes.push_back(iSonNode);
        }
    }
    if (sonNodes.size() == 1) {// On an offshoot
        OptimizeBifurcations(sonNodes[0], fStepLen, fAffectRange, iBinMinPts, fMinCorrelation, skeletonPoints, skeletonEdges, bin);
        return;
    }
    else if (sonNodes.size() == 0) {// On a treetop
        return;
    }
    //On a bifurcation
    std::vector<pcl::PointXYZ> exploreSeq;// Separate an edge and store these nodes
    std::vector<int> correspondingStartSeq; // Store the start / end point of the corresponding edge of each nodes
    std::vector<int> correspondingEndSeq;
    std::vector<bool> vecFlagExistNode; // Flag whether a node is an existed node in the skeleton or not
    float fExploredDist = 0.0;

    int iBeginExploreFatherNo = iBeginNodeNo;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<1>(skeletonEdges[i]) != iBeginExploreFatherNo)
            continue;

        pcl::PointXYZ ptBegin = skeletonPoints[std::get<0>(skeletonEdges[i])];
        pcl::PointXYZ ptEnd = skeletonPoints[std::get<1>(skeletonEdges[i])];

        float fSkeletonLineLength = GetDistanceBetween2pts(ptBegin, ptEnd);
        int iPartition = std::max(int(fSkeletonLineLength / fStepLen), 1);// Divide the skeleton line 
        for (int j = 1; j <= iPartition; j++) {
            fExploredDist += (float)1 / iPartition * fSkeletonLineLength;
            if (fExploredDist > fAffectRange)break;
            pcl::PointXYZ ptExplore = {
                (float)(ptEnd.x + (float)j / iPartition * (ptBegin.x - ptEnd.x)),
                (float)(ptEnd.y + (float)j / iPartition * (ptBegin.y - ptEnd.y)),
                (float)(ptEnd.z + (float)j / iPartition * (ptBegin.z - ptEnd.z))
            };
            correspondingStartSeq.push_back(std::get<0>(skeletonEdges[i]));
            correspondingEndSeq.push_back(std::get<1>(skeletonEdges[i]));

            exploreSeq.push_back(ptExplore);
            if (j == iPartition) {
                vecFlagExistNode.push_back(true);
            }
            else {
                vecFlagExistNode.push_back(false);
            }
        }
        if (fExploredDist > fAffectRange)break;
        iBeginExploreFatherNo = std::get<0>(skeletonEdges[i]);
        i = 0;
    }
    exploreSeq.insert(exploreSeq.begin(), skeletonPoints[iBeginNodeNo]);
    correspondingStartSeq.insert(correspondingStartSeq.begin(), iBeginNodeNo);
    correspondingEndSeq.insert(correspondingEndSeq.begin(), -1);
    vecFlagExistNode.insert(vecFlagExistNode.begin(), true);
    
    std::vector<int> changedSonNodes; // Store son nodes that need changed
    std::vector<int> changedSonOptFatherSeqNode; //Store the optimized father node of each son nodes in the explore sequence
    
    for (auto sonNode : sonNodes) {
        pcl::PointCloud<pcl::PointXYZ> pcDataset;
        pcDataset += bin[sonNode];
        pcDataset += bin[iBeginNodeNo];
        float fMaxCosineCorrelation = 0;
        int iOptimizedCenterSeqNo = -1;

        int iPreviousExistedFatherNode = -1; // Flag the previous explored existed father node to determine whether to add point cloud into dataset or not


        for (int i = 1; i < exploreSeq.size(); i++) {
            if (iPreviousExistedFatherNode != correspondingStartSeq[i - 1]) {
                pcDataset += bin[correspondingStartSeq[i - 1]];  // When the program explores in a new edges, the dataset should be expanded.
                iPreviousExistedFatherNode = correspondingStartSeq[i - 1];

                RemoveClosePoint(pcDataset, 0.000001);
            }


            pcl::PointXYZ ptCurrentSon = skeletonPoints[sonNode];
            pcl::PointXYZ ptCurrentCenter = exploreSeq[i - 1];
            pcl::PointXYZ ptCurrentFather = exploreSeq[i];

            Eigen::Vector3f vecCenterToSon = {
                ptCurrentSon.x - ptCurrentCenter.x,
                ptCurrentSon.y - ptCurrentCenter.y,
                ptCurrentSon.z - ptCurrentCenter.z
            };
            Eigen::Vector3f vecCenterToFather = {
                ptCurrentFather.x - ptCurrentCenter.x,
                ptCurrentFather.y - ptCurrentCenter.y,
                ptCurrentFather.z - ptCurrentCenter.z
            };

            //Use a plane of the angle bisector to divide the point set
            Eigen::Vector3f vecBisector = vecCenterToSon / vecCenterToSon.norm() + vecCenterToFather / vecCenterToFather.norm();
            vecBisector.normalize();

            pcl::PointXYZ ptPointOnBisector = {
                ptCurrentCenter.x + vecBisector.x(),
                ptCurrentCenter.y + vecBisector.y(),
                ptCurrentCenter.z + vecBisector.z()
            };

            pcl::PointXYZ ptFoot = getFootPt(ptCurrentCenter, ptPointOnBisector, ptCurrentSon);
            Eigen::Vector3f vecCuttingPlaneNormal = {
                ptCurrentSon.x - ptFoot.x,
                ptCurrentSon.y - ptFoot.y,
                ptCurrentSon.z - ptFoot.z,
            };

            double A = vecCuttingPlaneNormal.x();
            double B = vecCuttingPlaneNormal.y();
            double C = vecCuttingPlaneNormal.z();
            double D = -(A * ptCurrentCenter.x + B * ptCurrentCenter.y + C * ptCurrentCenter.z);

            pcl::PointCloud<pcl::PointXYZ> pcToFather;
            pcl::PointCloud<pcl::PointXYZ> pcToSon;
            for (auto ptInDataSet : pcDataset) {
                //Don't use the point when it is equal to, which means that the two act at the same time
                if (A * ptInDataSet.x + B * ptInDataSet.y + C * ptInDataSet.z + D < 0) {
                    pcToFather.push_back(ptInDataSet);
                    continue;
                }
                if (A * ptInDataSet.x + B * ptInDataSet.y + C * ptInDataSet.z + D > 0) {
                    pcToSon.push_back(ptInDataSet);
                }
            }
            if (pcToFather.size() < iBinMinPts) {
                //When a certain kind of oblique cutting, the number of points may be too small. At this time, the cosine value must be low or even calculated incorrectly
                continue;
            }

            //To prevent the main direction from being too offset, two point sets add the central nodes -- Wrong, will disturb computation
            //pcToFather.push_back(ptCurrentCenter);
            //pcToSon.push_back(ptCurrentCenter);

            //Compute the main direction of point cloud

            if (pcToSon.size() <= iBinMinPts) {
                continue;
            }

            Eigen::Vector3f vecToFatherMain;
            PCAComputeCloudMainDirection(pcToFather, vecToFatherMain);
            Eigen::Vector3f vecToSonMain;
            PCAComputeCloudMainDirection(pcToSon, vecToSonMain);

            float fCosineOfSon = abs(vecToSonMain.dot(vecCenterToSon) / vecToSonMain.norm() / vecCenterToSon.norm());
            float fCosineOfFather = abs(vecToFatherMain.dot(vecCenterToFather) / vecToFatherMain.norm() / vecCenterToFather.norm());

            float fCurrentCosineCorrelation = fCosineOfFather * fCosineOfSon;
            if (fCurrentCosineCorrelation > fMaxCosineCorrelation) {
                fMaxCosineCorrelation = fCurrentCosineCorrelation;
                iOptimizedCenterSeqNo = i - 1;
            }
        }
        if (fMaxCosineCorrelation == 0 || iOptimizedCenterSeqNo == -1 || fMaxCosineCorrelation < fMinCorrelation)
            continue;
        changedSonOptFatherSeqNode.push_back(iOptimizedCenterSeqNo);
        changedSonNodes.push_back(sonNode);
    }


    if (changedSonNodes.size() == 0) {
        for (auto sonNode : sonNodes) {
            OptimizeBifurcations(sonNode, fStepLen, fAffectRange, iBinMinPts, fMinCorrelation, skeletonPoints, skeletonEdges, bin);
        }
        return;
    }
    //Sort by connecting nodes
    std::vector<std::tuple<int, int>> combinedRect;
    for (int i = 0; i < changedSonNodes.size(); i++) {
        combinedRect.push_back(std::make_tuple(changedSonNodes[i], changedSonOptFatherSeqNode[i]));
    }
    std::sort(combinedRect.begin(), combinedRect.end(), [](auto& e1, auto& e2) {
        return std::get<1>(e1) < std::get<1>(e2);
        });
    for (int i = 0; i < combinedRect.size(); i++) {
        changedSonNodes[i] = std::get<0>(combinedRect[i]);
        changedSonOptFatherSeqNode[i] = std::get<1>(combinedRect[i]);
    }
    
    std::vector<int> realSonNodes;// Before that, always use filtered son nodes
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<0>(skeletonEdges[i]) == iBeginNodeNo) {
            realSonNodes.push_back(std::get<1>(skeletonEdges[i]));
        }
    }
    

    //Break up all concerning lines and connect
    std::vector<int> vecNewNodeSeq(exploreSeq.size());
    int iPrevLineNodeNo = correspondingStartSeq[exploreSeq.size() - 1];
    int iEndExploreSeqNo = 0;
    if (realSonNodes.size() == changedSonNodes.size()) {
        changedSonOptFatherSeqNode[0] = changedSonOptFatherSeqNode[1]; //This can avoid the very small fracture
        iEndExploreSeqNo = changedSonOptFatherSeqNode[0];//Remove the rest part of the terminal father edges
    }
    for (int i = exploreSeq.size() - 1; i >= iEndExploreSeqNo; i--) {

        if (vecFlagExistNode[i] == true) {
            iPrevLineNodeNo = correspondingStartSeq[i];
            vecNewNodeSeq[i] = correspondingStartSeq[i];
            continue;
        }
        for (int h = 0; h < skeletonEdges.size(); h++) {
            if (std::get<0>(skeletonEdges[h]) == iPrevLineNodeNo &&
                std::get<1>(skeletonEdges[h]) == correspondingEndSeq[i]) {
                skeletonEdges.erase(skeletonEdges.begin() + h);
                break;
            }
        }
        skeletonPoints.push_back(exploreSeq[i]);
        pcl::PointCloud<pcl::PointXYZ> pcNewCenterBin;
        pcNewCenterBin += bin[correspondingStartSeq[i]];
        pcNewCenterBin += bin[correspondingEndSeq[i]];

        //Remove points not in a suitable step
        pcl::PointXYZ dir_begin = skeletonPoints[correspondingStartSeq[i]];
        pcl::PointXYZ dir_end = skeletonPoints[correspondingEndSeq[i]];
        Eigen::Vector3f dir = {
            dir_end.x - dir_begin.x,
            dir_end.y - dir_begin.y,
            dir_end.z - dir_begin.z
        };
        for (int s = 0; s < pcNewCenterBin.size(); s++) {
            float A = dir.x();
            float B = dir.y();
            float C = dir.z();
            float D = -A * exploreSeq[i].x - B * exploreSeq[i].y - C * exploreSeq[i].z;
            if (abs(A * pcNewCenterBin[s].x + B * pcNewCenterBin[s].y + C * pcNewCenterBin[s].z + D) / sqrt(A * A + B * B + C * C) > 5 * fStepLen) {
                pcNewCenterBin.erase(pcNewCenterBin.begin() + s);
                s--;
            }
        }
        if (pcNewCenterBin.size() <= 1) {
            pcNewCenterBin.clear();
            pcNewCenterBin += bin[correspondingStartSeq[i]];
            pcNewCenterBin += bin[correspondingEndSeq[i]];
        }
        
        bin.push_back(pcNewCenterBin);
        int iNewNodeNo = skeletonPoints.size() - 1;
        skeletonEdges.push_back(std::make_tuple(iPrevLineNodeNo, iNewNodeNo));
        skeletonEdges.push_back(std::make_tuple(iNewNodeNo, correspondingEndSeq[i]));

        vecNewNodeSeq[i] = iNewNodeNo;
        iPrevLineNodeNo = iNewNodeNo;
    }

    //Replace lines to son node
    for (int i = 0; i < changedSonNodes.size(); i++) {
        for (int h = 0; h < skeletonEdges.size(); h++) {
            if (std::get<1>(skeletonEdges[h]) == changedSonNodes[i]) {
                skeletonEdges.erase(skeletonEdges.begin() + h);
                break;
            }
        }
        skeletonEdges.push_back(std::make_tuple(vecNewNodeSeq[changedSonOptFatherSeqNode[i]], changedSonNodes[i]));
    }

    //Combine broken lines
    iPrevLineNodeNo = correspondingStartSeq[exploreSeq.size() - 1];
    for (int i = exploreSeq.size() - 1; i >= iEndExploreSeqNo; i--) {
        if (i == exploreSeq.size() - 1 && vecFlagExistNode[i] == true) {
            continue;
        }
        int node_outdegree = 0;
        int in_line_no;
        for (int j = 0; j < skeletonEdges.size(); j++) {
            if (std::get<0>(skeletonEdges[j]) == vecNewNodeSeq[i]) {
                node_outdegree++;
            }
            if (std::get<1>(skeletonEdges[j]) == vecNewNodeSeq[i]) {
                in_line_no = j;
            }
        }
        if (node_outdegree > 1 || vecFlagExistNode[i] == true || i == iEndExploreSeqNo) {
            skeletonEdges.erase(skeletonEdges.begin() + in_line_no);
            skeletonEdges.push_back(std::make_tuple(iPrevLineNodeNo, vecNewNodeSeq[i]));
            iPrevLineNodeNo = vecNewNodeSeq[i];
        }
        else {
            skeletonEdges.erase(skeletonEdges.begin() + in_line_no);
        }
    }
    for (auto sonNode : sonNodes) {
        OptimizeBifurcations(sonNode, fStepLen, fAffectRange, iBinMinPts, fMinCorrelation, skeletonPoints, skeletonEdges, bin);
    }
}

float getPrecision(float x) {
    float e = 0.000001;
    if (x >= 1) {
        int digit = 0;
        while (x >= 1) {
            x /= 10;
            digit++;
        }
        e *= pow(10, digit);
    }
    else {
        //In case of number like 0.0000..., small number are all considered as precision = 0.000001
        /*
        int digit = 0;
        while (x < 1) {
            x *= 10;
            digit++;
        }
        e /= pow(10, digit - 1);*/
    }
    return e;
}

void CombineEdgesOfSameDirections(pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges, std::vector<pcl::PointCloud<pcl::PointXYZ>>& bin) {
    for (int i = 0; i < skeletonEdges.size(); i++) {
        int line_prev = i;
        std::vector<int> lines_next;
        for (int j = 0; j < skeletonEdges.size(); j++) {
            if (std::get<1>(skeletonEdges[i]) == std::get<0>(skeletonEdges[j])) {
                lines_next.push_back(j);
            }
        }
        if (lines_next.size() != 1)
            continue;
        float e1x = getPrecision(skeletonPoints[std::get<0>(skeletonEdges[line_prev])].x);
        float e1y = getPrecision(skeletonPoints[std::get<0>(skeletonEdges[line_prev])].y);
        float e1z = getPrecision(skeletonPoints[std::get<0>(skeletonEdges[line_prev])].z);

        float e2x = getPrecision(skeletonPoints[std::get<1>(skeletonEdges[line_prev])].x);
        float e2y = getPrecision(skeletonPoints[std::get<1>(skeletonEdges[line_prev])].y);
        float e2z = getPrecision(skeletonPoints[std::get<1>(skeletonEdges[line_prev])].z);

        float e3x = getPrecision(skeletonPoints[std::get<1>(skeletonEdges[lines_next[0]])].x);
        float e3y = getPrecision(skeletonPoints[std::get<1>(skeletonEdges[lines_next[0]])].y);
        float e3z = getPrecision(skeletonPoints[std::get<1>(skeletonEdges[lines_next[0]])].z);

        pcl::PointXYZ A1 = {
            (float)((float)(int(skeletonPoints[std::get<0>(skeletonEdges[line_prev])].x / e1x)) * e1x),
            (float)((float)(int(skeletonPoints[std::get<0>(skeletonEdges[line_prev])].y / e1y)) * e1y),
            (float)((float)(int(skeletonPoints[std::get<0>(skeletonEdges[line_prev])].z / e1z)) * e1z)
        };
        pcl::PointXYZ A2 = {
            (float)((float)(int(skeletonPoints[std::get<1>(skeletonEdges[line_prev])].x / e2x)) * e2x),
            (float)((float)(int(skeletonPoints[std::get<1>(skeletonEdges[line_prev])].y / e2y)) * e2y),
            (float)((float)(int(skeletonPoints[std::get<1>(skeletonEdges[line_prev])].z / e2z)) * e2z)
        };
        pcl::PointXYZ A3 = {
            (float)((float)(int(skeletonPoints[std::get<1>(skeletonEdges[lines_next[0]])].x / e3x)) * e3x),
            (float)((float)(int(skeletonPoints[std::get<1>(skeletonEdges[lines_next[0]])].y / e3y)) * e3y),
            (float)((float)(int(skeletonPoints[std::get<1>(skeletonEdges[lines_next[0]])].z / e3z)) * e3z)
        };

        typedef boost::geometry::model::d2::point_xy<double> BoostPoint;
        typedef boost::geometry::model::polygon<BoostPoint> BoostPolygon;
        //typedef std::deque<BoostPolygon> BoostPolygonSet;

        //XOY Plane
        bool bXOY = false;
        Eigen::Vector2d B1D2_xy{
            A2.x - (A1.x + e1x),
            (A2.y + e2y) - A1.y
        };
        Eigen::Vector2d D1B2_xy{
            (A2.x + e2x) - A1.x,
            A2.y - (A1.y + e1y)
        };
        B1D2_xy.normalize();
        D1B2_xy.normalize();
        BoostPolygon OB2D2_xy;
        OB2D2_xy.outer().push_back(BoostPoint(0.0, 0.0));
        OB2D2_xy.outer().push_back(BoostPoint(B1D2_xy.x(), B1D2_xy.y()));
        OB2D2_xy.outer().push_back(BoostPoint(D1B2_xy.x(), D1B2_xy.y()));
        OB2D2_xy.outer().push_back(BoostPoint(0.0, 0.0));
        boost::geometry::correct(OB2D2_xy);

        Eigen::Vector2d B2D3_xy{
            A3.x - (A2.x + e2x),
            (A3.y + e3y) - A2.y
        };
        Eigen::Vector2d D2B3_xy{
            (A3.x + e3x) - A2.x,
            A3.y - (A2.y + e2y)
        };
        B2D3_xy.normalize();
        D2B3_xy.normalize();
        BoostPolygon OB3D3_xy;
        OB3D3_xy.outer().push_back(BoostPoint(0.0, 0.0));
        OB3D3_xy.outer().push_back(BoostPoint(B2D3_xy.x(), B2D3_xy.y()));
        OB3D3_xy.outer().push_back(BoostPoint(D2B3_xy.x(), D2B3_xy.y()));
        OB3D3_xy.outer().push_back(BoostPoint(0.0, 0.0));
        boost::geometry::correct(OB3D3_xy);
        bXOY = boost::geometry::intersects(OB2D2_xy, OB3D3_xy);

        //XOZ Plane
        bool bXOZ = false;
        Eigen::Vector2d B1D2_xz{
            A2.x - (A1.x + e1x),
            (A2.z + e2z) - A1.z
        };
        Eigen::Vector2d D1B2_xz{
            (A2.x + e2x) - A1.x,
            A2.z - (A1.z + e1z)
        };
        B1D2_xz.normalize();
        D1B2_xz.normalize();
        BoostPolygon OB2D2_xz;
        OB2D2_xz.outer().push_back(BoostPoint(0.0, 0.0));
        OB2D2_xz.outer().push_back(BoostPoint(B1D2_xz.x(), B1D2_xz.y()));
        OB2D2_xz.outer().push_back(BoostPoint(D1B2_xz.x(), D1B2_xz.y()));
        OB2D2_xz.outer().push_back(BoostPoint(0.0, 0.0));
        boost::geometry::correct(OB2D2_xz);
        Eigen::Vector2d B2D3_xz{
            A3.x - (A2.x + e2x),
            (A3.z + e3z) - A2.z
        };
        Eigen::Vector2d D2B3_xz{
            (A3.x + e3x) - A2.x,
            A3.z - (A2.z + e2z)
        };
        B2D3_xz.normalize();
        D2B3_xz.normalize();
        BoostPolygon OB3D3_xz;
        OB3D3_xz.outer().push_back(BoostPoint(0.0, 0.0));
        OB3D3_xz.outer().push_back(BoostPoint(B2D3_xz.x(), B2D3_xz.y()));
        OB3D3_xz.outer().push_back(BoostPoint(D2B3_xz.x(), D2B3_xz.y()));
        OB3D3_xz.outer().push_back(BoostPoint(0.0, 0.0));
        boost::geometry::correct(OB3D3_xz);
        bXOZ = boost::geometry::intersects(OB2D2_xz, OB3D3_xz);

        //YOZ Plane(In case of extreme situation)
        bool bYOZ = false;
        Eigen::Vector2d B1D2_yz{
            A2.y - (A1.y + e1y),
            (A2.z + e2z) - A1.z
        };
        Eigen::Vector2d D1B2_yz{
            (A2.y + e2y) - A1.y,
            A2.z - (A1.z + e1z)
        };
        B1D2_yz.normalize();
        D1B2_yz.normalize();
        BoostPolygon OB2D2_yz;
        OB2D2_yz.outer().push_back(BoostPoint(0.0, 0.0));
        OB2D2_yz.outer().push_back(BoostPoint(B1D2_yz.x(), B1D2_yz.y()));
        OB2D2_yz.outer().push_back(BoostPoint(D1B2_yz.x(), D1B2_yz.y()));
        OB2D2_yz.outer().push_back(BoostPoint(0.0, 0.0));
        boost::geometry::correct(OB2D2_yz);
        Eigen::Vector2d B2D3_yz{
            A3.y - (A2.y + e2y),
            (A3.z + e3z) - A2.z
        };
        Eigen::Vector2d D2B3_yz{
            (A3.y + e3y) - A2.y,
            A3.z - (A2.z + e2z)
        };
        B2D3_yz.normalize();
        D2B3_yz.normalize();
        BoostPolygon OB3D3_yz;
        OB3D3_yz.outer().push_back(BoostPoint(0.0, 0.0));
        OB3D3_yz.outer().push_back(BoostPoint(B2D3_yz.x(), B2D3_yz.y()));
        OB3D3_yz.outer().push_back(BoostPoint(D2B3_yz.x(), D2B3_yz.y()));
        OB3D3_yz.outer().push_back(BoostPoint(0.0, 0.0));
        boost::geometry::correct(OB3D3_yz);
        bYOZ = boost::geometry::intersects(OB2D2_yz, OB3D3_yz);

        // Judge whether need combination
        if (!(bXOY == true && bXOZ == true && bYOZ == true))
            continue;

        std::tuple<int, int> newEdge = std::make_tuple(std::get<0>(skeletonEdges[line_prev]), std::get<1>(skeletonEdges[lines_next[0]]));
        skeletonEdges[line_prev] = newEdge;
        skeletonEdges.erase(skeletonEdges.begin() + lines_next[0]);

        if (line_prev > lines_next[0]) {
            i--;
        }
    }

    std::vector<int> indegree(skeletonPoints.size(), 0);
    std::vector<int> outdegree(skeletonPoints.size(), 0);

    for (int s = 0; s < skeletonEdges.size(); s++) {
        int v1 = std::get<0>(skeletonEdges[s]);
        int v2 = std::get<1>(skeletonEdges[s]);
        outdegree[v1]++;
        indegree[v2]++;
    }

    for (int i = 0; i < skeletonPoints.size(); i++) {
        if (indegree[i] == 0 && outdegree[i] == 0) {
            skeletonPoints.erase(skeletonPoints.begin() + i);
            bin.erase(bin.begin() + i);
            indegree.erase(indegree.begin() + i);
            outdegree.erase(outdegree.begin() + i);
            for (int j = 0; j < skeletonEdges.size(); j++) {
                int v1 = std::get<0>(skeletonEdges[j]);
                int v2 = std::get<1>(skeletonEdges[j]);
                if (v1 > i) {
                    v1--;
                }
                if (v2 > i) {
                    v2--;
                }
                skeletonEdges[j] = std::make_tuple(v1, v2);
            }
            i--;
        }
    }
}
/*
void CombineEdgesOfSameDirections(pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int,int>>& skeletonEdges, std::vector<pcl::PointCloud<pcl::PointXYZ>>& bin) {
    for (int i = 0; i < skeletonEdges.size(); i++) {
        int line_prev = i;
        std::vector<int> lines_next;
        for (int j = 0; j < skeletonEdges.size(); j++) {
            if (std::get<1>(skeletonEdges[i]) == std::get<0>(skeletonEdges[j])) {
                lines_next.push_back(j);
            }
        }
        if (lines_next.size() != 1)
            continue;
        Eigen::Vector3f vec1{
            skeletonPoints[std::get<1>(skeletonEdges[line_prev])].x - skeletonPoints[std::get<0>(skeletonEdges[line_prev])].x,
            skeletonPoints[std::get<1>(skeletonEdges[line_prev])].y - skeletonPoints[std::get<0>(skeletonEdges[line_prev])].y,
            skeletonPoints[std::get<1>(skeletonEdges[line_prev])].z - skeletonPoints[std::get<0>(skeletonEdges[line_prev])].z
        };
        Eigen::Vector3f vec2{
            skeletonPoints[std::get<1>(skeletonEdges[lines_next[0]])].x - skeletonPoints[std::get<0>(skeletonEdges[lines_next[0]])].x,
            skeletonPoints[std::get<1>(skeletonEdges[lines_next[0]])].y - skeletonPoints[std::get<0>(skeletonEdges[lines_next[0]])].y,
            skeletonPoints[std::get<1>(skeletonEdges[lines_next[0]])].z - skeletonPoints[std::get<0>(skeletonEdges[lines_next[0]])].z
        };
        // Considering the numerical error
        pcl::PointXYZ next_start = skeletonPoints[std::get<0>(skeletonEdges[lines_next[0]])];
        pcl::PointXYZ next_end = skeletonPoints[std::get<1>(skeletonEdges[lines_next[0]])];
        pcl::PointXYZ ideal_next_end{
            next_start.x + vec1.normalized().x() * vec2.norm(),
            next_start.y + vec1.normalized().y() * vec2.norm(),
            next_start.z + vec1.normalized().z() * vec2.norm()
        };
        if (fabs(ideal_next_end.x - next_end.x) < 0.00001 && fabs(ideal_next_end.y - next_end.y) < 0.00001 && fabs(ideal_next_end.z - next_end.z) < 0.00001) {
            std::tuple<int, int> newEdge = std::make_tuple(std::get<0>(skeletonEdges[line_prev]), std::get<1>(skeletonEdges[lines_next[0]]));
            skeletonEdges[line_prev] = newEdge;
            skeletonEdges.erase(skeletonEdges.begin() + lines_next[0]);

            int iTransitionNode = std::get<1>(skeletonEdges[line_prev]);

            if (line_prev > lines_next[0]) {
                i--;
            }
        }
    }
    
    std::vector<int> indegree(skeletonPoints.size(), 0);
    std::vector<int> outdegree(skeletonPoints.size(), 0);

    for (int s = 0; s < skeletonEdges.size(); s++) {
        int v1 = std::get<0>(skeletonEdges[s]);
        int v2 = std::get<1>(skeletonEdges[s]);
        outdegree[v1]++;
        indegree[v2]++;
    }
    
    for (int i = 0; i < skeletonPoints.size(); i++) {
        if (indegree[i] == 0 && outdegree[i] == 0) {
            skeletonPoints.erase(skeletonPoints.begin() + i);
            bin.erase(bin.begin() + i);
            indegree.erase(indegree.begin() + i);
            outdegree.erase(outdegree.begin() + i);
            for (int j = 0; j < skeletonEdges.size(); j++) {
                int v1 = std::get<0>(skeletonEdges[j]);
                int v2 = std::get<1>(skeletonEdges[j]);
                if (v1 > i) {
                    v1--;
                }
                if (v2 > i) {
                    v2--;
                }
                skeletonEdges[j] = std::make_tuple(v1, v2);
            }
            i--;
        }
    }
}*/

void SmoothSkeleton(pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges, std::vector<std::vector<int>>& edgesInOffshoots, float fLambda = 0.0, int iTurn = 1) {
    while (iTurn > 0) {
        for (auto offshoot : edgesInOffshoots) {
            if (offshoot.size() < 2)
                continue;
            std::vector<int> offshoot_nodes;
            for (int i = 0; i < offshoot.size(); i++) {
                if (i == 0) {
                    offshoot_nodes.push_back(std::get<0>(skeletonEdges[offshoot[i]]));
                }
                offshoot_nodes.push_back(std::get<1>(skeletonEdges[offshoot[i]]));
            }

            std::vector<pcl::PointXYZ> from_head_to_tail(offshoot_nodes.size());
            for (int i = 0; i < offshoot_nodes.size(); i++) {
                from_head_to_tail[i] = skeletonPoints[offshoot_nodes[i]];
            }
            std::vector<pcl::PointXYZ> from_head_to_tail_opt(from_head_to_tail.size());
            if (fLambda > 0.0 || fLambda <= 1.0) {
                for (int i = 0; i < from_head_to_tail.size(); i++) {
                    if (i == 0 || i == from_head_to_tail.size() - 1) {
                        from_head_to_tail_opt[i] = from_head_to_tail[i];
                        continue;
                    }
                    Eigen::Vector3f L{
                        (float)(0.5 * (from_head_to_tail[i + 1].x - from_head_to_tail[i].x) + 0.5 * (from_head_to_tail[i - 1].x - from_head_to_tail[i].x)),
                        (float)(0.5 * (from_head_to_tail[i + 1].y - from_head_to_tail[i].y) + 0.5 * (from_head_to_tail[i - 1].y - from_head_to_tail[i].y)),
                        (float)(0.5 * (from_head_to_tail[i + 1].z - from_head_to_tail[i].z) + 0.5 * (from_head_to_tail[i - 1].z - from_head_to_tail[i].z))
                    };
                    from_head_to_tail_opt[i] = {
                        from_head_to_tail[i].x + fLambda * L.x(),
                        from_head_to_tail[i].y + fLambda * L.y(),
                        from_head_to_tail[i].z + fLambda * L.z()
                    };
                }
            }
            else {
                for (int i = 0; i < from_head_to_tail.size(); i++) {
                    if (i == 0 || i == from_head_to_tail.size() - 1) {
                        from_head_to_tail_opt[i] = from_head_to_tail[i];
                        continue;
                    }
                    from_head_to_tail_opt[i] = {
                        (from_head_to_tail[i - 1].x + from_head_to_tail[i].x + from_head_to_tail[i + 1].x) / 3,
                        (from_head_to_tail[i - 1].y + from_head_to_tail[i].y + from_head_to_tail[i + 1].y) / 3,
                        (from_head_to_tail[i - 1].z + from_head_to_tail[i].z + from_head_to_tail[i + 1].z) / 3
                    };
                }
            }
            for (int i = 1; i < from_head_to_tail_opt.size() - 1; i++) {
                skeletonPoints[offshoot_nodes[i]] = from_head_to_tail_opt[i];
            }
        }
        iTurn--;
    }
}

int WriteSkeletonToOBJ(pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges, std::string vtkFilePath) {
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    for (int i = 0; i < skeletonPoints.size(); i++) {
        points->InsertNextPoint(
            skeletonPoints[i].x,
            skeletonPoints[i].y,
            skeletonPoints[i].z
        );
    }
    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
    for (int s = 0; s < skeletonEdges.size(); s++) {
        vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
        int node_begin = std::get<0>(skeletonEdges[s]);
        int node_end = std::get<1>(skeletonEdges[s]);
        line->GetPointIds()->SetId(0, node_begin);
        line->GetPointIds()->SetId(1, node_end);
        lines->InsertNextCell(line);
    }
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);
    polyData->SetLines(lines);
    vtkSmartPointer<vtkOBJWriter> objWriter = vtkSmartPointer<vtkOBJWriter>::New();
    objWriter->SetFileName(vtkFilePath.c_str());
    objWriter->SetInputData(polyData);
    int status = objWriter->Write();
    return status;
}

void ViewBranchCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pcBranch) {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> clrTree(pcBranch, 128, 64, 0);
    viewer->addPointCloud<pcl::PointXYZ>(pcBranch, clrTree, "tree");
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