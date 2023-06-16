#include<iostream>
#include<vector>
#include<time.h>
#include<algorithm>
#include<string>

#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<pcl/search/kdtree.h>
#include<pcl/console/parse.h>
#include<pcl/segmentation/impl/extract_clusters.hpp>
#include<pcl/common/centroid.h>

#include<pcl/visualization/cloud_viewer.h>
#include<boost/thread/thread.hpp>

#include <Eigen/Eigen>
#include<vtkOBJReader.h>

#include <vtkSmartPointer.h>
#include <vtkLine.h>
#include <vtkCellArray.h>
#include <vtkTubeFilter.h>
#include <vtkLineSource.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkCylinderSource.h>
#include <vtkAutoInit.h>
#include <vtkCamera.h>
#include <vtkTriangle.h>
#include <vtkCleanPolyData.h>
#include <vtkTriangleFilter.h>
#include <vtkMassProperties.h>
#include <vtkOBJWriter.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Convex_hull_3.h>
#include <CGAL/polygon_mesh_processing.h>
#include <CGAL/convex_decomposition_3.h> 

VTK_MODULE_INIT(vtkRenderingOpenGL2); // VTK was built with vtkRenderingOpenGL2
VTK_MODULE_INIT(vtkInteractionStyle);

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

void ReadSkeletonFromOBJ(std::string strOBJFilePath, pcl::PointCloud<pcl::PointXYZ>& skeletonPoints, std::vector<std::tuple<int, int>>& skeletonEdges) {
	vtkSmartPointer<vtkOBJReader> reader = vtkSmartPointer<vtkOBJReader>::New();
	reader->SetFileName(strOBJFilePath.c_str());
	reader->Update();

	vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
	polyData = (vtkPolyData*)reader->GetOutputDataObject(0);

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
	points = polyData->GetPoints();
	lines = polyData->GetLines();

	for (int i = 0; i < points->GetNumberOfPoints(); i++) {
		double* point = points->GetPoint(i);
		skeletonPoints.push_back({ (float)point[0],(float)point[1],(float)point[2] });
	}

	vtkIdType npts, * pts_id;// npts is number of vertices in a cell, pts_id is ids of npts vertices
	while (lines->GetNextCell(npts, pts_id)) {
		int node_begin = pts_id[0];
		int node_end = pts_id[1];
		skeletonEdges.push_back(std::make_tuple(node_begin, node_end));
	}
}

void OpenPCD(std::string strFilePath, pcl::PointCloud<pcl::PointXYZ>& pcCloud) {
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(strFilePath, pcCloud) == -1) {
		PCL_ERROR("[ERROR]Cloud not read file %s\n", strFilePath);
		exit(-1);
	}
}

void FitCircleByLeastSquares(pcl::PointCloud<pcl::PointXY> points, pcl::PointXY& center, float& radius)
{
    double X1 = 0;
    double Y1 = 0;
    double X2 = 0;
    double Y2 = 0;
    double X3 = 0;
    double Y3 = 0;
    double X1Y1 = 0;
    double X1Y2 = 0;
    double X2Y1 = 0;

    for (int i = 0; i < points.size(); i++)
    {
        X1 = X1 + points[i].x;
        Y1 = Y1 + points[i].y;
        X2 = X2 + points[i].x * points[i].x;
        Y2 = Y2 + points[i].y * points[i].y;
        X3 = X3 + points[i].x * points[i].x * points[i].x;
        Y3 = Y3 + points[i].y * points[i].y * points[i].y;
        X1Y1 = X1Y1 + points[i].x * points[i].y;
        X1Y2 = X1Y2 + points[i].x * points[i].y * points[i].y;
        X2Y1 = X2Y1 + points[i].x * points[i].x * points[i].y;
    }

    double C, D, E, G, H, N;
    double a, b, c;
    N = points.size();
    C = N * X2 - X1 * X1;
    D = N * X1Y1 - X1 * Y1;
    E = N * X3 + N * X1Y2 - (X2 + Y2) * X1;
    G = N * Y2 - Y1 * Y1;
    H = N * X2Y1 + N * Y3 - (X2 + Y2) * Y1;
    a = (H * D - E * G) / (C * G - D * D);
    b = (H * C - E * D) / (D * D - G * C);
    c = -(a * X1 + b * Y1 + X2 + Y2) / N;

    double A = a / (-2);
    double B = b / (-2);
    double R = sqrt(a * a + b * b - 4 * c) / 2;
    center = {
        (float)A,
        (float)B
    };
    radius = (float)R;
}
double DistanceOfPointToLine(pcl::PointXYZ a, pcl::PointXYZ b, pcl::PointXYZ s)
{
    double ab = sqrt(pow((a.x - b.x), 2.0) + pow((a.y - b.y), 2.0) + pow((a.z - b.z), 2.0));
    double as = sqrt(pow((a.x - s.x), 2.0) + pow((a.y - s.y), 2.0) + pow((a.z - s.z), 2.0));
    double bs = sqrt(pow((s.x - b.x), 2.0) + pow((s.y - b.y), 2.0) + pow((s.z - b.z), 2.0));
    double cos_A = (pow(as, 2.0) + pow(ab, 2.0) - pow(bs, 2.0)) / (2 * ab * as);
    double sin_A = sqrt(1 - pow(cos_A, 2.0));
    return as * sin_A;
}

float GetDistanceBetween2pts(pcl::PointXYZ p1, pcl::PointXYZ p2) {
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
}

float ComputeSupportLength(int vertex,pcl::PointCloud<pcl::PointXYZ> skeletonPoints, std::vector<std::tuple<int, int>> skeletonEdges, float& total_len) {
    std::vector<int> nextVertices;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<0>(skeletonEdges[i]) == vertex) {
            total_len += GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[i])],skeletonPoints[std::get<1>(skeletonEdges[i])]);
            nextVertices.push_back(std::get<1>(skeletonEdges[i]));
        }
    }
    for (int i = 0; i < nextVertices.size(); i++) {
        ComputeSupportLength(nextVertices[i], skeletonPoints, skeletonEdges, total_len);
    }
}


void GetTreetopFromAVertex(int vertex, std::vector<std::tuple<int, int>> skeletonEdges, std::vector<int>& treetop_vertices) {
    std::vector<int> nextVertices;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        int begin_vertex = std::get<0>(skeletonEdges[i]);
        int end_vertex = std::get<1>(skeletonEdges[i]);
        if (begin_vertex == vertex) {
            nextVertices.push_back(end_vertex);
        }
    }
    if (nextVertices.size() == 0) {
        treetop_vertices.push_back(vertex);
        return;
    }
    else {
        for (auto each_vertex : nextVertices) {
            GetTreetopFromAVertex(each_vertex, skeletonEdges, treetop_vertices);
        }
    }
}

void ComputeDistanceBetweenTwoVertices(int vertex_start, int vertex_end, pcl::PointCloud<pcl::PointXYZ> skeletonPoints, std::vector<std::tuple<int, int>> skeletonEdges, float& distance) {
    if (vertex_start == vertex_end)
        return;
    int parent_vertex_end;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<1>(skeletonEdges[i]) == vertex_end) {
            parent_vertex_end = std::get<0>(skeletonEdges[i]);
            distance += GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[i])], skeletonPoints[std::get<1>(skeletonEdges[i])]);
            break;
        }
    }
    

    ComputeDistanceBetweenTwoVertices(vertex_start, parent_vertex_end, skeletonPoints, skeletonEdges, distance);
}

void EdgesBetweenTwoVertices(int vertex_start, int vertex_end, pcl::PointCloud<pcl::PointXYZ> skeletonPoints, std::vector<std::tuple<int, int>> skeletonEdges, std::vector<int>& edges) {
    if (vertex_start == vertex_end)
        return;

    int parent_vertex_end;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<1>(skeletonEdges[i]) == vertex_end) {
            parent_vertex_end = std::get<0>(skeletonEdges[i]);
            edges.push_back(i);
            break;
        }
    }

    EdgesBetweenTwoVertices(vertex_start, parent_vertex_end, skeletonPoints, skeletonEdges, edges);
}

void RecalculateRadius(int edge, pcl::PointCloud<pcl::PointXYZ> skeletonPoints, std::vector<std::tuple<int, int>> skeletonEdges, std::vector<float>& radii) {
    std::vector<int> nextEdges;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<1>(skeletonEdges[edge]) == std::get<0>(skeletonEdges[i])) {
            nextEdges.push_back(i);
        }
    }
    switch (nextEdges.size()) {
    case 0:
        return;
    case 1: {
        int parent_vertex = std::get<1>(skeletonEdges[edge]);
        std::vector<int> treetop_vertices_from_parent_vertex;
        GetTreetopFromAVertex(parent_vertex, skeletonEdges, treetop_vertices_from_parent_vertex);

        float lp = 0;
        std::set<int> all_edges_from_parent;
        for (auto each_treetop_vertex : treetop_vertices_from_parent_vertex) {
            std::vector<int> edges;
            EdgesBetweenTwoVertices(parent_vertex, each_treetop_vertex, skeletonPoints, skeletonEdges, edges);
            all_edges_from_parent.insert(edges.begin(), edges.end());
        }
        for (auto each_edge : all_edges_from_parent) {
            lp += GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[each_edge])], skeletonPoints[std::get<1>(skeletonEdges[each_edge])]);
        }

        lp += 0.5 * GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[edge])], skeletonPoints[std::get<1>(skeletonEdges[edge])]);

        float lc = lp - 0.5 * GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[edge])], skeletonPoints[std::get<1>(skeletonEdges[edge])]) - 0.5 * GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[nextEdges[0]])], skeletonPoints[std::get<1>(skeletonEdges[nextEdges[0]])]);
        
        /*
        float lc = 0;
        
        int son_vertex = std::get<1>(skeletonEdges[nextEdges[0]]);
        std::vector<int> treetop_vertices_from_a_son_vertex;
        GetTreetopFromAVertex(son_vertex, skeletonEdges, treetop_vertices_from_a_son_vertex);

        std::set<int> all_edges_from_son;

        for (auto each_treetop_vertex : treetop_vertices_from_a_son_vertex) {
            std::vector<int> edges;
            EdgesBetweenTwoVertices(son_vertex, each_treetop_vertex, skeletonPoints, skeletonEdges, edges);
            all_edges_from_son.insert(edges.begin(), edges.end());
        }

        for (auto each_edge : all_edges_from_son) {
            lc += GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[each_edge])], skeletonPoints[std::get<1>(skeletonEdges[each_edge])]);
        }

        lc += 0.5 * GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[nextEdges[0]])], skeletonPoints[std::get<1>(skeletonEdges[nextEdges[0]])]);
        */

        float rp = radii[edge];
        float rc = rp * pow(lc / lp, 3.0 / 2.0);

        radii[nextEdges[0]] = fmax(rc,0.001);

        RecalculateRadius(nextEdges[0], skeletonPoints, skeletonEdges, radii);
        break;
    }
    default: {
        int parent_vertex = std::get<1>(skeletonEdges[edge]);
        std::vector<int> treetop_vertices_from_parent_vertex;
        GetTreetopFromAVertex(parent_vertex, skeletonEdges, treetop_vertices_from_parent_vertex);

        float lp = 0;
        std::set<int> all_edges_from_parent;
        for (auto each_treetop_vertex : treetop_vertices_from_parent_vertex) {
            std::vector<int> edges;
            EdgesBetweenTwoVertices(parent_vertex, each_treetop_vertex, skeletonPoints, skeletonEdges, edges);
            all_edges_from_parent.insert(edges.begin(), edges.end());
        }
        for (auto each_edge : all_edges_from_parent) {
            lp += GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[each_edge])], skeletonPoints[std::get<1>(skeletonEdges[each_edge])]);
        }

        lp += 0.5 * GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[edge])], skeletonPoints[std::get<1>(skeletonEdges[edge])]);
        
        
        std::vector<float> lci(nextEdges.size());

        for (int i = 0; i < nextEdges.size(); i++) {
            int a_son_vertex = std::get<1>(skeletonEdges[nextEdges[i]]);
            std::vector<int> treetop_vertices_from_a_son_vertex;
            GetTreetopFromAVertex(a_son_vertex, skeletonEdges, treetop_vertices_from_a_son_vertex);

            float lc = 0;
            std::set<int> all_edges_from_son;

            for (auto each_treetop_vertex : treetop_vertices_from_a_son_vertex) {
                std::vector<int> edges;
                EdgesBetweenTwoVertices(a_son_vertex, each_treetop_vertex, skeletonPoints, skeletonEdges, edges);
                all_edges_from_son.insert(edges.begin(), edges.end());
            }

            for (auto each_edge : all_edges_from_son) {
                lc += GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[each_edge])], skeletonPoints[std::get<1>(skeletonEdges[each_edge])]);
            }

            lc += 0.5 * GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[nextEdges[i]])], skeletonPoints[std::get<1>(skeletonEdges[nextEdges[i]])]);

            lci[i] = lc;
        }

        float rp = radii[edge];
        std::vector<float> rci(nextEdges.size());
        for (int i = 0; i < nextEdges.size(); i++) {
            //rci[i] = rp * pow(lci[i] / std::accumulate(lci.begin(), lci.end(), 0.0), 1.0 / 2.49);
            rci[i] = rp * pow(lci[i] / lp, 1.0 / 2.49);

        }
        for (int i = 0; i < nextEdges.size(); i++) {
            radii[nextEdges[i]] = fmax(rci[i],0.001);
        }
        
        for (int i = 0; i < nextEdges.size(); i++) {
            RecalculateRadius(nextEdges[i], skeletonPoints, skeletonEdges, radii);
        }
        break;
    }
    }
}

void CalculateRadius(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ> skeletonPoints, std::vector<std::tuple<int, int>> skeletonEdges, std::vector<float>& radii, float range, bool bLS = false) {
    for (int i = 0; i < skeletonEdges.size(); i++) {
        pcl::PointXYZ ptBegin = skeletonPoints[std::get<0>(skeletonEdges[i])];
        pcl::PointXYZ ptEnd = skeletonPoints[std::get<1>(skeletonEdges[i])];

        pcl::PointXYZ ptMiddle = {
            (float)(0.5 * (ptBegin.x + ptEnd.x)),
            (float)(0.5 * (ptBegin.y + ptEnd.y)),
            (float)(0.5 * (ptBegin.z + ptEnd.z))
        };

        Eigen::Vector3f vDirection{
            ptEnd.x - ptBegin.x,
            ptEnd.y - ptBegin.y,
            ptEnd.z - ptBegin.z
        };
        vDirection.normalize();

        float A = vDirection.x();
        float B = vDirection.y();
        float C = vDirection.z();
        float D = -A * ptMiddle.x - B * ptMiddle.y - C * ptMiddle.z;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cutted_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (int j = 0; j < cloud->size(); j++) {
            if (abs(A * cloud->points[j].x + B * cloud->points[j].y + C * cloud->points[j].z + D) / sqrt(A * A + B * B + C * C) <= range / 2) {
                cutted_cloud->push_back(cloud->points[j]);
            }
        }

        // Find the nearest cluster 
        ECE* ece(new ECE(*cutted_cloud, range, 1));
        std::vector<pcl::PointIndices> inlier = ece->GetIndices();

        pcl::PointCloud<pcl::PointXYZ>::Ptr local_branch_segment(new  pcl::PointCloud<pcl::PointXYZ>);

        float minimum_sqr_dist = INT_MAX;
        for (int j = 0; j < inlier.size(); j++) {
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cutted_cloud, inlier[j], centroid);
            float current_sqr_dist = pow(centroid.x() - ptMiddle.x, 2) + pow(centroid.y() - ptMiddle.y, 2) + pow(centroid.z() - ptMiddle.z, 2);
            if (current_sqr_dist < minimum_sqr_dist) {
                minimum_sqr_dist = current_sqr_dist;
                local_branch_segment->clear();
                pcl::copyPointCloud(*cutted_cloud, inlier[j], *local_branch_segment);
            }
        }

        // Get radius after rotation
        float x_l_min = INT_MAX;
        float y_l_min = INT_MAX;
        float z_l_min = INT_MAX;
        float x_l_max = INT_MIN;
        float y_l_max = INT_MIN;
        float z_l_max = INT_MIN;

        for (int t = 0; t < local_branch_segment->size(); t++) {
            x_l_min = fmin(x_l_min, local_branch_segment->points[t].x);
            y_l_min = fmin(y_l_min, local_branch_segment->points[t].y);
            z_l_min = fmin(z_l_min, local_branch_segment->points[t].z);

            x_l_max = fmax(x_l_max, local_branch_segment->points[t].x);
            y_l_max = fmax(y_l_max, local_branch_segment->points[t].y);
            z_l_max = fmin(z_l_max, local_branch_segment->points[t].z);
        }
        Eigen::Vector3f centroid{
            (float)(0.5 * (x_l_min + x_l_max)),
            (float)(0.5 * (y_l_min + y_l_max)),
            (float)(0.5 * (z_l_min + z_l_max))
        };

        //Eigen::Vector4f centroid;
        //pcl::compute3DCentroid(*local_branch_segment, centroid);
        pcl::PointXYZ ptNewBegin = {
            centroid.x(),
            centroid.y(),
            centroid.z()
        };
        pcl::PointXYZ ptNewEnd = {
            ptNewBegin.x + ptEnd.x - ptBegin.x,
            ptNewBegin.y + ptEnd.y - ptBegin.y,
            ptNewBegin.z + ptEnd.z - ptBegin.z
        };
        
        if (bLS == false) {
            float dis = 0.0;
            for (int j = 0; j < local_branch_segment->size(); j++) {
                dis+= (float)DistanceOfPointToLine(ptNewBegin, ptNewEnd, local_branch_segment->points[j]);
            }
            dis /= local_branch_segment->size();
            radii[i] = fmax(dis, 0.001);
            
            // IDW will make the branch "thinner" than the normal
            /* Use distances between points and the axis to get a radius 
              Original method adopts directly averaging distance. 
              In this work, we consider that Inverse Distance Weight(IDW) can be used to make distances more reasonable
              Because when encountering a bifurcation, distances between the closest part of points and the centerline can be averaged to get the real radius,
              and the farther the distance is, the smaller the weight is
            */
            /*
            std::vector<float> dist_list(local_branch_segment->size());
            for (int j = 0; j < local_branch_segment->size(); j++) {
                dist_list[j] = (float)DistanceOfPointToLine(ptNewBegin, ptNewEnd, local_branch_segment->points[j]);
            }

            float dist_max = *std::max_element(dist_list.begin(), dist_list.end());
            float dist_min = *std::min_element(dist_list.begin(), dist_list.end());
            //IDW
            std::vector<float> weights(dist_list.size());
            for (int j = 0; j < dist_list.size(); j++) {
                weights[j] = pow((dist_max - dist_list[j]) / (dist_max - dist_min), 2);
            }

            float r = 0;
            for (int j = 0; j < dist_list.size(); j++) {
                r += dist_list[j] * weights[j];
            }
            r /= std::accumulate(weights.begin(), weights.end(), 0.0);
            radii[i] = fmax(r, 0.001);*/
        }
        else {
            //Use fitting planar circles with least squares to get a radius
            Eigen::Vector3f z_positive{ 0,0,1 };
            Eigen::Matrix3f rot_mat;
            rot_mat = Eigen::Quaternionf::FromTwoVectors(vDirection, z_positive).toRotationMatrix();

            pcl::PointCloud<pcl::PointXY> cloud_for_fitting;

            for (int j = 0; j < local_branch_segment->size(); j++) {
                Eigen::Vector3f old_point{
                    local_branch_segment->points[j].x - centroid.x(),
                    local_branch_segment->points[j].y - centroid.y(),
                    local_branch_segment->points[j].z - centroid.z(),
                };
                Eigen::Vector3f new_point;
                new_point = rot_mat * old_point;
                cloud_for_fitting.push_back(
                    { old_point.x(),old_point.y() }
                );
            }

            pcl::PointXY center;
            float r = 0;
            FitCircleByLeastSquares(cloud_for_fitting, center, r);
            radii[i] = fmax(r, 0.001);
        }
    }
}

void ReverselyDeriveRadius(int edge, pcl::PointCloud<pcl::PointXYZ> skeletonPoints, std::vector<std::tuple<int, int>> skeletonEdges, std::vector<float>& radii) {
    int prev_edge = -1;
    int edge_in_vertex = std::get<0>(skeletonEdges[edge]);
    std::vector<int> edges_with_same_in_vertex;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<1>(skeletonEdges[i]) == edge_in_vertex) {
            prev_edge = i;
        }
        if (std::get<0>(skeletonEdges[i]) == edge_in_vertex) {
            edges_with_same_in_vertex.push_back(i);
        }
    }
    if (prev_edge == -1)
        return;

    switch (edges_with_same_in_vertex.size()) {
    case 1: {
        int son_vertex = std::get<1>(skeletonEdges[edge]);
        std::vector<int> treetop_vertices_from_son_vertex;
        GetTreetopFromAVertex(son_vertex, skeletonEdges, treetop_vertices_from_son_vertex);

        float lc = 0;
        std::set<int> all_edges_from_son;
        for (auto each_treetop_vertex : treetop_vertices_from_son_vertex) {
            std::vector<int> edges;
            EdgesBetweenTwoVertices(son_vertex, each_treetop_vertex, skeletonPoints, skeletonEdges, edges);
            all_edges_from_son.insert(edges.begin(), edges.end());
        }
        for (auto each_edge : all_edges_from_son) {
            lc += GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[each_edge])], skeletonPoints[std::get<1>(skeletonEdges[each_edge])]);
        }

        lc+= 0.5* GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[edge])], skeletonPoints[std::get<1>(skeletonEdges[edge])]);

        float lp = lc + 0.5 * GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[edge])], skeletonPoints[std::get<1>(skeletonEdges[edge])]) + 0.5 * GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[prev_edge])], skeletonPoints[std::get<1>(skeletonEdges[prev_edge])]);

        float rc = radii[edge];
        float rp = rc / pow(lc / lp, 3.0 / 2);
        radii[prev_edge] = fmax(rp,0.001);

        break;
    }
    default: {
        float rp = 0;
        for (auto each_edge : edges_with_same_in_vertex) {
            float rci = radii[each_edge];
            rp += pow(rci, 2.49);
        }
        rp = pow(rp, 1.0 / 2.49);
        radii[prev_edge] = fmax(rp,0.001);

        break;
    }
    }

    ReverselyDeriveRadius(prev_edge, skeletonPoints, skeletonEdges, radii);
}

void BranchLevelRecognize(int vertex, std::vector<std::tuple<int, int>> skeletonEdges, std::vector<float> radii, std::vector<int>& levels, int level = 0) {
    std::vector<int> edges;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<0>(skeletonEdges[i]) == vertex) {
            edges.push_back(i);
        }
    }
    switch (edges.size()) {
    case 0:
        return;
    case 1: {
        int next_vertex = std::get<1>(skeletonEdges[edges[0]]);
        levels[edges[0]] = level;
        BranchLevelRecognize(next_vertex, skeletonEdges, radii, levels, level);
        break;
    }
    default: {
        int prev_edges = -1;
        for (int i = 0; i < skeletonEdges.size(); i++) {
            if (std::get<1>(skeletonEdges[i]) == vertex) {
                prev_edges = i;
                break;
            }
        }
        
        int trunk_idx;
        if (prev_edges != -1) {
            float rp = radii[prev_edges];
            std::vector<float> angles(edges.size());
            for (int i = 0; i < edges.size(); i++) {
                float rc = radii[edges[i]];
                float theta = acos((pow(rp, 4) + pow(rc, 4) - pow(pow(rp, 3) - pow(rc, 3), 4.0 / 3)) / (2 * pow(rp, 2) * pow(rc, 2)));
                angles[i] = theta;
            }
            trunk_idx = std::distance(angles.begin(), std::min_element(angles.begin(), angles.end()));
        }
        else {
            //When the first vertex in the skeleton is a bifurcation point
            std::vector<float> local_radii(edges.size());
            for (int i = 0; i < edges.size(); i++) {
                local_radii[i] = radii[edges[i]];
            }
            trunk_idx = std::distance(local_radii.begin(), std::max_element(local_radii.begin(), local_radii.end()));
        }
        for (int i = 0; i < edges.size(); i++) {
            if (i == trunk_idx) {
                levels[edges[i]] = level;
                int next_vertex = std::get<1>(skeletonEdges[edges[i]]);
                BranchLevelRecognize(next_vertex, skeletonEdges, radii, levels, level);
            }
            else {
                levels[edges[i]] = level + 1;
                int next_vertex = std::get<1>(skeletonEdges[edges[i]]);
                BranchLevelRecognize(next_vertex, skeletonEdges, radii, levels, level + 1);
            }
        }
        break;
    }
    }
}

/*int VTKBranchModelingByCylinder(std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ>>& axis, std::vector<float>& radius, std::vector<std::tuple<float, float, float>>& colors, int edgeNum = 50)
{
    vtkSmartPointer<vtkRenderer> renderer =
        vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow =
        vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(1280, 720);
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    for (int i = 0; i < axis.size(); i++) {

        pcl::PointXYZ ptBegin = std::get<0>(axis[i]);
        pcl::PointXYZ ptEnd = std::get<1>(axis[i]);

        vtkSmartPointer<vtkLineSource> lineSource =
            vtkSmartPointer<vtkLineSource>::New();
        lineSource->SetPoint1(ptBegin.x, ptBegin.y, ptBegin.z);
        lineSource->SetPoint2(ptEnd.x, ptEnd.y, ptEnd.z);
        vtkSmartPointer<vtkTubeFilter> tubeFilter = vtkSmartPointer<vtkTubeFilter>::New();
        tubeFilter->SetInputConnection(lineSource->GetOutputPort());
        tubeFilter->SetRadius(radius[i]);
        tubeFilter->SetNumberOfSides(edgeNum);
        tubeFilter->CappingOn();

        vtkSmartPointer<vtkPolyDataMapper> cylinderMapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
        cylinderMapper->SetInputConnection(tubeFilter->GetOutputPort());

        vtkSmartPointer<vtkActor> cylinderActor =
            vtkSmartPointer<vtkActor>::New();
        cylinderActor->SetMapper(cylinderMapper);

        std::tuple<float, float, float> color = colors[i];

        cylinderActor->GetProperty()->SetColor(std::get<0>(color), std::get<1>(color), std::get<2>(color));

        renderer->AddActor(cylinderActor);
    }

    renderer->SetBackground(1.0, 1.0, 1.0);
    //renderer->GetActiveCamera()->SetViewUp(0.0,1.0,0.0);
    renderWindow->Render();
    renderWindowInteractor->Start();

    return 0;
}*/

int PCLBranchModelingByCylinder(pcl::PointCloud<pcl::PointXYZ> skeletonPoints, std::vector<std::tuple<int,int>> skeletonEdges, std::vector<float>& radii, std::vector<std::tuple<float, float, float>>& colors, int edgeNum = 50)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);

    for (int i = 0; i < skeletonEdges.size(); i++) {
        pcl::PointXYZ ptBegin = skeletonPoints[std::get<0>(skeletonEdges[i])];
        pcl::PointXYZ ptEnd = skeletonPoints[std::get<1>(skeletonEdges[i])];

        vtkSmartPointer<vtkLineSource> lineSource =
            vtkSmartPointer<vtkLineSource>::New();
        lineSource->SetPoint1(ptBegin.x, ptBegin.y, ptBegin.z);
        lineSource->SetPoint2(ptEnd.x, ptEnd.y, ptEnd.z);
        vtkSmartPointer<vtkTubeFilter> tubeFilter = vtkSmartPointer<vtkTubeFilter>::New();
        tubeFilter->SetInputConnection(lineSource->GetOutputPort());
        tubeFilter->SetRadius(radii[i]);
        tubeFilter->SetNumberOfSides(edgeNum);
        tubeFilter->CappingOn();
        tubeFilter->Update();

        vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
        polyData = tubeFilter->GetOutput();
        vtkSmartPointer<vtkUnsignedCharArray> ptColor =
            vtkSmartPointer<vtkUnsignedCharArray>::New();
        ptColor->SetNumberOfComponents(3);
        unsigned char color[3] = {
                (int)(std::get<0>(colors[i]) * 255),
                (int)(std::get<1>(colors[i]) * 255),
                (int)(std::get<2>(colors[i]) * 255)
        };
        
        for (int j = 0; j < polyData->GetNumberOfPoints(); j++) {
            ptColor->InsertNextTypedTuple(color);
        }
        polyData->GetPointData()->SetScalars(ptColor);

        std::ostringstream ostr;
        ostr << i;
        viewer->addModelFromPolyData(polyData, ostr.str());
    }

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));

    }

    return 0;
}

void ComputeBranchParameter(pcl::PointCloud<pcl::PointXYZ> skeletonPoints, std::vector<std::tuple<int, int>> skeletonEdges, std::vector<float> radii, float z_ground, std::vector<int> edges_on_a_branch, pcl::PointXYZ& treetop_point, float& BL, float& BCL, float& BD, float& BH, float& BAH, float& IA, float& azimuth) {
    pcl::PointXYZ base_point = skeletonPoints[std::get<0>(skeletonEdges[edges_on_a_branch.front()])];

    treetop_point = skeletonPoints[std::get<1>(skeletonEdges[edges_on_a_branch.back()])];

    BCL = GetDistanceBetween2pts(base_point, treetop_point);

    BD = 2 * radii[edges_on_a_branch.front()];

    BL = 0;
    BAH = 0;

    Eigen::Vector3f z_positive = { 0,0,1 };

    for (int i = 0; i < edges_on_a_branch.size(); i++) {
        pcl::PointXYZ ptBegin = skeletonPoints[std::get<0>(skeletonEdges[edges_on_a_branch[i]])];
        pcl::PointXYZ ptEnd = skeletonPoints[std::get<1>(skeletonEdges[edges_on_a_branch[i]])];
        BL += GetDistanceBetween2pts(ptBegin, ptEnd);
        if(i != 0)
            BAH = fmax(BAH, DistanceOfPointToLine(base_point, treetop_point, ptBegin));
        if (i != edges_on_a_branch.size() - 1)
            BAH = fmax(BAH, DistanceOfPointToLine(base_point, treetop_point, ptEnd));
    }

    Eigen::Vector3f first_direction = {
        skeletonPoints[std::get<1>(skeletonEdges[edges_on_a_branch.front()])].x - skeletonPoints[std::get<0>(skeletonEdges[edges_on_a_branch.front()])].x,
        skeletonPoints[std::get<1>(skeletonEdges[edges_on_a_branch.front()])].y - skeletonPoints[std::get<0>(skeletonEdges[edges_on_a_branch.front()])].y,
        skeletonPoints[std::get<1>(skeletonEdges[edges_on_a_branch.front()])].z - skeletonPoints[std::get<0>(skeletonEdges[edges_on_a_branch.front()])].z
    };

    float angle_between_first_direction_and_z_positive = acos(abs(first_direction.dot(z_positive) / first_direction.norm() / z_positive.norm()));
    BH = base_point.z - (angle_between_first_direction_and_z_positive != 0.0 ? radii[edges_on_a_branch.front()] / sin(angle_between_first_direction_and_z_positive) : 0) - z_ground;

    Eigen::Vector3f main_direction = {
        treetop_point.x - base_point.x,
        treetop_point.y - base_point.y,
        treetop_point.z - base_point.z
    };

    float angle_between_main_direction_and_z_positive = acos(main_direction.dot(z_positive) / main_direction.norm() / z_positive.norm()) * 180 / M_PI;
    IA = angle_between_main_direction_and_z_positive > 90 ? angle_between_main_direction_and_z_positive - 90: 90 - angle_between_main_direction_and_z_positive;


    Eigen::Vector2f y_positive = { 0,1 };
    Eigen::Vector2f main_direction_xoy = { main_direction.x(), main_direction.y() };

    float angle_between_main_direction_xoy_and_y_positive = acos(y_positive.dot(main_direction_xoy) / y_positive.norm() / main_direction_xoy.norm()) * 180 / M_PI;

    azimuth = main_direction.x() <= 0 ? angle_between_main_direction_xoy_and_y_positive : 360 - angle_between_main_direction_xoy_and_y_positive;
}

//Print_info: Treetop_point | Level, Parent branch index | Branch Length(BL), Branch Chord Length(BCL), Branch Diameter(BD), Branch Height(BH), Branch Arc Height(BAH), Inclination Angle(IA), Azimuth, [Axil Angle - Skeleton(AA), Axil Angle - Murray's law (AAM),  Branching Angle(BA) <NOT USED>Local Relative Inclination Angle - Murray's law(LRIAM), Local Inclination Angle LIA]
//The north direction of the azimuth reference is the positive direction of the y-axis. If the point cloud coordinate system is not a projection coordinate system, it is necessary to measure the angle between the true north direction and the current north direction and process the results.
void SeparateBranches(int edge, pcl::PointCloud<pcl::PointXYZ> skeletonPoints, std::vector<std::tuple<int, int>> skeletonEdges, std::vector<float> radii, float z_ground, int level, std::vector<int> levels, std::vector<std::vector<int>>& edges_on_branches, std::vector<std::tuple<pcl::PointXYZ, std::vector<int>, std::vector<float>>>& print_info, Eigen::Vector3f parentDirection = Eigen::Vector3f::Zero(), std::vector<int> edges_on_a_branch = std::vector<int>(), int parent_branch_idx = -1) {
    std::vector<int> following_edges;
    std::vector<int> following_edges_level;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<0>(skeletonEdges[i]) == std::get<1>(skeletonEdges[edge])) {
            following_edges.push_back(i);
            following_edges_level.push_back(levels[i]);
        }
    }
    switch (following_edges.size()) {
    case 0: {
        edges_on_a_branch.push_back(edge);
        edges_on_branches.push_back(edges_on_a_branch);

        std::vector<int> complete_branch_edges = edges_on_branches.back();
        pcl::PointXYZ treetop_point;

        float BL;
        float BCL;
        float BD;
        float BH;
        float BAH;
        float IA;
        float azimuth;
        ComputeBranchParameter(skeletonPoints, skeletonEdges, radii, z_ground, complete_branch_edges, treetop_point, BL, BCL, BD, BH, BAH, IA, azimuth);

        std::vector<int> layer_info;
        layer_info.push_back(level);
        layer_info.push_back(parent_branch_idx);

        std::vector<float> branch_parameters;
        branch_parameters.push_back(BL);
        branch_parameters.push_back(BCL);
        branch_parameters.push_back(BD);
        branch_parameters.push_back(BH);
        branch_parameters.push_back(BAH);
        branch_parameters.push_back(IA);
        branch_parameters.push_back(azimuth);

        print_info.push_back(std::make_tuple(treetop_point, layer_info, branch_parameters));
        return;
    }
    case 1: {
        edges_on_a_branch.push_back(edge);
        SeparateBranches(following_edges[0], skeletonPoints, skeletonEdges, radii, z_ground, level, levels, edges_on_branches, print_info, parentDirection, edges_on_a_branch, parent_branch_idx);

        break;
    }
    default: {
        std::vector<std::tuple<int, int>> combined_edges_and_levels;
        for (int i = 0; i < following_edges.size(); i++) {
            combined_edges_and_levels.push_back(std::make_tuple(following_edges[i], following_edges_level[i]));
        }
        std::sort(combined_edges_and_levels.begin(), combined_edges_and_levels.end(), [](auto& e1, auto& e2) {
            return std::get<1>(e1) < std::get<1>(e2);
            });
        Eigen::Vector3f main_local_direction;
        int new_parent_branch_idx = edges_on_branches.size();

        for (int i = 0; i < combined_edges_and_levels.size(); i++) {
            int following_edge = std::get<0>(combined_edges_and_levels[i]);
            if (i == 0) {
                edges_on_a_branch.push_back(edge);
                SeparateBranches(following_edge, skeletonPoints, skeletonEdges, radii, z_ground, level, levels, edges_on_branches, print_info, parentDirection, edges_on_a_branch, parent_branch_idx);

                Eigen::Vector3f v1 = {
                    skeletonPoints[std::get<1>(skeletonEdges[edge])].x - skeletonPoints[std::get<0>(skeletonEdges[edge])].x,
                    skeletonPoints[std::get<1>(skeletonEdges[edge])].y - skeletonPoints[std::get<0>(skeletonEdges[edge])].y,
                    skeletonPoints[std::get<1>(skeletonEdges[edge])].z - skeletonPoints[std::get<0>(skeletonEdges[edge])].z
                };
                Eigen::Vector3f v2 = {
                    skeletonPoints[std::get<1>(skeletonEdges[following_edge])].x - skeletonPoints[std::get<0>(skeletonEdges[following_edge])].x,
                    skeletonPoints[std::get<1>(skeletonEdges[following_edge])].y - skeletonPoints[std::get<0>(skeletonEdges[following_edge])].y,
                    skeletonPoints[std::get<1>(skeletonEdges[following_edge])].z - skeletonPoints[std::get<0>(skeletonEdges[following_edge])].z
                };
                main_local_direction = (v1 + v2).normalized();

                if (level != 0) {

                    std::vector<int> complete_branch_edges = edges_on_branches.back();

                    Eigen::Vector3f first_sub_direction = {
                        skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.front()])].x - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].x,
                        skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.front()])].y - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].y,
                        skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.front()])].z - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].z
                    };
                    Eigen::Vector3f main_sub_direction = {
                        skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.back()])].x - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].x,
                        skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.back()])].y - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].y,
                        skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.back()])].z - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].z
                    };

                    float AA = acos(first_sub_direction.dot(parentDirection) / first_sub_direction.norm() / parentDirection.norm()) * 180 / M_PI;
                    
                    
                    int main_begin_edge;
                    int off_begin_edge = complete_branch_edges.front();
                    int off_begin_vertex = std::get<0>(skeletonEdges[off_begin_edge]);

                    for (int t = 0; t < skeletonEdges.size(); t++) {
                        if (std::get<0>(skeletonEdges[t]) == off_begin_vertex && levels[t] < level) {
                            main_begin_edge = t;
                            break;
                        }
                    }

                    float r1 = radii[main_begin_edge];
                    float r2 = radii[off_begin_edge];
                    float AAM = acos((pow(pow(r1, 3) + pow(r2, 3), 4.0 / 3.0) - pow(r1, 4) - pow(r2, 4)) / (2 * pow(r1, 2) * pow(r2, 2))) * 180 / M_PI;

                    /*
                    printf("+++++++++++++++++++++0\n");
                    int father_edge;
                    for (int t = 0; t < skeletonEdges.size(); t++) {
                        if (std::get<1>(skeletonEdges[t]) == off_begin_vertex) {
                            father_edge = t;
                            break;
                        }
                    }
                    float rp = radii[father_edge];
                    float rc = radii[off_begin_edge];

                    float LRIAM = 90 - acos((pow(rp, 4) + pow(rc, 4) - pow(pow(rp, 3) - pow(rc, 3), 4.0 / 3)) / (2 * pow(rp, 2) * pow(rc, 2))) * 180 / M_PI;

                    printf("+++++++++++++++++++++1\n");
                    */

                    float BA = acos(main_sub_direction.dot(parentDirection) / main_sub_direction.norm() / parentDirection.norm()) * 180 / M_PI;
                    /*
                    Eigen::Vector3f father_edge_direction{
                        skeletonPoints[std::get<1>(skeletonEdges[father_edge])].x - skeletonPoints[std::get<0>(skeletonEdges[father_edge])].x,
                        skeletonPoints[std::get<1>(skeletonEdges[father_edge])].y - skeletonPoints[std::get<0>(skeletonEdges[father_edge])].y,
                        skeletonPoints[std::get<1>(skeletonEdges[father_edge])].z - skeletonPoints[std::get<0>(skeletonEdges[father_edge])].z,
                    };

                    float LIA = 90 - acos(father_edge_direction.dot(main_sub_direction) / father_edge_direction.norm() / main_sub_direction.norm()) * 180 / M_PI;

                    printf("+++++++++++++++++++++2\n");*/

                    pcl::PointXYZ treetop_point = std::get<0>(print_info.back());
                    std::vector<int> layer_info = std::get<1>(print_info.back());;
                    std::vector<float> branch_parameters = std::get<2>(print_info.back());

                    branch_parameters.push_back(AA);
                    branch_parameters.push_back(AAM);
                    branch_parameters.push_back(BA);
                   // branch_parameters.push_back(LRIAM);
                    //branch_parameters.push_back(LIA);

                    print_info[print_info.size() - 1] = std::make_tuple(treetop_point, layer_info, branch_parameters);


                }

            }
            else {
                SeparateBranches(following_edge, skeletonPoints, skeletonEdges, radii, z_ground, level + 1, levels, edges_on_branches, print_info, main_local_direction, std::vector<int>(), new_parent_branch_idx);

                std::vector<int> complete_branch_edges = edges_on_branches.back();

                Eigen::Vector3f first_sub_direction = {
                    skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.front()])].x - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].x,
                    skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.front()])].y - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].y,
                    skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.front()])].z - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].z
                };
                Eigen::Vector3f main_sub_direction = {
                    skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.back()])].x - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].x,
                    skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.back()])].y - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].y,
                    skeletonPoints[std::get<1>(skeletonEdges[complete_branch_edges.back()])].z - skeletonPoints[std::get<0>(skeletonEdges[complete_branch_edges.front()])].z
                };

                float AA = acos(first_sub_direction.dot(main_local_direction) / first_sub_direction.norm() / main_local_direction.norm()) * 180 / M_PI;
                
                int main_following_edge = std::get<0>(combined_edges_and_levels[0]);
                float r1 = radii[main_following_edge];
                float r2 = radii[following_edge];
                float AAM = acos((pow(pow(r1, 3) + pow(r2, 3), 4.0 / 3.0) - pow(r1, 4) - pow(r2, 4)) / (2 * pow(r1, 2) * pow(r2, 2))) * 180 / M_PI;
                /*printf("-----------------------------0\n");
                int off_begin_edge = complete_branch_edges.front();
                int off_begin_vertex = std::get<0>(skeletonEdges[off_begin_edge]);
                int father_edge;
                for (int t = 0; t < skeletonEdges.size(); t++) {
                    if (std::get<1>(skeletonEdges[t]) == off_begin_vertex) {
                        father_edge = t;
                        break;
                    }
                }

                float rp = radii[father_edge];
                float rc = radii[following_edge];

                float LRIAM = 90 - acos((pow(rp, 4) + pow(rc, 4) - pow(pow(rp, 3) - pow(rc, 3), 4.0 / 3)) / (2 * pow(rp, 2) * pow(rc, 2))) * 180 / M_PI;
                printf("-----------------------------1\n");*/

                float BA = acos(main_sub_direction.dot(main_local_direction) / main_sub_direction.norm() / main_local_direction.norm()) * 180 / M_PI;
                
                /*
                Eigen::Vector3f father_edge_direction{
                        skeletonPoints[std::get<1>(skeletonEdges[father_edge])].x - skeletonPoints[std::get<0>(skeletonEdges[father_edge])].x,
                        skeletonPoints[std::get<1>(skeletonEdges[father_edge])].y - skeletonPoints[std::get<0>(skeletonEdges[father_edge])].y,
                        skeletonPoints[std::get<1>(skeletonEdges[father_edge])].z - skeletonPoints[std::get<0>(skeletonEdges[father_edge])].z,
                };

                float LIA = 90 - acos(father_edge_direction.dot(main_sub_direction) / father_edge_direction.norm() / main_sub_direction.norm()) * 180 / M_PI;

                printf("-----------------------------2\n");*/

                pcl::PointXYZ treetop_point = std::get<0>(print_info.back());
                std::vector<int> layer_info = std::get<1>(print_info.back());;
                std::vector<float> branch_parameters = std::get<2>(print_info.back());

                branch_parameters.push_back(AA);
                branch_parameters.push_back(AAM);
                branch_parameters.push_back(BA);
                //branch_parameters.push_back(LRIAM);
                //branch_parameters.push_back(LIA);

                print_info[print_info.size() - 1] = std::make_tuple(treetop_point, layer_info, branch_parameters);
            }
        }
    }
    }
}


void ASM(pcl::PointCloud<pcl::PointXYZ> skeletonPoints, std::vector<std::tuple<int, int>> skeletonEdges, std::vector<std::vector<int>> edges_on_branches, std::vector<float> radii, std::vector<int> branch_levels, std::vector<vtkSmartPointer<vtkPolyData>>& branch_models, std::vector<std::tuple<UCHAR, UCHAR, UCHAR>> level_color_mapping, int numberOfSides = 50) {
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef Kernel::Point_3 Point;
    typedef CGAL::Polyhedron_3<Kernel> Polyhedron;

    typedef CGAL::Nef_polyhedron_3<Kernel> Nef_polyhedron;

    // Reorder branches by radii then levels 
    std::vector<int> branch_order_idx;
    std::vector<std::tuple<int, float>> for_order_by_radii;
    for (int i = 0; i < edges_on_branches.size(); i++) {
        float radius = radii[edges_on_branches[i].front()]; // Use branch diameter(base)
        for_order_by_radii.push_back(std::make_tuple(i, radius));
    }
    std::sort(for_order_by_radii.begin(), for_order_by_radii.end(), [](auto& e1, auto& e2) {
        return std::get<1>(e1) > std::get<1>(e2); // The larger the radius is, the more prior the element is
        });
    for (int i = 0; i < for_order_by_radii.size(); i++) {
        branch_order_idx.push_back(std::get<0>(for_order_by_radii[i]));
    }
    std::vector<std::tuple<int, int>> for_order_by_levels;
    for (int i = 0; i < edges_on_branches.size(); i++) {
        for_order_by_levels.push_back(std::make_tuple(branch_order_idx[i], branch_levels[branch_order_idx[i]]));
    }
    std::stable_sort(for_order_by_levels.begin(), for_order_by_levels.end(), [](auto& e1, auto& e2) {
        return std::get<1>(e1) < std::get<1>(e2);
        });// stable_sort is to avoid the change of indices among element with the same value
    for (int i = 0; i < branch_order_idx.size(); i++) {
        branch_order_idx[i] = std::get<0>(for_order_by_levels[i]);
    }

    std::vector<Polyhedron> branch_models_cgal(edges_on_branches.size());

    //Criterion: Each branch should not have intersect with any same or lower level branches

    for (int i = 0; i < branch_order_idx.size(); i++) {
        std::vector<int> edges_on_a_branch = edges_on_branches[branch_order_idx[i]];

        Polyhedron branch_model;

        for (int j = 0; j < edges_on_a_branch.size(); j++) {
            pcl::PointXYZ ptBegin;
            Eigen::Vector3f direction_begin;
            float radius_begin;
            if (j == 0) {
                //The first plane on a new branch should link to a parent branch or use the ground.
                if (i == 0) {
                    ptBegin = skeletonPoints[std::get<0>(skeletonEdges[edges_on_a_branch.front()])];
                    direction_begin = {
                        skeletonPoints[std::get<1>(skeletonEdges[edges_on_a_branch.front()])].x - skeletonPoints[std::get<0>(skeletonEdges[edges_on_a_branch.front()])].x,
                        skeletonPoints[std::get<1>(skeletonEdges[edges_on_a_branch.front()])].y - skeletonPoints[std::get<0>(skeletonEdges[edges_on_a_branch.front()])].y,
                        skeletonPoints[std::get<1>(skeletonEdges[edges_on_a_branch.front()])].z - skeletonPoints[std::get<0>(skeletonEdges[edges_on_a_branch.front()])].z
                    };
                    radius_begin = radii[edges_on_a_branch.front()];
                }
                else {
                    int begin_vertex = std::get<0>(skeletonEdges[edges_on_a_branch.front()]);

                    int parent_edge;
                    for (int k = 0; k < skeletonEdges.size(); k++) {
                        if (std::get<1>(skeletonEdges[k]) == begin_vertex) {
                            parent_edge = k;
                            break;
                        }
                    }
                    ptBegin = { (float)(0.5 * (skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].x + skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].x)),
                                (float)(0.5 * (skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].y + skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].y)),
                                (float)(0.5 * (skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].z + skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].z))
                    };
                    direction_begin = {
                        skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].x - skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].x,
                        skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].y - skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].y,
                        skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].z - skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].z
                    };
                    radius_begin = radii[parent_edge];
                }
            }
            else {
                int parent_edge = edges_on_a_branch[j - 1];
                ptBegin = { (float)(0.5 * (skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].x + skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].x)),
                            (float)(0.5 * (skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].y + skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].y)),
                            (float)(0.5 * (skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].z + skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].z))
                };
                direction_begin = {
                    skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].x - skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].x,
                    skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].y - skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].y,
                    skeletonPoints[std::get<1>(skeletonEdges[parent_edge])].z - skeletonPoints[std::get<0>(skeletonEdges[parent_edge])].z
                };
                radius_begin = radii[parent_edge];
            }


            std::vector<Point> points;
            Eigen::Vector3f z_positive{ 0.0,0.0,1.0 };
            Eigen::Matrix3f rot_mat1;

            rot_mat1 = Eigen::Quaternionf::FromTwoVectors( z_positive, direction_begin.normalized()).toRotationMatrix();
            for (int k = 0; k < numberOfSides;k++) {
                Eigen::Vector3f point{
                    (float)(radius_begin * cos((float)k / numberOfSides * 2.0 * M_PI)),
                    (float)(radius_begin * sin((float)k / numberOfSides * 2.0 * M_PI)),
                    0.0
                };
                Eigen::Vector3f new_point = rot_mat1 * point;

                points.push_back(Point(
                    new_point.x() + ptBegin.x,
                    new_point.y() + ptBegin.y,
                    new_point.z() + ptBegin.z
                ));
            }

            int current_edge = edges_on_a_branch[j];
            pcl::PointXYZ ptEnd = { (float)(0.5 * (skeletonPoints[std::get<0>(skeletonEdges[current_edge])].x + skeletonPoints[std::get<1>(skeletonEdges[current_edge])].x)),
                        (float)(0.5 * (skeletonPoints[std::get<0>(skeletonEdges[current_edge])].y + skeletonPoints[std::get<1>(skeletonEdges[current_edge])].y)),
                        (float)(0.5 * (skeletonPoints[std::get<0>(skeletonEdges[current_edge])].z + skeletonPoints[std::get<1>(skeletonEdges[current_edge])].z))
            };
            Eigen::Vector3f direction_end = {
                skeletonPoints[std::get<1>(skeletonEdges[current_edge])].x - skeletonPoints[std::get<0>(skeletonEdges[current_edge])].x,
                skeletonPoints[std::get<1>(skeletonEdges[current_edge])].y - skeletonPoints[std::get<0>(skeletonEdges[current_edge])].y,
                skeletonPoints[std::get<1>(skeletonEdges[current_edge])].z - skeletonPoints[std::get<0>(skeletonEdges[current_edge])].z
            };
            float radius_end = radii[current_edge];

            std::vector<Point> points2; // Only work on treetop
            Eigen::Matrix3f rot_mat2;
            rot_mat2 = Eigen::Quaternionf::FromTwoVectors(z_positive, direction_end.normalized()).toRotationMatrix();
            for (int k = 0; k < numberOfSides; k++) {
                Eigen::Vector3f point{
                    (float)(radius_end * cos((float)k / numberOfSides * 2 * M_PI)),
                    (float)(radius_end * sin((float)k / numberOfSides * 2 * M_PI)),
                    0.0
                };
                Eigen::Vector3f new_point = rot_mat2 * point;

                points.push_back(Point(
                    new_point.x() + ptEnd.x,
                    new_point.y() + ptEnd.y,
                    new_point.z() + ptEnd.z
                ));
                if (j == edges_on_a_branch.size() - 1) {
                    points2.push_back(Point(
                        new_point.x() + ptEnd.x,
                        new_point.y() + ptEnd.y,
                        new_point.z() + ptEnd.z
                    ));
                }
            }
            // Convex hull
            Polyhedron convex_hull;
            CGAL::convex_hull_3(points.begin(), points.end(), convex_hull);


            CGAL::Polygon_mesh_processing::corefine_and_compute_union(convex_hull, branch_model, branch_model);

            if (j == edges_on_a_branch.size() - 1) {
                // Treetop use a conic
                pcl::PointXYZ treetop_point = skeletonPoints[std::get<1>(skeletonEdges[edges_on_a_branch.back()])];
                points2.push_back(Point(treetop_point.x, treetop_point.y, treetop_point.z));

                Polyhedron convex_hull2;
                CGAL::convex_hull_3(points2.begin(), points2.end(), convex_hull2);
                CGAL::Polygon_mesh_processing::corefine_and_compute_union(convex_hull2, branch_model, branch_model);
            }
        }

        //The union of CGAL has problems. 
        //CGAL::Polygon_mesh_processing::corefine_and_compute_union(treeModel, branch_model, treeModel);

        /*
        for (int j = 0; j < i; j++) {
            CGAL::Polygon_mesh_processing::corefine_and_compute_difference(branch_model, branch_models_cgal[j], branch_model);
        }*/

        branch_models_cgal[i] = branch_model;
    }

    //double trunk_volume = CGAL::Polygon_mesh_processing::volume(branch_models_cgal[0]);
    //Convert to real value
    //double tree_volume = CGAL::Polygon_mesh_processing::volume(treeModel);

    branch_models.clear();
    branch_models.resize(branch_models_cgal.size());

    for (int i = 0; i < branch_models_cgal.size(); i++) {
        Polyhedron branch_model = branch_models_cgal[i];
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

        for (Polyhedron::Vertex_iterator it = branch_model.vertices_begin(); it != branch_model.vertices_end(); ++it) {
            points->InsertNextPoint(it->point().x(), it->point().y(), it->point().z());
        }
        vtkSmartPointer<vtkCellArray> cellArray = vtkSmartPointer<vtkCellArray>::New();

        for (Polyhedron::Facet_iterator it = branch_model.facets_begin(); it != branch_model.facets_end(); ++it) {
            Polyhedron::Halfedge_around_facet_circulator facet = it->facet_begin();

            vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();

            int cnt = 0;
            do {
                int point_idx = std::distance(branch_model.vertices_begin(), facet->vertex());
                triangle->GetPointIds()->SetId(cnt, point_idx);
                cnt++;
            } while (++facet != it->facet_begin());

            cellArray->InsertNextCell(triangle);
        }

        vtkSmartPointer<vtkPolyData> polyData_branch = vtkSmartPointer<vtkPolyData>::New();
        polyData_branch->SetPoints(points);
        polyData_branch->SetPolys(cellArray);

        
        vtkSmartPointer<vtkUnsignedCharArray> ptColor =
            vtkSmartPointer<vtkUnsignedCharArray>::New();
        ptColor->SetNumberOfComponents(3);
        unsigned char color[3] = {
                std::get<0>(level_color_mapping[branch_levels[branch_order_idx[i]]]),
                std::get<1>(level_color_mapping[branch_levels[branch_order_idx[i]]]),
                std::get<2>(level_color_mapping[branch_levels[branch_order_idx[i]]])
        };

        for (int j = 0; j < polyData_branch->GetNumberOfPoints(); j++) {
            ptColor->InsertNextTypedTuple(color);
        }
        polyData_branch->GetPointData()->SetScalars(ptColor);

        branch_models[branch_order_idx[i]] = polyData_branch;
    }

    //printf("Volume trunk %.4lf", trunk_volume);
}


int ViewBranchModel(std::vector<vtkSmartPointer<vtkPolyData>> branch_models, std::vector<pcl::PointXYZ> treetop_points, bool bEnableNo)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);

    viewer->setCameraPosition(pos_x, pos_y, pos_z, view_x, view_y, view_z, up_x, up_y, up_z);

    for (int i = 0; i < treetop_points.size(); i++) {
        std::ostringstream ostr;
        ostr <<  i;
        viewer->addModelFromPolyData(branch_models[i],ostr.str());
        if (bEnableNo == true) {
            std::ostringstream ostr2;
            ostr2 << "t"<<i;
            viewer->addText3D(ostr.str(), treetop_points[i], 0.125, 0, 0, 0, ostr2.str());
        }
    }
    

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));

        SaveCurrentCameraParameters(viewer);

        std::ostringstream ostr;
        ostr << "Pos(" << pos_x << "," << pos_y << "," << pos_z << ") View(" << view_x << "," << view_y << "," << view_z << ") Up(" << up_x << "," << up_y << "," << up_z << ")";
        viewer->setWindowName(ostr.str());
    }

    return 0;
}

void CombineAndGetBranchTotalAttr(std::vector<vtkSmartPointer<vtkPolyData>>& branch_models,double& trunk_volume, double& total_volume, double& total_area, vtkSmartPointer<vtkPolyData>& treeModel) {
    vtkSmartPointer<vtkTriangleFilter> triFilter1 = vtkSmartPointer<vtkTriangleFilter>::New();
    triFilter1->SetInputData(branch_models[0]);
    triFilter1->Update();

    vtkSmartPointer<vtkMassProperties> massProp1 = vtkSmartPointer<vtkMassProperties>::New();
    massProp1->SetInputData(triFilter1->GetOutput());
    trunk_volume = massProp1->GetVolume();


    auto appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
    for (int i = 0; i < branch_models.size(); i++) {
        appendFilter->AddInputData(branch_models[i]);
    }
    auto cleanFilter =
        vtkSmartPointer<vtkCleanPolyData>::New();
    cleanFilter->SetInputConnection(appendFilter->GetOutputPort());
    cleanFilter->Update();
    
    vtkSmartPointer<vtkTriangleFilter> triFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    triFilter->SetInputData(cleanFilter->GetOutput());
    triFilter->Update();

    vtkSmartPointer<vtkMassProperties> massProp = vtkSmartPointer<vtkMassProperties>::New();
    massProp->SetInputData(triFilter->GetOutput());

    total_area = massProp->GetSurfaceArea();
    total_volume = massProp->GetVolume();

    treeModel = triFilter->GetOutput();
}

int WritePolyDataToOBJ(vtkSmartPointer<vtkPolyData> polyData, std::string vtkFilePath) {
    vtkSmartPointer<vtkOBJWriter> objWriter = vtkSmartPointer<vtkOBJWriter>::New();
    objWriter->SetFileName(vtkFilePath.c_str());
    objWriter->SetInputData(polyData);
    int status = objWriter->Write();
    return status;
}

void GetInitialCylinderModel(pcl::PointCloud<pcl::PointXYZ> skeletonPoints, std::vector<std::tuple<int,int>> skeletonEdges, std::vector<float> radius, vtkSmartPointer<vtkPolyData>& polyData, int edgeNum = 50)
{
    auto appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
    for (int i = 0; i < skeletonEdges.size(); i++) {
        pcl::PointXYZ ptBegin = skeletonPoints[std::get<0>(skeletonEdges[i])];
        pcl::PointXYZ ptEnd = skeletonPoints[std::get<1>(skeletonEdges[i])];

        vtkSmartPointer<vtkLineSource> lineSource =
            vtkSmartPointer<vtkLineSource>::New();
        lineSource->SetPoint1(ptBegin.x, ptBegin.y, ptBegin.z);
        lineSource->SetPoint2(ptEnd.x, ptEnd.y, ptEnd.z);
        vtkSmartPointer<vtkTubeFilter> tubeFilter = vtkSmartPointer<vtkTubeFilter>::New();
        tubeFilter->SetInputConnection(lineSource->GetOutputPort());
        tubeFilter->SetRadius(radius[i]);
        tubeFilter->SetNumberOfSides(edgeNum);
        tubeFilter->CappingOn();
        tubeFilter->Update();

        appendFilter->AddInputData(tubeFilter->GetOutput());
    }

    auto cleanFilter =
        vtkSmartPointer<vtkCleanPolyData>::New();
    cleanFilter->SetInputConnection(appendFilter->GetOutputPort());
    cleanFilter->Update();

    vtkSmartPointer<vtkTriangleFilter> triFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    triFilter->SetInputData(cleanFilter->GetOutput());
    triFilter->Update();

    polyData = triFilter->GetOutput();
}

void PrintUsage(const char* progName) {
    printf("Usage: %s -p PCDPOINTCLOUD -o OBJSKELETON [-l] [-a] [-h LAYERHEIGHT] [-m] [-c CAMERAFILE]\n\
Options:\n\
-p PCDPOINTCLOUD\t\tInput tree BRANCH point cloud file(.pcd);\n\
-o OBJSKELETON\t\t\tInput tree SKELETON (.obj);\n\
-l \t\t\t\tGet initial radii by fitting circles with Least squares (default by averaging the distances without [-l])\n\
-a \t\t\t\tAutomatically specify the height as 1.3 m if a tree is more than 5 m and tree height * 1.3 m / 5 m if a tree is no more than 5 m (NOT RECOMMEND. You can manually specify the height at a certain moment)\n\
-h LAYERHEIGHT\t\t\tLayer height for computing the initial radii.\n\
-m \t\t\t\tHide the number of the branch in visualization.\n\
-c CAMERAFILE\t\t\tInput camera file(.txt);\n\
Output model will save to *_model.obj in the same directory of input tree BRANCH point cloud. \n", progName);
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

int main(int argc, char** argv) {
    std::cout << "Near Real Tree Model Constructer by J.YANG (Version 1.0 release)" << std::endl << std::endl;


    std::string strPCDPath;
    std::string strOBJPath;
    if (pcl::console::parse(argc, argv, "-p", strPCDPath) < 0) {
        PrintUsage(argv[0]);
        return -1;
    }
    if (pcl::console::parse(argc, argv, "-o", strOBJPath) < 0) {
        PrintUsage(argv[0]);
        return -1;
    }
    

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ> skeletonPoints;
    std::vector<std::tuple<int, int>> skeletonEdges;

    OpenPCD(strPCDPath, *cloud);
    ReadSkeletonFromOBJ(strOBJPath, skeletonPoints, skeletonEdges);

    
    bool bLS = false;
    if (pcl::console::find_argument(argc, argv, "-l") >= 0) {
        bLS = true;
    }

    bool bUseDefaultHeight = false;
    if (pcl::console::find_argument(argc, argv, "-a") >= 0) {
        bUseDefaultHeight = true;
    }

    bool bOnlyDisplayModel = false;
    if (pcl::console::find_argument(argc, argv, "-m") >= 0) {
        bOnlyDisplayModel = true;
    }
    std::string strCameraFilePath;
    if (pcl::console::parse(argc, argv, "-c", strCameraFilePath) >= 0) {
        if (OpenCameraFile(strCameraFilePath) == -1) {
            //Invalid camera parameters' file, create a new file
            PCL_WARN("[WARNING]Invalid camera parameters' file, will use default parameters.\n");
        }
    }

    PCL_INFO("[INFO] Initializing...\n");
    float fMinInterval = ComputeMinInterval(cloud);
    float search_radius = fMinInterval + 0.000001;

    float temp_search_radius = -1;
    if (pcl::console::parse(argc, argv, "-h", temp_search_radius) >= 0 ) {
        if (temp_search_radius < search_radius) {
            PCL_WARN("[WARNING] Too small layer height. Use default value.\n");
        }
        else {
            search_radius = temp_search_radius;
        }
    }
    
    
    //srand((unsigned)time(NULL)); //NO USE TO KEEP CONSISTENCE OF LEVEL COLOR

    std::vector<int> indegree(skeletonPoints.size(), 0);
    std::vector<int> outdegree(skeletonPoints.size(), 0);
    for (int s = 0; s < skeletonEdges.size(); s++) {
        int v1 = std::get<0>(skeletonEdges[s]);
        int v2 = std::get<1>(skeletonEdges[s]);
        outdegree[v1]++;
        indegree[v2]++;
    }
    std::vector<int> vertices_with_0indegree;
    for (int i = 0; i < indegree.size(); i++) {
        if (indegree[i] == 0) {
            vertices_with_0indegree.push_back(i);
        }
    }
    if (vertices_with_0indegree.size() != 1) {
        return -1;
    }

    //Find a stable height to get a benchmark radius
    float z_min = INT_MAX;
    float z_max = INT_MIN;
    for (int i = 0; i < cloud->size(); i++) {
        z_min = fmin(cloud->points[i].z, z_min);
        z_max = fmax(cloud->points[i].z, z_max);
    }

    float base_height = z_min + fmin(1.3, (z_max - z_min) * 1.3 / 5);

    std::vector<int> stable_edges; // Some trees have many offshoots.
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (skeletonPoints[std::get<0>(skeletonEdges[i])].z <= base_height && skeletonPoints[std::get<1>(skeletonEdges[i])].z > base_height) {
            stable_edges.push_back(i);
        }
    }

    // Only use for separating branches of each levels
    PCL_INFO("[INFO] Estimating radii...\n");

    std::vector<float> radii(skeletonEdges.size());
    CalculateRadius(cloud, skeletonPoints, skeletonEdges, radii, search_radius, bLS);

    std::string strInputFileDir, strInputFilename;
    SubstrFromPath(strPCDPath, strInputFileDir, strInputFilename);

    // Specifying a particular height
    if (bUseDefaultHeight == false) {
        vtkSmartPointer<vtkPolyData> polyData;
        GetInitialCylinderModel(skeletonPoints, skeletonEdges, radii, polyData);

        std::string strOutputObj = strInputFileDir + strInputFilename.substr(0, strInputFilename.rfind(".")) + "_initmdl.obj";
        WritePolyDataToOBJ(polyData, strOutputObj);

        PCL_WARN("[WARNING] Default height is not specified. Please enter a height and press [ENTER] to continue.\n");
        PCL_WARN("[WARNING] The input height is the height relative to the ground, rather than the z-axis value of a point.\n");
        PCL_WARN("[WARNING] You can use any third party software to open the point cloud and the produced initial model to find a suitable height. The produced initial skeleton is %s.\n", strOutputObj.c_str());
        float height;

        while (true) {
            printf(">");
            char height_string[16];
            scanf("%s", &height_string);
            height = atoi(height_string);
            if (height <= 0 || height >= z_max - z_min) {
                PCL_ERROR("[ERROR] Invalid input. Height must be in range [%f,%f).\n", 0.000001, z_max-z_min);
                continue;
            }
            break;
        }
        base_height = height + z_min;
    }



    PCL_INFO("[INFO] Optimizing radii...\n");
    
    for (int i = 0; i < stable_edges.size(); i++) {
        // Beyond base height
        RecalculateRadius(stable_edges[i], skeletonPoints, skeletonEdges, radii);
        // Below base height
        // Don't worry the same part. In the final turn(or branch), the previous radii have fixed and subsequently the radius of a parent branch at a bifurcation can be correctly calculated.
        ReverselyDeriveRadius(stable_edges[i], skeletonPoints, skeletonEdges, radii);
    }

    // Trunk extraction
    PCL_INFO("[INFO] Recognizing branch levels...\n");
    std::vector<int> levels(skeletonEdges.size(), -1);
    BranchLevelRecognize(vertices_with_0indegree[0], skeletonEdges, radii, levels);
    int min_level = *std::min_element(levels.begin(), levels.end());
    int cnt_level = *std::max_element(levels.begin(), levels.end()) - min_level + 1;

    
    std::vector<std::tuple<UCHAR,UCHAR,UCHAR>> level_color_mapping(cnt_level);
    for (int i = 0; i < cnt_level; i++) {
        while (true) {
            //Use 8-bit color to increase color discrimination
            int r = rand() % 8;
            int g = rand() % 8;
            int b = rand() % 8;

            if ((r == 0 && g == 0 && b == 0) || (r == 7 && g == 7 && b == 7)) {
                //Disable white and black color
                continue;
            }

            if (r<=3&& g<=3 && b<=3) {
                //Too dark
                continue;
            }

            if (i != 0) {
                int prev_r = std::get<0>(level_color_mapping[i - 1]);
                int prev_g = std::get<1>(level_color_mapping[i - 1]);
                int prev_b = std::get<2>(level_color_mapping[i - 1]);
                if (abs(prev_r + prev_g + prev_b - r - g - b) < 6) {
                    //In order to increase color discrimination of two neighbor branch levels
                    continue;
                }
            }
            level_color_mapping[i] = std::make_tuple( (UCHAR)(int)((float)r/7 *255), (UCHAR)(int)((float)g / 7 * 255), (UCHAR)(int)((float)b / 7 * 255));
            break;
        }
    }

    //PCLBranchModelingByCylinder(skeletonPoints, skeletonEdges, radii, colors);

    // Construct the radius of the stump (because there is a gap between each skeleton and the ground )
    float gap_between_skeleton_and_ground = skeletonPoints[vertices_with_0indegree[0]].z - z_min;
    pcl::PointXYZ stump_center{
        skeletonPoints[vertices_with_0indegree[0]].x,
        skeletonPoints[vertices_with_0indegree[0]].y,
        z_min
    };

    float r_stump;
    int extend_level_edge;

    std::vector<int> indegree_0_outedges;
    for (int i = 0; i < skeletonEdges.size(); i++) {
        if (std::get<0>(skeletonEdges[i]) == vertices_with_0indegree[0]) {
            indegree_0_outedges.push_back(i);
        }
    }
    
    if (indegree_0_outedges.size() == 1) {
        int son_vertex = std::get<1>(skeletonEdges[indegree_0_outedges[0]]);
        std::vector<int> treetop_vertices_from_son_vertex;
        GetTreetopFromAVertex(son_vertex, skeletonEdges, treetop_vertices_from_son_vertex);

        float lc = 0;
        std::set<int> all_edges_from_son;
        for (auto each_treetop_vertex : treetop_vertices_from_son_vertex) {
            std::vector<int> edges;
            EdgesBetweenTwoVertices(son_vertex, each_treetop_vertex, skeletonPoints, skeletonEdges, edges);
            all_edges_from_son.insert(edges.begin(), edges.end());
        }
        for (auto each_edge : all_edges_from_son) {
            lc += GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[each_edge])], skeletonPoints[std::get<1>(skeletonEdges[each_edge])]);
        }

        lc += 0.5 * GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[indegree_0_outedges[0]])], skeletonPoints[std::get<1>(skeletonEdges[indegree_0_outedges[0]])]);

        float lp = lc + 0.5 * GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[indegree_0_outedges[0]])], skeletonPoints[std::get<1>(skeletonEdges[indegree_0_outedges[0]])]) + gap_between_skeleton_and_ground;

        float rc = radii[indegree_0_outedges[0]];
        float rp = rc / pow(lc / lp, 3.0 / 2);
        r_stump = rp;

        extend_level_edge = indegree_0_outedges[0];
    }
    else {
        float rp = 0;
        int min_level = INT_MAX;
        for (int i = 0; i < indegree_0_outedges.size();i++) {
            float rci = radii[indegree_0_outedges[i]];
            rp += pow(rci, 2.49);
            if (levels[indegree_0_outedges[i]] < min_level) {
                min_level = levels[indegree_0_outedges[i]];
                extend_level_edge = indegree_0_outedges[i];
            }
        }
        rp = pow(rp, 1.0 / 2.49);
        r_stump = rp;
    }

    skeletonPoints.push_back(stump_center);
    skeletonEdges.push_back(std::make_tuple(skeletonPoints.size() - 1, vertices_with_0indegree[0]));
    radii.push_back(r_stump);
    levels.push_back(levels[extend_level_edge]);
    //colors.push_back(colors[extend_level_edge]);

    std::vector<std::vector<int>> branch_edges;
    std::vector<std::tuple<pcl::PointXYZ, std::vector<int>, std::vector<float>>> branch_info;

    PCL_INFO("[INFO] Separating branches...\n");
    SeparateBranches(skeletonEdges.size() - 1, skeletonPoints, skeletonEdges, radii, z_min, levels.back(), levels, branch_edges, branch_info);


    PCL_INFO("[INFO] Converting Generalized Cylinder Model to Arterial Snakes Model...\n");
    std::vector<int> branch_levels;
    for (int i = 0; i < branch_info.size(); i++) {
        std::vector<int> layer_info = std::get<1>(branch_info[i]);
        branch_levels.push_back(layer_info[0]);
    }

    std::vector<pcl::PointXYZ> treetop_points;
    std::vector<vtkSmartPointer<vtkPolyData>> branch_models;
    ASM(skeletonPoints, skeletonEdges, branch_edges, radii, branch_levels, branch_models, level_color_mapping);


    PCL_INFO("[INFO] Combining branch models...\n");
    double trunk_volume;
    double total_volume;
    double total_area;
    vtkSmartPointer<vtkPolyData> treeModel = vtkSmartPointer<vtkPolyData>::New();
    CombineAndGetBranchTotalAttr(branch_models, trunk_volume, total_volume, total_area, treeModel);


    std::string strOutputObj = strInputFileDir + strInputFilename.substr(0, strInputFilename.rfind(".")) + "_model.obj";
    WritePolyDataToOBJ(treeModel, strOutputObj);

    PCL_INFO("[INFO] Tree model has been saved to file %s.\n", strOutputObj.c_str());

    PCL_INFO("[INFO] Computing some parameters...\n");
    // Measuing DBH and Diameter at Stump Height Over Bark (DSHOB)
    // DBH: 1.3 m / 4.5 feet( = 1.3716 m ) / 1.2 m
    // DSHOB: 30 cm / 20 cm / 10 cm / 5cm

    float measuring_height[7] = { 1.3,\
        1.3716,\
        1.2,\
        0.3,\
        0.2,\
        0.1,\
        0.05 };
    std::vector<std::vector<float>> diameters_at_different_heights(7);
    for (int i = 0; i < 7; i++) {
        float check_height = z_min + measuring_height[i];
        std::vector<float> diameters_at_a_height;
        if (check_height >= z_max) {
            diameters_at_a_height.push_back(0);
            diameters_at_different_heights[i] = diameters_at_a_height;
            continue;
        }
        std::vector<int> edges_crossing_a_height;
        for (int j = 0; j < skeletonEdges.size(); j++) {
            if (skeletonPoints[std::get<0>(skeletonEdges[j])].z <= check_height && skeletonPoints[std::get<1>(skeletonEdges[j])].z > check_height) {
                edges_crossing_a_height.push_back(j);
            }
        }
        for (int j = 0; j < edges_crossing_a_height.size(); j++) {
            int vertex_Begin = std::get<0>(skeletonEdges[edges_crossing_a_height[j]]);
            int vertex_End = std::get<1>(skeletonEdges[edges_crossing_a_height[j]]);

            float center_z = (skeletonPoints[vertex_Begin].z + skeletonPoints[vertex_End].z) / 2;
            if (center_z == check_height) {
                diameters_at_a_height.push_back(2 * radii[edges_crossing_a_height[j]]);
                continue;
            }

            std::vector<int> treetop_vertices_from_son_vertex;
            GetTreetopFromAVertex(vertex_End, skeletonEdges, treetop_vertices_from_son_vertex);

            float l = 0;
            std::set<int> all_edges_from_son;
            for (auto each_treetop_vertex : treetop_vertices_from_son_vertex) {
                std::vector<int> edges;
                EdgesBetweenTwoVertices(vertex_End, each_treetop_vertex, skeletonPoints, skeletonEdges, edges);
                all_edges_from_son.insert(edges.begin(), edges.end());
            }
            for (auto each_edge : all_edges_from_son) {
                l += GetDistanceBetween2pts(skeletonPoints[std::get<0>(skeletonEdges[each_edge])], skeletonPoints[std::get<1>(skeletonEdges[each_edge])]);
            }

            pcl::PointXYZ ptBegin = skeletonPoints[vertex_Begin];
            pcl::PointXYZ ptEnd = skeletonPoints[vertex_End];
            Eigen::Vector3f slope = {
                ptEnd.x - ptBegin.x,
                ptEnd.y - ptBegin.y,
                ptEnd.z - ptBegin.z
            };

            float t = (check_height - ptBegin.z) / slope.z();
            pcl::PointXYZ ptHeight = {
                ptBegin.x + slope.x() * t,
                ptBegin.y + slope.y() * t,
                check_height
            };

            if (edges_crossing_a_height[j] == radii.size() - 1) { // Only the latest added skeleton edge uses the grond radius 
                float lc = l + GetDistanceBetween2pts(ptHeight, skeletonPoints[vertex_End]);
                float lp = l + GetDistanceBetween2pts(skeletonPoints[vertex_Begin], skeletonPoints[vertex_End]);
                float rp = radii[edges_crossing_a_height[j]];
                float rc = rp * pow(lc / lp, 3.0 / 2.0);
                diameters_at_a_height.push_back(2 * rc);
            }
            else {
                if (check_height < center_z) {
                    float lc = l + 0.5 * GetDistanceBetween2pts(skeletonPoints[vertex_Begin], skeletonPoints[vertex_End]);
                    float lp = l + GetDistanceBetween2pts(ptHeight, skeletonPoints[vertex_End]);
                    float rc = radii[edges_crossing_a_height[j]];
                    float rp = rc / pow(lc / lp, 3.0 / 2);
                    diameters_at_a_height.push_back(2 * rp);
                }
                else {
                    float lp = l + 0.5 * GetDistanceBetween2pts(skeletonPoints[vertex_Begin], skeletonPoints[vertex_End]);
                    float lc = l + GetDistanceBetween2pts(ptHeight, skeletonPoints[vertex_End]);
                    float rp = radii[edges_crossing_a_height[j]];
                    float rc = rp * pow(lc / lp, 3.0 / 2.0);
                    diameters_at_a_height.push_back(2 * rc);
                }
            }
        }
        diameters_at_different_heights[i] = diameters_at_a_height;
    }

    ofstream ofs;
    ofs.open("temp_treeinfo.txt", ios::out);
    ofs << "[Abbreviations]\nBranch Length(BL), Branch Chord Length(BCL), Branch Diameter(BD), Branch Height(BH), Branch Arc Height(BAH), Inclination Angle(IA), Azimuth, [Axil Angle - Skeleton(AA), Axil Angle - Murray's law (AAM), Branching Angle(BA)]\n\n";

    for (int i = 0; i < branch_info.size(); i++) {
        pcl::PointXYZ treetop_point = std::get<0>(branch_info[i]);
        std::vector<int> layer_info = std::get<1>(branch_info[i]);
        std::vector<float> branch_parameters = std::get<2>(branch_info[i]);

        treetop_points.push_back(treetop_point);

        ofs << "Branch No."<< i <<": " ;
        if (layer_info[0] == 0) {
            ofs << "Trunk\n" ;
        }
        else {
            ofs << "["<< layer_info[0]<< "]-level branch, parent branch No. "<< layer_info[1] <<"\n";
        }
        ofs << std::fixed << std::setprecision(4)
            << "    BL: " << branch_parameters[0]
            << "    BCL: " << branch_parameters[1]
            << "    BD: " << branch_parameters[2]
            << "    BH: " << branch_parameters[3]
            << "    BAH:" << branch_parameters[4]
            << std::endl
            << "    IA: " << branch_parameters[5]
            << "deg    Azimuth: " << branch_parameters[6] << "deg";
        if (branch_parameters.size() != 7) {
            ofs << std::fixed << std::setprecision(4)
                << "    AA:  " << branch_parameters[7]
                << "deg    AA(M):" << branch_parameters[8]
                << "deg    BA:" << branch_parameters[9]<< "deg";
            /*
            ofs << std::fixed << std::setprecision(4)
                << "    AA:  " << branch_parameters[7]
                << "deg    AA(M):" << branch_parameters[8]
                << "deg    BA:" << branch_parameters[9]
                << "deg    *LRIAM:"<< branch_parameters[10]
                << "deg    *LIA"<< branch_parameters[11]<<"deg";
            */
        }
        ofs << std::endl << std::endl;
    }

    ofs << "Tree information:\n";

    for (int i = 0; i < 7; i++) {
        ofs << "Diameter(s) at "<< measuring_height[i] <<" m: ";
        std::vector<float> diameters_at_a_height = diameters_at_different_heights[i];
        for (int j = 0; j < diameters_at_a_height.size(); j++) {
            ofs << std::fixed << std::setprecision(4) << diameters_at_a_height[j] << " ";
        }
        ofs << std::endl;
    }
    //ofs << std::fixed << std::setprecision(4) << "Diameter at 0 m: " << 2 * radii[radii.size() - 1] << std::endl; // BD of trunk instead.
    ofs << std::fixed << std::setprecision(4) << "Height: " << z_max - z_min << std::endl;

    ofs << std::fixed << std::setprecision(4) << "Trunk volume " << trunk_volume <<", total volume "<< total_volume <<", total_area: "<< total_area<< "\n";

    ofs.close();
    system("start temp_treeinfo.txt & exit");

    // Branch models for visualization should be used as the combined model exists color transition.
    // Limited to VTK, the output .obj should use the combined model.
    ViewBranchModel(branch_models, treetop_points, !bOnlyDisplayModel);
    
    remove("temp_treeinfo.txt");


    return 0;
}