#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/header.hpp>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <cmath>
#include <algorithm>

/**
 * @brief Multi-threaded LIDAR obstacle detector node
 * 
 * Subscribes to /lidar/scan, processes point cloud data in parallel,
 * and publishes detected obstacles to /lidar/obstacles
 */
class LidarObstacleDetector : public rclcpp::Node
{
public:
    LidarObstacleDetector()
        : Node("lidar_obstacle_detector")
        , processing_(false)
        , num_threads_(std::thread::hardware_concurrency())
    {
        // Initialize parameters
        this->declare_parameter<double>("min_range", 0.1);
        this->declare_parameter<double>("max_range", 10.0);
        this->declare_parameter<double>("obstacle_threshold", 0.5);
        this->declare_parameter<int>("num_processing_threads", num_threads_);
        
        min_range_ = this->get_parameter("min_range").as_double();
        max_range_ = this->get_parameter("max_range").as_double();
        obstacle_threshold_ = this->get_parameter("obstacle_threshold").as_double();
        num_threads_ = this->get_parameter("num_processing_threads").as_int();
        
        if (num_threads_ < 1) {
            num_threads_ = 1;
        }
        
        RCLCPP_INFO(this->get_logger(), "LIDAR Obstacle Detector Node started");
        RCLCPP_INFO(this->get_logger(), "Using %d processing threads", num_threads_);
        RCLCPP_INFO(this->get_logger(), "Min range: %.2f m, Max range: %.2f m", min_range_, max_range_);
        RCLCPP_INFO(this->get_logger(), "Obstacle threshold: %.2f m", obstacle_threshold_);
        
        // Create subscriber for LIDAR scans
        subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/lidar/scan",
            10,
            std::bind(&LidarObstacleDetector::scan_callback, this, std::placeholders::_1)
        );
        
        // Create publisher for detected obstacles
        publisher_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "/lidar/obstacles",
            10
        );
        
        // Initialize processing threads
        processing_ = true;
        for (int i = 0; i < num_threads_; ++i) {
            processing_threads_.emplace_back(
                std::thread(&LidarObstacleDetector::processing_worker, this, i)
            );
        }
        
        RCLCPP_INFO(this->get_logger(), "Subscribing to: /lidar/scan");
        RCLCPP_INFO(this->get_logger(), "Publishing to: /lidar/obstacles");
    }
    
    ~LidarObstacleDetector()
    {
        // Stop processing threads
        processing_ = false;
        cv_.notify_all();
        
        for (auto& thread : processing_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

private:
    /**
     * @brief Structure to hold scan data for processing
     */
    struct ScanData
    {
        sensor_msgs::msg::LaserScan scan;
        rclcpp::Time timestamp;
    };
    
    /**
     * @brief Callback function for incoming LIDAR scans
     */
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        // Add scan to processing queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            ScanData scan_data;
            scan_data.scan = *msg;
            scan_data.timestamp = this->now();
            scan_queue_.push(scan_data);
        }
        
        // Notify processing threads
        cv_.notify_one();
        
        RCLCPP_DEBUG(this->get_logger(), "Received LIDAR scan with %zu ranges", msg->ranges.size());
    }
    
    /**
     * @brief Worker thread function for processing scans
     */
    void processing_worker(int thread_id)
    {
        RCLCPP_DEBUG(this->get_logger(), "Processing thread %d started", thread_id);
        
        while (processing_ || !scan_queue_.empty()) {
            ScanData scan_data;
            bool has_data = false;
            
            // Get scan from queue
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this] { return !scan_queue_.empty() || !processing_; });
                
                if (!scan_queue_.empty()) {
                    scan_data = scan_queue_.front();
                    scan_queue_.pop();
                    has_data = true;
                }
            }
            
            if (has_data) {
                // Process the scan
                std::vector<geometry_msgs::msg::PointStamped> obstacles = 
                    process_scan(scan_data.scan, scan_data.timestamp);
                
                // Publish detected obstacles
                for (const auto& obstacle : obstacles) {
                    publisher_->publish(obstacle);
                }
                
                if (!obstacles.empty()) {
                    RCLCPP_DEBUG(this->get_logger(), 
                                "Thread %d: Detected %zu obstacles", 
                                thread_id, obstacles.size());
                }
            }
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Processing thread %d stopped", thread_id);
    }
    
    /**
     * @brief Process LIDAR scan and detect obstacles
     * 
     * @param scan The LIDAR scan message
     * @param timestamp Timestamp of the scan
     * @return Vector of detected obstacle positions
     */
    std::vector<geometry_msgs::msg::PointStamped> process_scan(
        const sensor_msgs::msg::LaserScan& scan,
        const rclcpp::Time& timestamp)
    {
        std::vector<geometry_msgs::msg::PointStamped> obstacles;
        
        // Validate scan data
        if (scan.ranges.empty() || scan.angle_increment == 0.0) {
            return obstacles;
        }
        
        // Process scan data in chunks (for multi-threading demonstration)
        size_t chunk_size = scan.ranges.size() / num_threads_;
        if (chunk_size == 0) {
            chunk_size = scan.ranges.size();
        }
        
        // Detect obstacles by finding points closer than threshold
        for (size_t i = 0; i < scan.ranges.size(); ++i) {
            float range = scan.ranges[i];
            
            // Filter invalid ranges
            if (std::isnan(range) || std::isinf(range) || 
                range < min_range_ || range > max_range_) {
                continue;
            }
            
            // Check if this point represents an obstacle (closer than threshold)
            if (range < obstacle_threshold_) {
                // Calculate obstacle position in polar coordinates
                float angle = scan.angle_min + i * scan.angle_increment;
                
                // Convert to Cartesian coordinates
                float x = range * std::cos(angle);
                float y = range * std::sin(angle);
                float z = 0.0;  // 2D LIDAR, z = 0
                
                // Create obstacle message
                geometry_msgs::msg::PointStamped obstacle;
                obstacle.header.stamp = timestamp;
                obstacle.header.frame_id = scan.header.frame_id;
                obstacle.point.x = x;
                obstacle.point.y = y;
                obstacle.point.z = z;
                
                obstacles.push_back(obstacle);
            }
        }
        
        // Additional processing: cluster nearby obstacles (placeholder logic)
        obstacles = cluster_obstacles(obstacles);
        
        return obstacles;
    }
    
    /**
     * @brief Cluster nearby obstacles together (placeholder implementation)
     * 
     * @param obstacles Vector of obstacle points
     * @return Clustered obstacles
     */
    std::vector<geometry_msgs::msg::PointStamped> cluster_obstacles(
        const std::vector<geometry_msgs::msg::PointStamped>& obstacles)
    {
        if (obstacles.empty()) {
            return obstacles;
        }
        
        // Simple clustering: group obstacles within 0.2m of each other
        const double cluster_distance = 0.2;
        std::vector<geometry_msgs::msg::PointStamped> clustered;
        std::vector<bool> processed(obstacles.size(), false);
        
        for (size_t i = 0; i < obstacles.size(); ++i) {
            if (processed[i]) {
                continue;
            }
            
            // Find cluster center
            double sum_x = obstacles[i].point.x;
            double sum_y = obstacles[i].point.y;
            int count = 1;
            processed[i] = true;
            
            for (size_t j = i + 1; j < obstacles.size(); ++j) {
                if (processed[j]) {
                    continue;
                }
                
                double dx = obstacles[i].point.x - obstacles[j].point.x;
                double dy = obstacles[i].point.y - obstacles[j].point.y;
                double distance = std::sqrt(dx * dx + dy * dy);
                
                if (distance < cluster_distance) {
                    sum_x += obstacles[j].point.x;
                    sum_y += obstacles[j].point.y;
                    count++;
                    processed[j] = true;
                }
            }
            
            // Create clustered obstacle
            geometry_msgs::msg::PointStamped clustered_obstacle;
            clustered_obstacle.header = obstacles[i].header;
            clustered_obstacle.point.x = sum_x / count;
            clustered_obstacle.point.y = sum_y / count;
            clustered_obstacle.point.z = obstacles[i].point.z;
            
            clustered.push_back(clustered_obstacle);
        }
        
        return clustered;
    }
    
    // ROS2 subscribers and publishers
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr publisher_;
    
    // Threading components
    std::vector<std::thread> processing_threads_;
    std::queue<ScanData> scan_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> processing_;
    int num_threads_;
    
    // Processing parameters
    double min_range_;
    double max_range_;
    double obstacle_threshold_;
};

/**
 * @brief Main function
 */
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<LidarObstacleDetector>();
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}



