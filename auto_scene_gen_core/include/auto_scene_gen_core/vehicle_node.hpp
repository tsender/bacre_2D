#pragma once

#include <string>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/clock.hpp"
#include "rosgraph_msgs/msg/clock.hpp"
#include "builtin_interfaces/msg/time.hpp"

#include "auto_scene_gen_msgs/msg/status_code.hpp"
#include "auto_scene_gen_msgs/msg/vehicle_status.hpp"
#include "auto_scene_gen_msgs/msg/vehicle_node_operating_info.hpp"
#include "auto_scene_gen_msgs/msg/scene_description.hpp"

#include "auto_scene_gen_msgs/srv/register_vehicle_node.hpp"
#include "auto_scene_gen_msgs/srv/notify_ready.hpp"

namespace auto_scene_gen_core
{

struct TimeInstance
{
    bool b_valid = false;
    std::chrono::time_point<std::chrono::steady_clock> time;

    TimeInstance()
    {
        b_valid = false;
    }

    void set_time(std::chrono::time_point<std::chrono::steady_clock> new_time)
    {
        time = new_time;
        b_valid = true;
    }
};

class AutoSceneGenVehicleNode : public rclcpp::Node
{
public:
    AutoSceneGenVehicleNode(const std::string &node_name);

public:
    const double DUR_BEFORE_RESUBMITTING = 30.;
    static const uint8_t LOG_INFO = 0;
    static const uint8_t LOG_WARN = 1;
    static const uint8_t LOG_ERROR = 2;

    /**
    * Log information via ROS and possibly write to a log file.
    * @param log_level Log level (LOG_INFO, LOG_WARN, LOG_ERROR)
    * @param msg The message to log
    * @param b_write_to_file Indicates if we should write to the temporary log file, if it exists (will eventually be copied to the actual save directory)
    */
    void log(uint8_t log_level, std::string msg, bool b_write_to_file = true);

    /**
    * Log information via ROS and possibly write to a log file.
    * @param log_level Log level (LOG_INFO, LOG_WARN, LOG_ERROR)
    * @param msg The message to log
    * @param b_write_to_file Indicates if we should write to the temporary log file, if it exists (will eventually be copied to the actual save directory)
    */
    void log(uint8_t log_level, const char* msg, bool b_write_to_file = true);

    /**
     * Indicates if the vehicle is enabled and if this node is registered with the AutoSceneGen client
     */
    bool vehicle_ok();

    /**
    * Call this function immediately after the node dies (but before the main function terminates)
    */
    void shutdown();
    
protected:
    // ROS parameters
    std::string vehicle_name;
    uint8_t wid; // Worker ID
    std::string asg_client_name;
    std::string asg_client_ip_addr;
    std::string ssh_username;
    bool b_debug_mode;

    bool b_registered_with_asg_client;
    bool b_services_online;
    bool b_vehicle_enabled;
    bool b_ok_to_run;
    bool b_last_vehicle_ok_status;
    bool b_use_ssh;

    uint8_t asg_client_status;
    uint8_t worker_status;
    TimeInstance register_node_send_time;
    TimeInstance notify_ready_send_time;
    bool b_request_rerun;
    std::string reason_for_rerun;
    std::shared_future<auto_scene_gen_msgs::srv::RegisterVehicleNode::Response::SharedPtr> register_node_future_response;
    std::shared_future<auto_scene_gen_msgs::srv::NotifyReady::Response::SharedPtr> notify_ready_future_response;

    int scenario_number;
    int last_scenario_number;
    std::string save_dir;
    bool b_save_minimal;
    auto_scene_gen_msgs::msg::SceneDescription::SharedPtr scene_description;

    std::string temp_save_dir;
    std::string temp_log_file_path;

    std::string worker_topic_prefix; // ROS topic prefix for all things AutoSceneGen worker rleated
    std::string vehicle_topic_prefix; // ROS topic prefix for all things AutoSceneGen vehicle node related

    // ROS clocks
    rclcpp::Clock::SharedPtr sim_clock;
    rclcpp::Clock::SharedPtr system_clock;

    // Callback groups
    rclcpp::CallbackGroup::SharedPtr clock_cb_group;

    // ROS subscribers
    rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr clock_sub;
    rclcpp::Subscription<auto_scene_gen_msgs::msg::StatusCode>::SharedPtr asg_client_status_sub;
    rclcpp::Subscription<auto_scene_gen_msgs::msg::StatusCode>::SharedPtr worker_status_sub;
    rclcpp::Subscription<auto_scene_gen_msgs::msg::VehicleStatus>::SharedPtr vehicle_status_sub;
    rclcpp::Subscription<auto_scene_gen_msgs::msg::VehicleNodeOperatingInfo>::SharedPtr vehicle_node_operating_info_sub;
    rclcpp::Subscription<auto_scene_gen_msgs::msg::SceneDescription>::SharedPtr scene_description_sub;

    // ROS clients
    rclcpp::Client<auto_scene_gen_msgs::srv::RegisterVehicleNode>::SharedPtr register_node_cli;
    rclcpp::Client<auto_scene_gen_msgs::srv::RegisterVehicleNode>::SharedPtr unregister_node_cli;
    rclcpp::Client<auto_scene_gen_msgs::srv::NotifyReady>::SharedPtr notify_ready_cli;

    // ROS timers
    rclcpp::TimerBase::SharedPtr register_node_timer;

    // Vehicle enabled/disabled timestamps
    builtin_interfaces::msg::Time vehicle_enabled_timestamp;
    builtin_interfaces::msg::Time vehicle_disabled_timestamp;

    /**
    * Save a file on the AutoSceneGenClient. If using ssh, then you must make sure that ssh keys are already setup.
    * @param local_filename The local filename within the temporary save directory
    * @param remote_filename The filename to use when saving on the AutoSceneGenClient
    */
    void save_file_on_remote_asg_client(const char* local_filename, const char* remote_filename = "");

    /**
     * Blocking function that waits for all required services to come online.
     */
    void wait_for_services();
    
    /**
     * Use this function to save any data to disk from the most recent run. This function should be overriden by all child classes. It will automatically get called when the vehicle is reset.
     */
    virtual void save_node_data() {}
    
    /**
     * Use this function to reset any internal variables for the vehicle node. 
     * This function should be overriden by all child classes. 
     * It will automatically get called when the vehicle is reset.
     */
    virtual void reset() {}

    /**
    * Vehicle nodes can implement this function as a means to check if they need to request a rerun 
    * for example, in case they encountered a problem during the simulation that would affect the results.
    * If a rerun is requested, overwrite this->b_request_rerun and this->reason_for_rerun.
    */
    virtual void check_for_rerun_request() {}

    /**
     * Callback for clock topic from AutoSceneGen worker
     */
    void clock_cb(const rosgraph_msgs::msg::Clock::SharedPtr msg);

    /**
     * Callback for AutoSceneGen client status
     */
    void asg_client_status_cb(const auto_scene_gen_msgs::msg::StatusCode::SharedPtr msg);
    
    /**
     * Callback for AutoSceneGen worker status
     */
    void worker_status_cb(const auto_scene_gen_msgs::msg::StatusCode::SharedPtr msg);
    
    /**
     * Callback for AutoSceneGenVehicle status
     */
    void vehicle_status_cb(const auto_scene_gen_msgs::msg::VehicleStatus::SharedPtr msg);
    
    /**
     * Callback for obtaining the vehicle node save directory
     */
    void vehicle_node_operating_info_cb(const auto_scene_gen_msgs::msg::VehicleNodeOperatingInfo::SharedPtr msg);
    
    /**
     * Callback for obtaining the current scene description (to be used for debugging/plotting purposes)
     */
    void scene_description_cb(const auto_scene_gen_msgs::msg::SceneDescription::SharedPtr msg);
    
    /**
     * Registers the vehicle node with the AutoSceneGenClient
     * @param b_resending Indicates if a first attempt to register the node has already occurred
     */
    void register_node_with_asg_client(bool b_resending = false);
    
    /**
     * Sends a NotifyReady request to the AutoSceneGenClient, indicating this node is ready for the next
     * @param b_resending Indicates if a first attempt to send the request has already occurred
     */
    void send_ready_notification(bool b_resending = false);
    
    /**
     * Sends a NotifyReady request to the AutoSceneGenClient, indicating this node is ready for the next
     */
    void register_node_timer_cb();
};

/**
* Spin a given vehicle node
* @param node The node to spin
* @param num_threads Number of threads to use with the multithreaded executor
*/
void spin_vehicle_node(std::shared_ptr<auto_scene_gen_core::AutoSceneGenVehicleNode> node, size_t num_threads=2);

} // namespace auto_scene_gen_core