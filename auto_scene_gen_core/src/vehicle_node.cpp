#include "auto_scene_gen_core/vehicle_node.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include "date/date.h"
#include <pwd.h>
#include <boost/filesystem.hpp>

#include "rcl/time.h"
#include "rclcpp/executors.hpp"
#include "rclcpp/time.hpp"
#include "builtin_interfaces/msg/time.hpp"

using namespace std::chrono_literals;

namespace auto_scene_gen_core
{

AutoSceneGenVehicleNode::AutoSceneGenVehicleNode(const std::string &node_name) : Node(node_name)
{
    // Get parameters
    vehicle_name = declare_parameter<std::string>("vehicle_name", "vehicle");
    wid = declare_parameter<int>("wid", 0);
    asg_client_name = declare_parameter<std::string>("asg_client_name", "asg_client");
    asg_client_ip_addr = declare_parameter<std::string>("asg_client_ip_addr", "");
    ssh_username = declare_parameter<std::string>("ssh_username", "");
    b_debug_mode = declare_parameter<bool>("debug_mode", false);

    b_vehicle_enabled = false;
    b_registered_with_asg_client = false;
    b_services_online = false;
    b_ok_to_run = false;
    b_last_vehicle_ok_status = false;

    b_request_rerun = false;
    reason_for_rerun = "";
    save_dir = "none";

    // Create an empty temporary save directory and log file
    std::string cwd = boost::filesystem::current_path().generic_string();
    temp_save_dir = cwd + "/vehicle_node_temp" + "/worker" + std::to_string(wid) + "/" + std::string(get_name());
    temp_log_file_path = temp_save_dir + "/log.txt";
    if (boost::filesystem::is_directory(boost::filesystem::status(temp_save_dir)))
        boost::filesystem::remove_all(temp_save_dir);
    boost::filesystem::create_directories(temp_save_dir);

    worker_topic_prefix = "/asg_worker" + std::to_string(wid);
    vehicle_topic_prefix = worker_topic_prefix + "/" + vehicle_name;
    std::string asg_client_topic_prefix = "/" + asg_client_name;

    // ROS clocks
    system_clock = std::make_shared<rclcpp::Clock>();
    sim_clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
    auto ret = rcl_enable_ros_time_override(sim_clock->get_clock_handle()); // For some reason we need to force enable ROS time on the sim clock
    std::string active = sim_clock->ros_time_is_active() ? "true" : "false";
    log(LOG_INFO, "Sim clock ROS time enabled: " + active); // Log to screen if the sim clock enabled ROS time
    if (ret != RCL_RET_OK)
        rclcpp::exceptions::throw_from_rcl_error(ret, "Failed to enable ros_time_override_status");

    // CallbackGroups
    clock_cb_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // Clock subscriber
    rclcpp::SubscriptionOptions options;
    options.callback_group = clock_cb_group;
    clock_sub = create_subscription<rosgraph_msgs::msg::Clock>("/clock" + std::to_string(wid), 1, std::bind(&AutoSceneGenVehicleNode::clock_cb, this, std::placeholders::_1));

    // Create ROS subscribers
    asg_client_status_sub = create_subscription<auto_scene_gen_msgs::msg::StatusCode>(asg_client_topic_prefix + "/status", 10, std::bind(&AutoSceneGenVehicleNode::asg_client_status_cb, this, std::placeholders::_1));
    worker_status_sub = create_subscription<auto_scene_gen_msgs::msg::StatusCode>(worker_topic_prefix + "/status", 10, std::bind(&AutoSceneGenVehicleNode::worker_status_cb, this, std::placeholders::_1));
    vehicle_status_sub = create_subscription<auto_scene_gen_msgs::msg::VehicleStatus>(vehicle_topic_prefix + "/status", 10, std::bind(&AutoSceneGenVehicleNode::vehicle_status_cb, this, std::placeholders::_1));
    vehicle_node_operating_info_sub = create_subscription<auto_scene_gen_msgs::msg::VehicleNodeOperatingInfo>(asg_client_topic_prefix + "/vehicle_node_operating_info", 10, std::bind(&AutoSceneGenVehicleNode::vehicle_node_operating_info_cb, this, std::placeholders::_1));
    scene_description_sub = create_subscription<auto_scene_gen_msgs::msg::SceneDescription>(worker_topic_prefix + "/scene_description", 10, std::bind(&AutoSceneGenVehicleNode::scene_description_cb, this, std::placeholders::_1));

    // ROS clients
    register_node_cli = create_client<auto_scene_gen_msgs::srv::RegisterVehicleNode>(asg_client_topic_prefix + "/services/register_vehicle_node");
    unregister_node_cli = create_client<auto_scene_gen_msgs::srv::RegisterVehicleNode>(asg_client_topic_prefix + "/services/unregister_vehicle_node");
    notify_ready_cli = create_client<auto_scene_gen_msgs::srv::NotifyReady>(asg_client_topic_prefix + "/services/notify_ready");

    // ROS timers
    register_node_timer = create_wall_timer(1s, std::bind(&AutoSceneGenVehicleNode::register_node_timer_cb, this));

    // Configure SSH settings
    b_use_ssh = false;
    if (ssh_username.empty()) // Get current username
    {
        uid_t uid = geteuid();
        struct passwd *pw = getpwuid(uid);
        if (pw)
            ssh_username = std::string(pw->pw_name);
    }
    if (!asg_client_ip_addr.empty())
        b_use_ssh = true;

    log(LOG_INFO, "Initialized AutoSceneGenVehicleNode for Worker " + std::to_string(wid));
}

void AutoSceneGenVehicleNode::log(uint8_t log_level, std::string msg, bool b_write_to_file)
{
    log(log_level, msg.c_str(), b_write_to_file);
}

void AutoSceneGenVehicleNode::log(uint8_t log_level, const char* msg, bool b_write_to_file)
{
    // Get date-time formatted as "yyyy-mm-dd HH:MM:SS.ssss", where SS.ssss shows the fractional seconds
    std::stringstream date_time_stringstream;
    {
        using namespace date; // Required to stream the chrono time point into a string
        date_time_stringstream << std::chrono::system_clock::now();
    }
    std::string date_time_string = date_time_stringstream.str();

    if (log_level == LOG_INFO)
        RCLCPP_INFO(get_logger(), "[%s] %s", date_time_string.c_str(), msg);
    else if (log_level == LOG_WARN)
        RCLCPP_WARN(get_logger(), "[%s] %s", date_time_string.c_str(), msg);
    else if (log_level == LOG_ERROR)
        RCLCPP_ERROR(get_logger(), "[%s] %s", date_time_string.c_str(), msg);
    else
        RCLCPP_INFO(get_logger(), "[%s] %s", date_time_string.c_str(), msg);

    if (b_write_to_file && boost::filesystem::is_directory(boost::filesystem::status(temp_save_dir)))
    {
        std::ofstream log_file;
        log_file.open(temp_log_file_path, std::ios::app);

        if (log_level == LOG_INFO)
            log_file << "[" << date_time_string << "] [INFO] " << msg << "\n";
        else if (log_level == LOG_WARN)
            log_file << "[" << date_time_string << "] [WARN] " << msg << "\n";
        else if (log_level == LOG_ERROR)
            log_file << "[" << date_time_string << "] [ERROR] " << msg << "\n";
        else
            log_file << "[" << date_time_string << "] [INFO] " << msg << "\n";
        log_file.close();
    }
}

void AutoSceneGenVehicleNode::save_file_on_remote_asg_client(const char* local_filename, const char* remote_filename)
{
    std::string local_temp_filepath = temp_save_dir + "/" + std::string(local_filename);
    if (!boost::filesystem::exists(boost::filesystem::status(local_temp_filepath)))
    {
        log(LOG_WARN, "Cannot save file. Local filepath does not exist: " + local_temp_filepath);
        return;
    }

    std::string client_filepath;
    if (std::strcmp(remote_filename, "") == 0)
        client_filepath = save_dir + "/" + local_filename;
    else
        client_filepath = save_dir + "/" + std::string(remote_filename);

    // Calling 'rsync' within the system(...) command may not be the best way to save files, 
    // but it certainly is simple and allows us to save files of any type.
    std::stringstream cmd;
    if (b_use_ssh)
        cmd << "rsync " << local_temp_filepath << " " << ssh_username << "@" << asg_client_ip_addr << ":" << client_filepath;
    else
        cmd << "rsync " << local_temp_filepath << " " << client_filepath;
    int ret = std::system(cmd.str().c_str());
    log(LOG_INFO, "Copied file (return status " + std::to_string(ret) + ") for file: " + client_filepath);
}

void AutoSceneGenVehicleNode::wait_for_services()
{
    log(LOG_INFO, "Waiting for services...");
    while(!register_node_cli->wait_for_service(1s)) {}
    while(!notify_ready_cli->wait_for_service(1s)) {}
    b_services_online = true;
    log(LOG_INFO, "Client is online.");
}

void AutoSceneGenVehicleNode::clock_cb(const rosgraph_msgs::msg::Clock::SharedPtr msg)
{
    auto ret = rcl_set_ros_time_override(sim_clock->get_clock_handle(), rclcpp::Time(msg->clock.sec, msg->clock.nanosec, RCL_ROS_TIME).nanoseconds());
    if (ret != RCL_RET_OK)
        rclcpp::exceptions::throw_from_rcl_error(ret, "Failed to set ros_time_override_status");
}

void AutoSceneGenVehicleNode::asg_client_status_cb(const auto_scene_gen_msgs::msg::StatusCode::SharedPtr msg)
{
    asg_client_status = msg->status;

    if (asg_client_status == auto_scene_gen_msgs::msg::StatusCode::OFFLINE)
    {
        log(LOG_INFO, "Client went offline.");
        b_registered_with_asg_client = false;
        b_services_online = false;
        b_ok_to_run = false;
    }
}

void AutoSceneGenVehicleNode::worker_status_cb(const auto_scene_gen_msgs::msg::StatusCode::SharedPtr msg)
{
    worker_status = msg->status;
}

void AutoSceneGenVehicleNode::vehicle_status_cb(const auto_scene_gen_msgs::msg::VehicleStatus::SharedPtr msg)
{
    if (b_debug_mode)
        b_vehicle_enabled = msg->enabled;
    else
        b_vehicle_enabled = msg->enabled && worker_status == auto_scene_gen_msgs::msg::StatusCode::ONLINE_AND_RUNNING;

    if (!vehicle_ok() && b_last_vehicle_ok_status)
    {
        vehicle_disabled_timestamp = sim_clock->now();
        log(LOG_INFO, "Vehicle is disabled.");

        if (b_registered_with_asg_client && !msg->preempted)
        {
            bool b_saved_node_data = true;
            if (strcmp(save_dir.c_str(), "none") == 0)
            {
                try
                {
                    save_node_data();
                }
                catch(const std::exception& e)
                {
                    b_saved_node_data = false;
                    b_request_rerun = true;
                    reason_for_rerun = "Base Class Issue: Encountered exception when saving node data: " + std::string(e.what());
                    log(LOG_WARN, "Requesting rerun with reason: " + reason_for_rerun);
                }
            }

            if (b_saved_node_data)
                check_for_rerun_request();

            // Save log file on client computer
            if (!save_dir.empty())
            {
                std::string client_filename = "vehicle_node_log_" + std::string(get_name()) + ".txt";
                save_file_on_remote_asg_client("log.txt", client_filename.c_str());
            }
        }

        // Empty contents in temporary save directory. Do this here, so if there is a problem resetting, we can still see the error
        if (boost::filesystem::is_directory(boost::filesystem::status(temp_save_dir)))
            boost::filesystem::remove_all(temp_save_dir);
        boost::filesystem::create_directories(temp_save_dir);

        reset();

        if (!msg->preempted)
        {
            last_scenario_number = scenario_number;
            send_ready_notification();
        }
    }

    if (vehicle_ok() && !b_last_vehicle_ok_status)
    {
        vehicle_enabled_timestamp = sim_clock->now();
        log(LOG_INFO, "Vehicle is enabled and ok.");
    }
    b_last_vehicle_ok_status = vehicle_ok();
}

void AutoSceneGenVehicleNode::vehicle_node_operating_info_cb(const auto_scene_gen_msgs::msg::VehicleNodeOperatingInfo::SharedPtr msg)
{
    for (uint16_t i=0; i < msg->worker_ids.size(); i++)
    {
        if (wid == msg->worker_ids[i])
        {
            scenario_number = msg->scenario_numbers[i];
            b_ok_to_run = msg->ok_to_run[i];
            save_dir = msg->save_dirs[i];
            b_save_minimal = msg->save_minimal[i];
            return;
        }
    }
}

void AutoSceneGenVehicleNode::scene_description_cb(const auto_scene_gen_msgs::msg::SceneDescription::SharedPtr msg)
{
    scene_description = msg;
}

void AutoSceneGenVehicleNode::register_node_with_asg_client(bool b_resending)
{
    if (!b_debug_mode && asg_client_status == auto_scene_gen_msgs::msg::StatusCode::ONLINE_AND_RUNNING)
    {
        auto request = std::make_shared<auto_scene_gen_msgs::srv::RegisterVehicleNode::Request>();
        request->worker_id = wid;
        request->node_name = std::string(get_name());
        register_node_send_time.set_time(std::chrono::steady_clock::now());
        register_node_future_response = register_node_cli->async_send_request(request);

        if (b_resending)
            log(LOG_INFO, "Resending RegisterVehicleNode request.");
        else
            log(LOG_INFO, "Sending RegisterVehicleNode request.");
    }
}

void AutoSceneGenVehicleNode::send_ready_notification(bool b_resending)
{
    if (!b_debug_mode && b_registered_with_asg_client)
    {
        auto request = std::make_shared<auto_scene_gen_msgs::srv::NotifyReady::Request>();
        request->worker_id = wid;
        request->node_name = std::string(get_name());
        request->last_scenario_number = last_scenario_number;
        request->request_rerun = b_request_rerun;
        request->reason_for_rerun = reason_for_rerun;

        notify_ready_send_time.set_time(std::chrono::steady_clock::now());
        notify_ready_future_response = notify_ready_cli->async_send_request(request);

        if (b_resending)
            log(LOG_INFO, "Resending NotifyReady request for scenario " + std::to_string(last_scenario_number) + ".");
        else
            log(LOG_INFO, "Sending NotifyReady request for scenario " + std::to_string(last_scenario_number) + ".");
    }
}

void AutoSceneGenVehicleNode::register_node_timer_cb()
{
    if (b_debug_mode) 
        return;

    if (!b_registered_with_asg_client && !b_services_online) 
        wait_for_services();
    
    // If vehicle node is not registered and register send time is invalid, then register node
    if (!b_registered_with_asg_client && !register_node_send_time.b_valid) 
        register_node_with_asg_client();

    // If vehicle node is not registered and register send time is valid, check if response is valid and process it
    if (!b_registered_with_asg_client && register_node_send_time.b_valid)
    {
        // If not registered within specified duration, attempt to re-register vehicle node
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::duration<double> >(now - register_node_send_time.time).count() > DUR_BEFORE_RESUBMITTING)
        {
            register_node_with_asg_client(true);
            return;
        }

        if (register_node_future_response.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready)
        {
            bool b_received = false;
            try
            {
                b_received = register_node_future_response.get()->received;
            }
            catch(const std::exception& e)
            {
                log(LOG_ERROR, "Registering vehicle node failed with exception: " + std::string(e.what()));
                register_node_with_asg_client(true); // Try again
                return;
            }

            if (b_received)
            {
                b_registered_with_asg_client = true;
                register_node_send_time.b_valid = false; // Send time is now invalid
                send_ready_notification();
            }
            else
            {
                register_node_with_asg_client(true); // Try again
                return;
            }
        }
    }

    // If vehicle node is registered and NotifyReady sent time is valid, check if response is valid and process it
    if (b_registered_with_asg_client && notify_ready_send_time.b_valid)
    {
        // If NotifyReady response not received within specified duration, attempt to send notification again
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::duration<double> >(now - notify_ready_send_time.time).count() > DUR_BEFORE_RESUBMITTING)
        {
            send_ready_notification(true);
            return;
        }

        if (notify_ready_future_response.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready)
        {
            bool b_received = false;
            bool b_accepted = false;
            try
            {
                b_received = notify_ready_future_response.get()->received;
                b_accepted = notify_ready_future_response.get()->accepted;
            }
            catch(const std::exception& e)
            {
                log(LOG_ERROR,"NotifyReady request failed with exception: " + std::string(e.what()));
                send_ready_notification(true); // Try again
            }

            if (b_received)
            {
                b_request_rerun = false;
                reason_for_rerun = "";
                notify_ready_send_time.b_valid = false; // Send time is now invalid
                std::string a = b_accepted ? "true" : "false";
                std::stringstream s;
                s << "NotifyReady response for scenario " << last_scenario_number << " was received with acceptance " << a;
                log(LOG_INFO, s.str());
            }
        }
    }
}

bool AutoSceneGenVehicleNode::vehicle_ok()
{
    if (b_debug_mode)
    {
        return b_vehicle_enabled;
    }
    else
    {
        return b_registered_with_asg_client  && !notify_ready_send_time.b_valid && b_vehicle_enabled && b_ok_to_run;
    }
}

void AutoSceneGenVehicleNode::shutdown()
{
    if (!b_debug_mode && asg_client_status == auto_scene_gen_msgs::msg::StatusCode::ONLINE_AND_RUNNING)
    {
        log(LOG_INFO, "Unregistering vehicle node.");
        auto request = std::make_shared<auto_scene_gen_msgs::srv::RegisterVehicleNode::Request>();
        request->worker_id = wid;
        request->node_name = std::string(get_name());
        std::shared_future<auto_scene_gen_msgs::srv::RegisterVehicleNode::Response::SharedPtr> response = unregister_node_cli->async_send_request(request);
        std::this_thread::sleep_for(100ms);
    }
}

void spin_vehicle_node(std::shared_ptr<auto_scene_gen_core::AutoSceneGenVehicleNode> node, size_t num_threads)
{
    rclcpp::ExecutorOptions options = rclcpp::ExecutorOptions();
    rclcpp::executors::MultiThreadedExecutor executor(options, num_threads);
    executor.add_node(node);

    try
    {
        while (rclcpp::ok())
            executor.spin_once();
    }
    catch(const std::runtime_error& e)
    {
        node->log(auto_scene_gen_core::AutoSceneGenVehicleNode::LOG_ERROR, "Runtime Error: " + std::string(e.what()));
    }
    catch(const std::exception& e)
    {
        node->log(auto_scene_gen_core::AutoSceneGenVehicleNode::LOG_ERROR, "Exception: " + std::string(e.what()));
    }

    node->shutdown();
    rclcpp::shutdown();
}

} // namespace auto_scene_gen_core