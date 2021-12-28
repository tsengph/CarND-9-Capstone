#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from geometry_msgs.msg import TwistStamped
import math

from std_msgs.msg import Int32
from sensor_msgs.msg import PointCloud2
from scipy.spatial import KDTree
import numpy as np
import math
from scipy.interpolate import spline
import tf
from geometry_msgs.msg import Quaternion
import copy

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/vehicle/obstacle_points', PointCloud2, self.obstacle_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.current_vel = 0

        self.stopline_wp_idx = -1
        self.base_lane = None
        self.red_light_lane_wps = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def velocity_cb(self, msg):
        self.current_vel = msg.twist.linear.x

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)


    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
            self.red_light_lane_wps = None
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def quaternion_from_yaw(self, yaw):
        return tf.transformations.quaternion_from_euler(0., 0., yaw)

    def distance_wp(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)

    def decelerate(self, waypoints):
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.

        for wp in waypoints[:-1][::-1]:
            dist = self.distance_wp(wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints

    def decelerate_waypoints(self, waypoints, closest_idx):

        temp = []
        closest_idx = self.get_closest_waypoint_idx()
        base_wp_len = len(self.base_waypoints.waypoints)
        car_nearwp_dis =  self.distance_wp(self.pose.pose.position,  self.base_waypoints.waypoints[(closest_idx - 1 ) % base_wp_len].pose.pose.position ) \
                          + self.distance_wp(self.pose.pose.position,  self.base_waypoints.waypoints[(closest_idx + 1 ) % base_wp_len].pose.pose.position )
        current_prev_dis = self.distance_wp(self.base_waypoints.waypoints[(closest_idx - 1 ) % base_wp_len ].pose.pose.position,  self.base_waypoints.waypoints[closest_idx].pose.pose.position )
        current_next_dis = self.distance_wp(self.base_waypoints.waypoints[(closest_idx + 1 ) % base_wp_len ].pose.pose.position,  self.base_waypoints.waypoints[closest_idx].pose.pose.position )

        if self.red_light_lane_wps:
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y
            red_light_lane_wps_2d = [[waypoint.pose.pose.position.x , waypoint.pose.pose.position.y] for waypoint in self.red_light_lane_wps]
            red_light_lane_wps_2d_tree = KDTree(red_light_lane_wps_2d)
            temp_closest_idx = red_light_lane_wps_2d_tree.query([x, y], 1)[1]
            for i in range(len(self.red_light_lane_wps)):
                if i >= temp_closest_idx:
                    temp.append(self.red_light_lane_wps[i])

        elif car_nearwp_dis > ( (current_prev_dis + current_next_dis ) * 1.2):
            waypoints_array = np.array([[waypoint.pose.pose.position.x , waypoint.pose.pose.position.y] for waypoint in self.base_waypoints.waypoints])
            car_x = None # the position x of the car
            car_y = None # the position y of the car

            if self.pose:
                car_x = self.pose.pose.position.x
                car_y = self.pose.pose.position.y

            spline_x = [] # x, use to spline
            spline_y = [] # y, use to spline
            spline_x.append(car_x)
            spline_y.append(car_y)
            spline_x.extend(waypoints_array[:, 0][(closest_idx + 5) % base_wp_len : max((closest_idx + 8) % base_wp_len, (self.stopline_wp_idx + 6) % base_wp_len) ])
            spline_y.extend(waypoints_array[:, 1][(closest_idx + 5) % base_wp_len : max((closest_idx + 8) % base_wp_len, (self.stopline_wp_idx + 6) % base_wp_len) ])

            spline_x = np.array(spline_x)
            spline_y = np.array(spline_y)

            x_new = np.linspace(spline_x.min(),spline_x.max(), 20)
            y_new = spline(spline_x, spline_y, x_new)

            x_new_len = len(x_new)
            vel = max(self.current_vel, self.base_waypoints.waypoints[0].twist.twist.linear.x)
            for i in range(x_new_len):
                yaw = 0
                if i < x_new_len - 1:
                    dx = x_new[i + 1] - x_new[i]
                    dy = y_new[i + 1] - y_new[i]
                    yaw = math.atan2(dy, dx)

                p = Waypoint()
                p.pose.pose.position.x = float(x_new[i])
                p.pose.pose.position.y = float(y_new[i])
                p.pose.pose.position.z = float(0)
                q = self.quaternion_from_yaw(float(yaw))
                p.pose.pose.orientation = Quaternion(*q)
                p.twist.twist.linear.x = float(0)

                temp.append(p)

            light_x = self.base_waypoints.waypoints[self.stopline_wp_idx].pose.pose.position.x
            light_y = self.base_waypoints.waypoints[self.stopline_wp_idx].pose.pose.position.y
            temp_wps_2d = [[waypoint.pose.pose.position.x , waypoint.pose.pose.position.y] for waypoint in temp]
            temp_wps_2d_tree = KDTree(temp_wps_2d)
            temp_closest_idx = temp_wps_2d_tree.query([light_x, light_y], 1)[1]
            temp = temp[:temp_closest_idx - 1]

        else:
            for i in range(closest_idx , self.stopline_wp_idx - 1):
                temp.append(copy.deepcopy(self.base_waypoints.waypoints[i]))

        temp_wp = self.decelerate(temp)

        # for debug only
        # temp_log = []
        # for i in range(len(temp_wp)):
        #     temp_log.append([temp_wp[i].pose.pose.position.x , temp_wp[i].pose.pose.position.y, temp_wp[i].twist.twist.linear.x ])
        # rospy.loginfo(" %s , --------------------- temp_log : %s ", self.show_how_to, temp_log)

        self.red_light_lane_wps = temp_wp
        return temp_wp

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        self.base_lane = waypoints

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        waypoints_len = len(waypoints)

        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i % waypoints_len].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
