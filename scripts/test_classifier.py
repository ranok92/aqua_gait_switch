#!/usr/bin/env python
import pickle
import rospy
import rostopic
import pdb
import pandas as pd
from sklearn import svm
import numpy as np
from sensor_msgs.msg import Imu
from aquacore.msg import StateMsg
from message_filters import Subscriber, ApproximateTimeSynchronizer, Cache
from geometry_msgs.msg import PoseStamped
from rospy_message_converter import message_converter
from std_msgs.msg import String
from matplotlib import pyplot as plt



#****************global variables******************

#name of the topics to subscribe
topics_to_subscribe = ['/aqua/state', '/aqua/imu']

#message types for the topics to subscribe
msg_types = [StateMsg, Imu]
'''
info_to_extract = ['/aqua/state.LegCurrents.0','/aqua/state.LegCurrents.1',
                   '/aqua/state.LegVoltages.0','/aqua/state.LegVoltages.1',
                   '/aqua/imu.orientation.x', '/aqua/imu.orientation.y']
'''

#the information from within the topics that are needed by the classifier
info_to_extract = ['/aqua/imu.orientation.y']

#the path to the classifier
path_to_classifier = "/home/abhisek/Study/Robotics/AquaAutonomousEntryExit/svm_classifier_sw15.model"

#load the classifier
classifier = pickle.load(open(path_to_classifier,'rb'))

#initialize the publisher
pub_classifier = rospy.Publisher('/gait_prediction', String, queue_size =10)

#dictionary containing the  
info_container_dict = {}
prediction_list = []
filtered_prediction = []


x_axis = []
start_of_feed_time = None
diff_in_secs = 0
median_filter_size = 1
#***************************************************
#*****************for debugging*********************
recreated_data = []
#***************************************************

'''
#class that holds two things: 
        1. the name of the topic 
        2. the corresponding 
#message received from the topic
'''

class InfoContainer():
    def __init__(self, topic):

        self.topic = topic
        self.data = None



def callback_state(statedata):
    '''
    the callback function for the state topic
    '''

    global info_container_dict, sync_data

    #store data received by the subscriber into the info container
    info_container_dict[topics_to_subscribe[0]].data = statedata

    '''
    rospy.loginfo(rospy.get_caller_id() + "I heard %s\n", imu.header)
    rospy.loginfo(rospy.get_caller_id() + "and also %s\n", state.header)
    rospy.loginfo("difference in time %d\n", float((state.header.stamp.nsecs - imu.header.stamp.nsecs)/10000))
    '''

    rospy.loginfo('Calling from state callback')

    #if both the imu and state data is available synchronize the data and
    #call the classifier for prediction

    if info_container_dict[topics_to_subscribe[0]].data is not None and info_container_dict[topics_to_subscribe[1]].data is not None:
        sync_data = synchronize_data(info_to_extract)
    else:
        rospy.loginfo("Still waiting for one of the data.")



def callback_imu(imudata):
    '''
    the callback function for IMU data
    '''
    global info_container_dict, sync_data


    info_container_dict[topics_to_subscribe[1]].data = imudata
    rospy.loginfo("Calling from imu callback")

    #if both the imu and the state data is available, synchronize the
    # data and call the classifier for prediction

    if info_container_dict[topics_to_subscribe[0]].data is not None and info_container_dict[topics_to_subscribe[1]].data is not None:
        sync_data = synchronize_data(info_to_extract)
    else:
        rospy.loginfo("Still waiting for one of the data.")


def retrieve_info(msg_dict, field_to_extract):
    '''
    Retrieve specific fields from a ros message dictionary
    input msg_dict: Dictionary version of a ros message
          field_to_extract : name of the field to extract
                            e.g. : /aqua/imu.orientation.x

    output : the value of the field in the message dictionary
    field_to_extract will be of the format  /msg/name.primargfield.secondaryfield.andso_on
    '''
    field_segments = field_to_extract.strip().split('.')

    #check if the field to extract is a particular index from a list
    '''
    e.g if we need to extract the value from the second index of a list or array (list1[2]),
    it is expressed as list1.2

    '''

    contains_list_index = False
    #check if last character is a digit 
    #if true, its a index in a list
    if field_segments[-1].isdigit():
        contains_list_index =True

    #the first value is the name of the topic, so that is excluded

    if contains_list_index:
        key_list = field_segments[1:-1]
    else:
        key_list = field_segments[1:]

    cur_dict = msg_dict
    #checks if there are nested dictionaries and
    #keeps going until it reaches the required value
    for key in key_list:

        cur_dict = cur_dict[key]

    if contains_list_index:
        return cur_dict[int(field_segments[-1])]
    else:
        return cur_dict





def synchronize_data(info_to_extract):
    '''
    Synchronizes data currently present in the info_container_dict
    and publishes a prediction based on the synchronized data

    input: info_to_extract : list of columns to extract information from
                            synchronizes the data and collects the necessary message fields from 
                            the entire message

    output : -
    '''
    #global state, imu 
    global start_of_feed_time
    global median_filter_size

    state =  info_container_dict[topics_to_subscribe[0]].data
    imu = info_container_dict[topics_to_subscribe[1]].data
    
    #convert message received from the subscribers into respective dictionaries
    state_dict = message_converter.convert_ros_message_to_dictionary(state)
    imu_dict = message_converter.convert_ros_message_to_dictionary(imu)

    #print "the state dictionary :", state_dict
    #print "the imu dictionary :", imu_dict

    if start_of_feed_time is None:
        start_of_feed_time = state_dict['header']['stamp']

    current_time = state_dict['header']['stamp']

    #get the current time wrt to the start time
    diff_sec = int(current_time['secs']) - int(start_of_feed_time['secs'])
    diff_nano_sec = int(current_time['nsecs']) - int(start_of_feed_time['nsecs'])

    time_diff = diff_sec*(10**9) + diff_nano_sec

    global diff_in_secs
    diff_in_secs = diff_sec
    x_axis.append(time_diff)
    data_list = []

    #pick the data columns from the topics needed to make the prediction
    for info_name in info_to_extract:

        topic = info_name.strip().split('.')[0]
        info_dict = message_converter.convert_ros_message_to_dictionary(info_container_dict[topic].data)
        retrieved_info = retrieve_info(info_dict, info_name)
        data_list.append(retrieved_info)
    
    #recreated_data is there for debugging purposes
    recreated_data.append(data_list)

    #make the prediction
    prediction = classifier.predict(np.expand_dims(np.asarray(data_list), axis=0)).item()
    prediction_list.append(prediction)

    #apply a median filter to the predictions to smooth out the prediction 
    #resutls
    if len(prediction_list) >= median_filter_size:
        median_val = np.floor(np.median(np.asarray(prediction_list[-median_filter_size:])))
    else:
        median_val = np.floor(np.median(np.asarray(prediction_list)))
    filtered_prediction.append(median_val)

    #publish the smoothed prediction value
    pub_classifier.publish(str(median_val))
    #rospy.loginfo("The data after synchronization :",  data_list )




def listener(topics_to_subscribe, msg_types):
    '''
    the listener function
    '''
    rospy.init_node('test_classifier', anonymous=True)

    #list of topics the classifier subscribes to
    global state, imu
    rospy.Subscriber(topics_to_subscribe[0], msg_types[0], callback_state)
    #pose_data = message_filters.Subscriber(topics_to_subscribe[0], msg_types[0])
    rospy.Subscriber(topics_to_subscribe[1], msg_types[1], callback_imu)

    rospy.spin()
    rospy.on_shutdown(myhook)



def myhook():

    '''
    plots the predictions made wrt time 
    '''
    data = np.asarray(recreated_data)
    
    print "the shape of the data : ", data.shape    
    data = pd.DataFrame(data=data, columns=info_to_extract)

    file_to_save = 'training_data_quebec_48-38-recreated.csv'
    data.to_pickle(file_to_save)
    plt.plot(x_axis, filtered_prediction)
    plt.xticks(np.arange(min(x_axis), max(x_axis), 10**10), np.arange(min(x_axis), max(x_axis), 10**10)/10**9)
    plt.xlabel('time in seconds ')
    plt.ylabel('prediction')
    plt.show()

    print 'data saved'




if __name__ == '__main__':

    global state, imu, info_container_dict

    for topic in topics_to_subscribe:
        
        info_container_dict[topic] = InfoContainer(topic)

    sync_data = None
    listener(topics_to_subscribe, msg_types)
