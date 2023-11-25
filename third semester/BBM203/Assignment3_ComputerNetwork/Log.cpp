//
// Created by alperen on 2.10.2023.
//

#include "Log.h"

Log::Log(const string &_timestamp, const string &_message, int _number_of_frames, int _number_of_hops, const string &_sender_id,
         const string &_receiver_id, bool _success, ActivityType _type) {
    timestamp = _timestamp;
    message_content = _message;
    number_of_frames = _number_of_frames;
    number_of_hops = _number_of_hops;
    sender_id = _sender_id;
    receiver_id = _receiver_id;
    success_status = _success;
    activity_type = _type;
}

Log::~Log() {
    // TODO: Free any dynamically allocated memory if necessary.
}
void Log::print() {
    /*    Activity: Message Forwarded
    Timestamp: 2023-11-22 20:30:03
    Number of frames: 4
    Number of hops: 2
    Sender ID: C
    Receiver ID: E
    Success: Yes*/
    string activity;
    if (activity_type == ActivityType::MESSAGE_RECEIVED) {
        activity = "Message Received";
    } else if (activity_type == ActivityType::MESSAGE_FORWARDED) {
        activity = "Message Forwarded";
    } else if (activity_type == ActivityType::MESSAGE_SENT) {
        activity = "Message Sent";
    } else if (activity_type == ActivityType::MESSAGE_DROPPED) {
        activity = "Message Dropped";
    }
    cout << "Activity: " << activity << endl;
    cout << "Timestamp: " << timestamp << endl;
    cout << "Number of frames: " << number_of_frames << endl;
    cout << "Number of hops: " << number_of_hops << endl;
    cout << "Sender ID: " << sender_id << endl;
    cout << "Receiver ID: " << receiver_id << endl;
    string success;
    if (success_status) {
        success = "Yes";
    } else {
        success = "No";
    }
    cout << "Success: " << success << endl;
    if ( message_content != ""){cout << "Message: " << '"' << message_content << '"' << endl;}
    
}