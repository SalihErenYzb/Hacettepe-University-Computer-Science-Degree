#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <iostream>
#include "Packet.h"
#include "Client.h"

using namespace std;

class Network {
public:
    Network();
    ~Network();
    vector<Client> clientss;
    // Executes commands given as a vector of strings while utilizing the remaining arguments.
    void process_commands(vector<Client> &clients, vector<string> &commands, int message_limit, const string &sender_port,
                     const string &receiver_port);
    std::string deleteSubstring(std::string str);
    Client* find_client( string id);
    Client* find_client_MAC( string MAC);

    void put_to_Queue( string sender_id, string receiver_id,
     string message,int message_limit,const string &sender_port, const string &receiver_port);
    int find_frame_size(string message, int message_limit);
    // Initialize the network from the input files.
    vector<Client> read_clients(string const &filename);
    void read_routing_tables(vector<Client> & clients, string const &filename);
    vector<string> read_commands(const string &filename); 
    void print_frame(stack<Packet*> frame);
    void show_frame_info(string info_id , string out_in, int frame_no);
    void show_Q_info(string info_id , string out_in);
    void send();
    bool check_end(stack<Packet*> frame);
    bool containsPunctuation(const std::string& str);
    string find_MAC(string id);
    void receive();
    void print_log(string log_id);
    
};

#endif  // NETWORK_H
