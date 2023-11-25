#include "Network.h"
#include <sstream>
#include <string>
#include <ctime>
Network::Network() {

}
std::string Network::deleteSubstring(std::string str) {
    // Removing spaces from the end
    while (!str.empty() && std::isspace(str.back())) {
        str.pop_back();
    }

    // Removing spaces from the beginning
    auto it = str.begin();
    while (it != str.end() && std::isspace(*it)) {
        ++it;
    }
    str.erase(str.begin(), it);
    str = str.substr(1, str.size() - 2);
    return str;
}
int Network::find_frame_size(string message, int message_limit){
    int msize = message.size();
    int size = msize / message_limit;
    if (msize % message_limit != 0){
        size++;
    }
    return size;
}
Client* Network::find_client( string id){
    for (int i = 0; i < clientss.size(); ++i) {
        if (clientss[i].client_id == id){
            return &clientss[i];
        }
    }
    return nullptr;
}
Client* Network::find_client_MAC( string mac){
    for (int i = 0; i < clientss.size(); ++i) {
        if (clientss[i].client_mac == mac){
            return &clientss[i];
        }
    }
    return nullptr;
}

string Network::find_MAC(string id){
    Client* client = find_client(id);
    return client->client_mac;
}

void Network::print_frame(stack<Packet*> frame){
    /*Sender MAC address: DDDDDDDDDD, Receiver MAC address: EEEEEEEEEE
    Sender IP address: 8.8.8.8, Receiver IP address: 0.0.1.1
    Sender port number: 0706, Receiver port number: 0607
    Sender ID: C, Receiver ID: E
    Message chunk carried: "ge."
    Number of hops so far: 3
    */
   PhysicalLayerPacket* pypacket = dynamic_cast<PhysicalLayerPacket*>(frame.top());
   pypacket->print();
    frame.pop();
    NetworkLayerPacket* nlpacket = dynamic_cast<NetworkLayerPacket*>(frame.top());
    nlpacket->print();
    frame.pop();
    TransportLayerPacket* tlpacket = dynamic_cast<TransportLayerPacket*>(frame.top());
    tlpacket->print();
    frame.pop();
    ApplicationLayerPacket* alpacket = dynamic_cast<ApplicationLayerPacket*>(frame.top());
    alpacket->print();
    cout << "Message chunk carried: " << "'" << alpacket->message_data  << '"' << endl;
    cout << "Number of hops so far: " << pypacket->nofhops << endl;
    cout << "--------" << endl;

}
void Network::put_to_Queue(string sender_id, string receiver_id,
 string message,int message_limit,const string &sender_port, const string &receiver_port){
    //find size and initialize the no
    int frame_size = find_frame_size(message, message_limit);
    int no = 0;

    //find the client sender and receiver
    Client* client = find_client( sender_id);
    Client* receiver = find_client(receiver_id);


    //print the title of message
    //Message to be sent: "A few small hops for frames, but a giant leap for this message."

    cout << "Message to be sent: " <<'"' <<  message << '"'     << endl << endl;
    for (int i = 0; i < frame_size; ++i) {
        no++;

        //message to be added
        string data = message.substr(i*message_limit, message_limit);

        //add it to queue
        stack<Packet*> for_out_queue;
        for_out_queue.push(new ApplicationLayerPacket(0, sender_id, receiver_id, data));
        for_out_queue.push(new TransportLayerPacket(1, sender_port, receiver_port));
        for_out_queue.push(new NetworkLayerPacket(2, client->client_ip, receiver->client_ip));
        string tmpstr = find_MAC( client->routing_table[receiver_id]);
        for_out_queue.push(new PhysicalLayerPacket(3, client->client_mac,tmpstr ));
        client->outgoing_queue.push(for_out_queue);
        //should print
        cout << "Frame #" << no  << endl;
        print_frame(for_out_queue);
    }
}
void Network::show_frame_info(string info_id , string out_in, int frame_no){
    /*--------------------------------
    Command: SHOW_FRAME_INFO C out 3
    --------------------------------
    Current Frame #3 on the outgoing queue of client C
    Carried Message: " leap for this messa"
    Layer 0 info: Sender ID: C, Receiver ID: E
    Layer 1 info: Sender port number: 0706, Receiver port number: 0607
    Layer 2 info: Sender IP address: 8.8.8.8, Receiver IP address: 0.0.1.1
    Layer 3 info: Sender MAC address: CCCCCCCCCC, Receiver MAC address: BBBBBBBBBB
    Number of hops so far: 0
    */
    Client* client = find_client(info_id);
    queue<stack<Packet*>> temp;
    if (out_in == "out"){
        temp = client->outgoing_queue;
    }else{
        temp = client->incoming_queue;
    }
    if (frame_no > temp.size()){
        cout << "No such frame." << endl;
        return;
    }
    for (int i = 0; i+1 < frame_no; i++) {
        temp.pop();
    }
    stack<Packet*> frame = temp.front();
    cout << "Current Frame #" << frame_no << " on the " << out_in << "going queue of client " << info_id << endl;
    
    PhysicalLayerPacket* pypacket = dynamic_cast<PhysicalLayerPacket*>(frame.top());
    frame.pop();
    NetworkLayerPacket* nlpacket = dynamic_cast<NetworkLayerPacket*>(frame.top());
    frame.pop();
    TransportLayerPacket* tlpacket = dynamic_cast<TransportLayerPacket*>(frame.top());
    frame.pop();
    ApplicationLayerPacket* alpacket = dynamic_cast<ApplicationLayerPacket*>(frame.top());
    cout << "Carried Message: " << '"' << alpacket->message_data << '"' << endl;
    cout << "Layer 0 info: Sender ID: " << alpacket->sender_ID << ", Receiver ID: " << alpacket->receiver_ID << endl;
    cout << "Layer 1 info: Sender port number: " << tlpacket->sender_port_number << ", Receiver port number: " << tlpacket->receiver_port_number << endl;
    cout << "Layer 2 info: Sender IP address: " << nlpacket->sender_IP_address << ", Receiver IP address: " << nlpacket->receiver_IP_address << endl;
    cout << "Layer 3 info: Sender MAC address: " << pypacket->sender_MAC_address << ", Receiver MAC address: " << pypacket->receiver_MAC_address << endl;
    cout << "Number of hops so far: " << pypacket->nofhops << endl;

}
void Network::show_Q_info(string info_id , string out_in){
    /*Client C Outgoing Queue Status
    Current total number of frames: 5*/
    Client* client = find_client(info_id);
    queue<stack<Packet*>> temp;
    if (out_in == "out"){
        temp = client->outgoing_queue;
    }else{
        temp = client->incoming_queue;
    }
    cout << "Client " << info_id << " " << out_in << "going Queue Status" << endl;
    cout << "Current total number of frames: " << temp.size() << endl;
}
bool Network::containsPunctuation(const std::string& str) {
    return str.find_first_of(".?!") != std::string::npos;
}
bool Network::check_end(stack<Packet*> frame){
    frame.pop();
    frame.pop();
    frame.pop();
    ApplicationLayerPacket* alpacket = dynamic_cast<ApplicationLayerPacket*>(frame.top());
    if (containsPunctuation(alpacket->message_data)){
        return true;
    }
    return false;
}

void Network::send(){
    /*SEND command triggers the transmission of message frames from all clients’ outgoing queues
    to their respective next hop in the network, determined by the receiver MAC address for each
    message frame. The format of this command is:
    SEND
    When this command is given, all frames in any client’s outgoing queue will be forwarded from
    the sender to the next hop recipient’s incoming queue, with hop counts updated accordingly. A
    network trace will be printed to STDOUT in the following format:
    */

   // don't forget to update the hop count ok
   // don't forget to update log entries ok
    // don't forget to update the outgoing queue of the sender ok
    // don't forget to update the incoming queue of the receiverok
    //IMLPEMENT THE CASE WHERE IT CANT BE SENT there is no such case
   for (int i = 0; i < clientss.size(); i++){
        Client* client = &clientss[i];
        //initialize the frame
        queue<stack<Packet*>> temp;

        if (client->outgoing_queue.empty()){

            continue;
        }
        temp = client->outgoing_queue;
        int no = 0;
        string message = "";
        while (!temp.empty()){
            no++;

            stack<Packet*> frame = temp.front();
            //get the packets
            PhysicalLayerPacket* pypacket = dynamic_cast<PhysicalLayerPacket*>(frame.top());
            pypacket->nofhops++;
            frame.pop();
            NetworkLayerPacket* nlpacket = dynamic_cast<NetworkLayerPacket*>(frame.top());
            frame.pop();
            TransportLayerPacket* tlpacket = dynamic_cast<TransportLayerPacket*>(frame.top());
            frame.pop();
            ApplicationLayerPacket* alpacket = dynamic_cast<ApplicationLayerPacket*>(frame.top());
            //print them
            cout << "Client " << find_client_MAC(pypacket->sender_MAC_address)->client_id << " sending frame #" << no << " to client " << find_client_MAC(pypacket->receiver_MAC_address)->client_id << endl;
            print_frame(temp.front());
            //DONT FORGET  TO ADD NOTES.TXT
            message += alpacket->message_data;
            if (check_end(temp.front()) && pypacket->tolog){
                //update the log entries
                /*Log Entry #1:
                Activity: Message Forwarded
                Timestamp: 2023-11-22 20:30:03
                Number of frames: 4
                Number of hops: 2
                Sender ID: C
                Receiver ID: E
                Success: Yes
                */
                std::time_t currentTime = std::time(nullptr);
                std::tm* timestamp = std::localtime(&currentTime);
                ActivityType type = ActivityType::MESSAGE_SENT;
                std::ostringstream oss;

                oss <<  std::put_time(timestamp, "%Y-%m-%d %H:%M:%S");
                std::string timestampString = oss.str();
                Log log(timestampString,message,no,pypacket->nofhops-1,alpacket->sender_ID,alpacket->receiver_ID,true,type);
                client->log_entries.push_back(log);
                no = 0;

                message = "";
            }


            //update the outgoing queue of the sen ,der
            string to_go = client->routing_table[alpacket->receiver_ID];
            find_client(to_go)->incoming_queue.push(temp.front());
            //delete the frame from the outgoing queue
            client->outgoing_queue.pop();
            temp = client->outgoing_queue;

        }

   }
}
void Network::receive(){
    for (int i = 0; i < clientss.size(); i++){
        Client* client = &clientss[i];
        //initialize the frame
        queue<stack<Packet*>> temp;

        if (client->incoming_queue.empty()){

            continue;
        }
        temp = client->incoming_queue;
        int no = 0;
        string message = "";

        while (!temp.empty()){
            no++;

            stack<Packet*> frame = temp.front();
            //get the packets
             PhysicalLayerPacket* pypacket = dynamic_cast<PhysicalLayerPacket*>(frame.top());
            frame.pop();
            NetworkLayerPacket* nlpacket = dynamic_cast<NetworkLayerPacket*>(frame.top());
            frame.pop();
            TransportLayerPacket* tlpacket = dynamic_cast<TransportLayerPacket*>(frame.top());
            frame.pop();
            ApplicationLayerPacket* alpacket = dynamic_cast<ApplicationLayerPacket*>(frame.top());

            // 3 possible cases
            // 1. the receiver is the client
            // 2. the receiver is not the client and will be forwarded
            // 3. the receiver is not the client and will be dropped
            bool succes = true;
            ActivityType type = ActivityType::MESSAGE_RECEIVED;
            bool needDel = false;
            if (alpacket->receiver_ID == client->client_id){
                //case 1
                //print them
                //Client E receiving frame #1 from client D, originating from client C
                message += alpacket->message_data;
                cout << "Client " << client->client_id << " receiving frame #" << no << " from client " 
                << find_client_MAC(pypacket->sender_MAC_address)->client_id << ", originating from client " << alpacket->sender_ID << endl;
                print_frame(temp.front());
                needDel = true;
                //no sondamı bak
            }//case 3 
            else if (client->routing_table.count(client->routing_table[alpacket->receiver_ID]) == 0){
                //print them
                /* Client B receiving frame #1 from client C, but intended for client E. Forwarding...
                Error: Unreachable destination. Packets are dropped after 1 hops!*/
                cout << "Client " << client->client_id << " receiving frame #" << no << " from client "
                << find_client_MAC(pypacket->sender_MAC_address)->client_id << ", but intended for client " << alpacket->receiver_ID << ". Forwarding..." << endl;
                cout << "Error: Unreachable destination. Packets are dropped after " << pypacket->nofhops << " hops!" << endl;
                type = ActivityType::MESSAGE_DROPPED;
                succes = false;
                needDel = true;
                
            }else{
                /*Client B receiving a message from client C, but intended for client E. Forwarding... 
                Frame #1 MAC address change: New sender MAC BBBBBBBBBB, new receiver MAC DDDDDDDDDD
                Frame #2 MAC address change: New sender MAC BBBBBBBBBB, new receiver MAC DDDDDDDDDD
                Frame #3 MAC address change: New sender MAC BBBBBBBBBB, new receiver MAC DDDDDDDDDD
                Frame #4 MAC address change: New sender MAC BBBBBBBBBB, new receiver MAC DDDDDDDDDD*/
                if (no == 1){
                    cout << "Client " << client->client_id << " receiving a message from client " << find_client_MAC(pypacket->sender_MAC_address)->client_id << ", but intended for client " << alpacket->receiver_ID << ". Forwarding..." << endl;

                    //cout << client->client_id << " receiving frame #" << no << " from client "
                    //<< find_client_MAC(pypacket->sender_MAC_address)->client_id << ", but intended for client " << alpacket->receiver_ID << ". Forwarding..." << endl;
                }
                pypacket->sender_MAC_address = client->client_mac;
                pypacket->receiver_MAC_address = find_client(client->routing_table[alpacket->receiver_ID])->client_mac;
                cout << "Frame #" << no << " MAC address change: New sender MAC " << pypacket->sender_MAC_address << ", new receiver MAC " << pypacket->receiver_MAC_address << endl;

                //add to the outgoing queue of the sender
                string to_go = client->routing_table[alpacket->receiver_ID];
                client->outgoing_queue.push(temp.front());
                type = ActivityType::MESSAGE_FORWARDED;
            }
            if (type == ActivityType::MESSAGE_FORWARDED){
                pypacket->tolog = false;
            }
             if (check_end(temp.front())){
                //update the log entries
                /*Log Entry #1:
                Activity: Message Forwarded
                Timestamp: 2023-11-22 20:30:03
                Number of frames: 4
                Number of hops: 2
                Sender ID: C
                Receiver ID: E
                Success: Yes
                */
                std::time_t currentTime = std::time(nullptr);
                std::tm* timestamp = std::localtime(&currentTime);
                std::ostringstream oss;

                oss <<  std::put_time(timestamp, "%Y-%m-%d %H:%M:%S");
                std::string timestampString = oss.str();

                Log log(timestampString,message,no,pypacket->nofhops,alpacket->sender_ID,alpacket->receiver_ID,succes,type);
                client->log_entries.push_back(log);
                no = 0;
                if (type==ActivityType::MESSAGE_RECEIVED){
                    //Client E received the message "A few small hops for frames, but a giant leap for this message." from client C.
                    cout << "Client " << client->client_id << " received the message " << '"' << message << '"' << " from client " << alpacket->sender_ID << "." << endl;
                }
                cout << "--------" << endl;
                message = "";

            }
            
            //delete the frame from the nigoing queue
            stack<Packet*> tmpPack =client->incoming_queue.front();
            if (needDel){
                delete tmpPack.top();
                tmpPack.pop();
                delete tmpPack.top();
                tmpPack.pop();
                delete tmpPack.top();
                tmpPack.pop();
                delete tmpPack.top();
            }

             client->incoming_queue.pop();
            
            temp = client->incoming_queue;

        }
    }
}
void Network::print_log(string log_id){
    Client* client = find_client(log_id);
    vector<Log> logs = client->log_entries;
    if (logs.size()==0){
        return;
    }
    cout << "Client " << log_id << " Logs:" << endl;
    /*--------------
    Log Entry #1:
    Activity: Message Forwarded
    Timestamp: 2023-11-22 20:30:03
    Number of frames: 4
    Number of hops: 2
    Sender ID: C
    Receiver ID: E
    Success: Yes*/
    for (int i = 0; i < logs.size(); ++i) {
        cout << "--------------" << endl;
        cout << "Log Entry #" << i+1 << ":" << endl;
        logs[i].print();
    }
}
void Network::process_commands(vector<Client> &clients, vector<string> &commands, int message_limit,
                      const string &sender_port, const string &receiver_port) {
    // TODO: Execute the commands given as a vector of strings while utilizing the remaining arguments.
    /* Don't use any static variables, assume this method will be called over and over during testing.
     Don't forget to update the necessary member variables after processing each command. For example,
     after the MESSAGE command, the outgoing queue of the sender must have the expected frames ready to send. */

     clientss = clients;
    for (int i = 0; i < commands.size(); ++i) {
        //get command
        std::stringstream ss(commands[i]);
        string command;
        ss >> command;


        //print command
        std::cout << "--------------------";
        std::cout << "\nCommand: " << commands[i] << std::endl;
        std::cout << "--------------------\n";
        if (command == "MESSAGE") {
            //find the client sender
            string sender_id, receiver_id, message;
            ss >> sender_id >> receiver_id ;
            std::getline(ss, message);
            message = deleteSubstring(message);
            put_to_Queue(sender_id, receiver_id, message, message_limit, sender_port, receiver_port);
    
        } else if (command == "SHOW_FRAME_INFO"){
            string info_id , out_in;
            int frame_no;
            ss >> info_id >> out_in >> frame_no;
            show_frame_info(info_id, out_in, frame_no);
        }else if (command == "SHOW_Q_INFO"){
            string info_id , out_in;
            ss >> info_id >> out_in ;
            show_Q_info(info_id, out_in);
        }else if (command == "SEND"){
            send();
        }else if (command == "RECEIVE"){
            receive();
        }else if (command == "PRINT_LOG"){
            string log_id;
            ss >> log_id;
            print_log(log_id);
        }


        else{
            cout << "Invalid command." << endl;
        }
    }
}

vector<Client> Network::read_clients(const string &filename) {
    // TODO: Read clients from the given input file and return a vector of Client instances.
    vector<Client> clients;
    std::string line;
    fstream file(filename, ios::in);
    int count ;
    file >> count;
    std::getline(file, line);
    for (int i = 0; i < count; ++i){
        std::getline(file, line);
        std::stringstream ss(line);
        string id, ip, mac;
        ss >> id >> ip >> mac;
        Client client(id, ip, mac);
        clients.push_back(client);
    }

    return clients;
}

void Network::read_routing_tables(vector<Client> &clients, const string &filename) {
    // TODO: Read the routing tables from the given input file and populate the clients' routing_table member variable.
    std::string line;
    fstream file(filename, ios::in);
    int count = clients.size();
    for (int i = 0; i < count; i++){
        for (int j = 0; j+1 < count; j++){

            std::getline(file, line);
            std::stringstream ss(line);
            string id,id2;
            ss >> id >> id2;
            clients[i].routing_table[id] = id2;
        }
        std::getline(file, line);
    }
}

// Returns a list of token lists for each command
vector<string> Network::read_commands(const string &filename) {
    vector<string> commands;
    // TODO: Read commands from the given input file and return them as a vector of strings.
    fstream file(filename, ios::in);
    string line;
    getline(file, line);
    int count = stoi(line);

    for (int i = 0; i < count; ++i) {
        getline(file, line);
        commands.push_back(line);
    }
    return commands;
}

Network::~Network() {
    // TODO: Free any dynamically allocated memory if necessary.

}
