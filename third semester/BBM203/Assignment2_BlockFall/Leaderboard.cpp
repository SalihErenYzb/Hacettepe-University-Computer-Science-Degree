#include "Leaderboard.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>

using namespace std;

void Leaderboard::insert_new_entry(LeaderboardEntry * new_entry) {
    // TODO: Insert a new LeaderboardEntry instance into the leaderboard, such that the order of the high-scores
    //       is maintained, and the leaderboard size does not exceed 10 entries at any given time (only the
    //       top 10 all-time high-scores should be kept in descending order by the score).
    LeaderboardEntry *curr = head_leaderboard_entry;
    LeaderboardEntry *prev = nullptr;
    size++;
    if (curr == nullptr) {
        head_leaderboard_entry = new_entry;
        head_leaderboard_entry->next_leaderboard_entry = nullptr;
        return;
    }
    while (curr != nullptr) {
        if (curr->score < new_entry->score) {
            new_entry->next_leaderboard_entry = curr;
            curr = new_entry;
            if (prev == nullptr) {
                head_leaderboard_entry = new_entry;
            } else {
                prev->next_leaderboard_entry = new_entry;
            }
            if (size > MAX_LEADERBOARD_SIZE) {
                delExtra();
            }
            return;
        }
        prev = curr;
        curr = curr->next_leaderboard_entry;
    }
    prev->next_leaderboard_entry = new_entry;
    new_entry->next_leaderboard_entry = nullptr;
    if (size > MAX_LEADERBOARD_SIZE) {
        delExtra();
    }

}
void Leaderboard::delExtra(){
    LeaderboardEntry *temp = head_leaderboard_entry;
    for (int i = 0; i < MAX_LEADERBOARD_SIZE - 1; i++) {
        temp = temp->next_leaderboard_entry;
    }
    delete temp->next_leaderboard_entry;
    temp->next_leaderboard_entry = nullptr;
    size--;
}
void Leaderboard::write_to_file(const string& filename) {
    // TODO: Write the latest leaderboard status to the given file in the format specified in the PA instructions
    fstream file(filename, ios::out);
    if (!file.is_open()) {
        cout << "File could not be opened" << endl;
        return;
    }
    LeaderboardEntry *curr = head_leaderboard_entry;
    while (curr != nullptr) {
        file << curr->score << " " << curr->last_played << " " << curr->player_name << endl;
        curr = curr->next_leaderboard_entry;
    }
    file.close();

}

void Leaderboard::read_from_file(const string& filename) {
    // TODO: Read the stored leaderboard status from the given file such that the "head_leaderboard_entry" member
    //       variable will point to the highest all-times score, and all other scores will be reachable from it
    //       via the "next_leaderboard_entry" member variable pointer.
    fstream file(filename);
    if (!file.is_open()) {
        return;
    }
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string name;
        int score;
        time_t time;
        ss >> score >> time >> name;
        insert_new_entry(new LeaderboardEntry(score, time, name));
    }
    file.close();
}


void Leaderboard::print_leaderboard() {
    cout << "Leaderboard" << endl;
    cout << "-----------" << endl;
    LeaderboardEntry *curr = head_leaderboard_entry;
    int i = 1;
    while (curr != nullptr) {
    
        // Convert time to struct tm
        std::tm* timeInfo = std::localtime(&curr->last_played);

        // Format the timestamp
        std::stringstream ss;
        ss << std::put_time(timeInfo, "%H:%M:%S/%d.%m.%Y");
        std::string formattedTime = ss.str();

        cout << i << ". " << curr->player_name << " " << curr->score << " " << formattedTime << endl;
        curr = curr->next_leaderboard_entry;
        i++;
    }
    // TODO: Print the current leaderboard status to the standard output in the format specified in the PA instructions
}

Leaderboard::~Leaderboard() {
    // TODO: Free dynamically allocated memory used for storing leaderboard entries
    while (head_leaderboard_entry != nullptr) {
        LeaderboardEntry* temp = head_leaderboard_entry;
        head_leaderboard_entry = head_leaderboard_entry->next_leaderboard_entry;
        delete temp;
    }
}
