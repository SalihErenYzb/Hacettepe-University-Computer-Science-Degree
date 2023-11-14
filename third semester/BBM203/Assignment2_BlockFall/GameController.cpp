#include "GameController.h"
#include <ctime>
#include <chrono>
#include <thread>
#include "cmdlib.h"
bool GameController::checkIfCollision(int y,BlockFall& game){
    if (game.active_rotation == nullptr) {
        return false;
    }
    vector<vector<bool>> x = game.active_rotation->shape;
    if (y+x[0].size() > game.grid[0].size() || y < 0) {
        return true;
    }
    for (int i = 0; i < game.grid.size(); i++) {
        for (int j = 0; j < game.grid[i].size(); j++) {
            int k = j -y;
            
            if (k < 0 || k >= x[0].size() || i >= x.size() ) {
                continue;
            }
            if (x[i][k] == 1 && game.grid[i][j] == 1) {
                return true;
            }
                
        }
    }
    return false;
}
void GameController::rotate_Right(BlockFall& game){
    game.active_rotation = game.active_rotation->right_rotation;
    if (checkIfCollision(game.y, game)) {
        game.active_rotation = game.active_rotation->left_rotation;
    }
}
void GameController::rotate_Left(BlockFall& game){
    game.active_rotation = game.active_rotation->left_rotation;
    if (checkIfCollision(game.y, game)) {
        game.active_rotation = game.active_rotation->right_rotation;
    }
}
void GameController::move_Left(BlockFall& game){

    game.y--;
    if (checkIfCollision(game.y, game)) {
        game.y++;
    }
}
void GameController::move_Right(BlockFall& game){
    game.y++;
    if (checkIfCollision(game.y, game)) {
        game.y--;
    }
}
int GameController::howMuchDown(BlockFall& game,int x1, int y1){
    int mainY = y1 + game.y;
    int ans = 0;
    for (int i = x1; i < game.grid.size(); i++) {
        if (game.grid[i][mainY] == 1) {
            break;
        }
        ans++;
    }
    return ans-1;
}
int GameController::howMuchDownForShape(BlockFall& game){
    vector<vector<bool>> x = game.active_rotation->shape;
    int ans = 1000000;
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[i].size(); j++) {
            if (x[i][j] == 1) {
                ans = min(ans, howMuchDown(game, i, j));
            }
        }
    }
    return ans;
}
int GameController::dropForGravity(BlockFall& game){// This only drops it for now
    vector<vector<bool>> x = game.active_rotation->shape;
    int ans = 1000000;
    int score2 = 0;
    int min1 = 10000;
    int count = 0;
    for (int i = x.size()-1; i >=0 ; i--) {
        for (int j = 0; j < x[i].size(); j++) {
            if (x[i][j] == 1) {
                ans = howMuchDown(game, i, j);
                game.grid[i+ans][j+game.y] = 1;
                count++;
                min1 = min(min1, ans);
            }
        }
    }
    return count*min1;
}
int GameController::dropForNormal(BlockFall& game){// This only drops it for now
    vector<vector<bool>> x = game.active_rotation->shape;
    int ans = howMuchDownForShape(game);
    int score = 0;
    for (int i = x.size()-1; i >=0 ; i--) {
        for (int j = 0; j < x[i].size(); j++) {
            if (x[i][j] == 1) {
                game.grid[i+ans][j+game.y] = 1;
                score += ans;
            }
        }
    }
    return score;
}
bool GameController::powerDetection(BlockFall& game){
    vector<vector<bool>> x = game.power_up;
    vector<vector<int>> y = game.grid;
    for (int i = 0; i <= y.size()-x.size(); i++) {
        for (int j = 0; j <= y[0].size()-x[0].size(); j++) {
            bool deneme = true;
            for (int k = 0; k < x.size(); k++) {
                for (int l = 0; l < x[0].size(); l++) {
                    if (x[k][l] != y[i+k][j+l] ) {
                        deneme = false;
                    }
                }
            }
            if (deneme) {
                return true;
            }
        }
    }
    return false;
}
int GameController::count(BlockFall& game){
    int ans = 0;
    for (int i = 0; i < game.grid.size(); i++) {
        for (int j = 0; j < game.grid[i].size(); j++) {
            if (game.grid[i][j] == 1) {
                ans++;
            }
        }
    }
    return ans;
}
int GameController::clear(BlockFall& game){
    int sum = 0;
    for (int i = 0; i < game.grid.size(); i++) {
        for (int j = 0; j < game.grid[i].size(); j++) {
            sum += game.grid[i][j];
            game.grid[i][j] = 0;
        }
    }
    return sum;
}
int GameController::clearRow(BlockFall& game, int row){
    int sum = 0;
    for (int i = row; i > 0; i--) {
        for (int j = 0; j < game.grid[i].size(); j++) {
            game.grid[i][j] = game.grid[i-1][j];
        }
    }
    sum = game.grid[0].size();
    for (int i = 0; i < game.grid[0].size(); i++) {
        game.grid[0][i] = 0;
    }
    return sum;
}
bool GameController::checkIfRowIsFull(BlockFall& game, int row){
    for (int i = 0; i < game.grid[row].size(); i++) {
        if (game.grid[row][i] == 0) {
            return false;
        }
    }
    return true;
}
void GameController::power_up(BlockFall& game){


    int sum = clear(game);
    game.current_score += sum+1000;
}
void GameController::gravitySwitch(BlockFall& game){
    game.gravity_mode_on = !game.gravity_mode_on;
    if (game.gravity_mode_on) {
        //make everything fall
        for (int i = game.grid.size()-1; i >= 0; i--) {
            for (int j = 0; j < game.grid[i].size(); j++) {
                if (game.grid[i][j] == 1) {
                    int k = i;
                    while (k < game.grid.size()-1 && game.grid[k+1][j] == 0) {
                        game.grid[k+1][j] = 1;
                        game.grid[k][j] = 0;
                        k++;
                    }
                }
            }
        }
        for (int i = 0; i < game.grid.size(); i++) {
            if (checkIfRowIsFull(game, i)) {

                game.current_score += clearRow(game, i);
            }
        }
    }
}
void GameController::drop(BlockFall& game){
    if (game.gravity_mode_on) {
        game.current_score += dropForGravity(game);
    }else{
        game.current_score += dropForNormal(game);
    }
    if (powerDetection(game)) {
        power_up(game);
    }
    bool isClear = true;
    for (int i = 0; i < game.grid.size(); i++) {
        if (checkIfRowIsFull(game, i)) {
            if (isClear){
        

                isClear = false;
            }

            game.current_score += clearRow(game, i);
        }
    }
    game.active_rotation = game.active_rotation->next_block;
    game.y = 0;
    if (game.active_rotation == nullptr) {
        game.gameOver = 1;
        return;
    }
    if (checkIfCollision(game.y, game)) {
        game.gameOver = 2;
        return;
    }
}
bool GameController::play(BlockFall& game, const string& commands_file){
        clearScreen();
        print_grid(game);


    // TODO: Implement the gameplay here while reading the commands from the input file given as the 3rd command-line
    //       argument. The return value represents if the gameplay was successful or not: false if game over,
    //       true otherwise.
    fstream file(commands_file);
    string line;
    bool first = true;
    goToMidX();

    while (game.gameOver == 0 && cin >> line  ) {

        clearScreen();
        if (line == "RR") {
            rotate_Right(game);
        } else if (line == "RL") {
            rotate_Left(game);
        } else if (line == "L") {
            move_Left(game);
        } else if (line == "R") {
            move_Right(game);
        } else if (line == "D") {
            cout << "wtf";
            drop(game);
        } else if (line == "P") {
            power_up(game);
        } else if (line == "G") {
            gravitySwitch(game);
        }else if(line == "PRINT_GRID"){

        }
        else{
        }
        

        if (checkIfCollision(game.y, game)) {
            game.gameOver = 2;
        }else{

            print_grid(game);
        }

    goToMidX();

    }

        clearScreen();
        bool ans = false;
        if (game.gameOver == 1) {
            string ck1 = "YOU WIN!",ck2 ="No more blocks.",ck3="Final grid and score:";
            printt(ck1,ck1.size(),true);
            printt(ck2,ck2.size(),true);
            printt(ck3,ck3.size(),true);
            ans = true;
        }else if (game.gameOver == 0){
            game.gameOver = 3;
            string ck1 = "GAME FINISHED!",ck2 ="No more commands.",ck3="Final grid and score:";
            printt(ck1,ck1.size(),true);
            printt(ck2,ck2.size(),true);
            printt(ck3,ck3.size(),true);
        }else{
            string ck1 = "GAME OVER!",ck2 ="Next block that couldn't fit:",ck3="Final grid and score:";
            printt(ck1,ck1.size(),true);
            printt(ck2,ck2.size()-1,true);
            vector<vector<bool>> x1= game.active_rotation->shape;

            for (int i = 0; i < x1.size(); i++) {
                    goToMidX(2*x1[0].size(),2);

                for (int j = 0; j < x1[i].size(); j++) {
                    if (x1[i][j] == 1 ) {
                        cout << occupiedCellChar;
                    } else {
                        cout << unoccupiedCellChar;
                    }
                }  
                cout << endl;


            }
            printt(ck3,ck3.size(),true);


        }   

    

    game.leaderboard.insert_new_entry(new LeaderboardEntry(game.current_score, std::time(nullptr), game.player_name));
    print_grid(game,true);
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    clearScreen();
    game.leaderboard.print_leaderboard();
    game.leaderboard.write_to_file("leaderboard2.txt");
    return ans;

}
void GameController::print_grid_dull(BlockFall& game){
        goToMidX(2*game.cols+4,2);

        for (int i = 0; i < game.cols+2; i++) {
            cout << occupiedCellChar;


        }
        goToMidX(2*game.cols+4,2);

    for (int i = 0; i < game.rows; i++) {
                    goToMidX(2*game.cols+4,2);
        cout << occupiedCellChar;
        for (int j = 0; j < game.cols; j++) {
            if (game.grid[i][j] == 1 ) {
                cout << occupiedCellChar;
            } else {
                cout << unoccupiedCellChar;
            }
        
        }
                cout << occupiedCellChar;
            cout << endl;
    }
        goToMidX(2*game.cols+4,2);

        for (int i = 0; i < game.cols+2; i++) {
            cout << occupiedCellChar;


        }
        cout << endl;
    }
void GameController::print_grid_dull(BlockFall& game,bool isShape){
    if (!isShape) {
       print_grid_dull(game);
    }else{
        vector<vector<bool>> x = game.active_rotation->shape;
        goToMidX(2*game.cols+4,2);

        for (int i = 0; i < game.cols+2; i++) {
            cout << occupiedCellChar;


        }
        cout << endl;
        for (int i = 0; i < game.rows; i++) {
                    goToMidX(2*game.cols+4,2);
            cout << occupiedCellChar;
            for (int j = 0; j < game.cols; j++) {
                int k = j -game.y;
                int z = 0;
                if (k >= 0 && k < x[0].size() && i < x.size() ) {
                    z = x[i][k];
                }
                if (game.grid[i][j] == 1 || z == 1 ) {
                    cout << occupiedCellChar;
                } else {
                    cout << unoccupiedCellChar;
                }
            }
            cout << occupiedCellChar;
            cout << endl;
        }
            goToMidX(2*game.cols+4,2);

        for (int i = 0; i < game.cols+2; i++) {
            cout << occupiedCellChar;


        }
        cout << endl;
}
}
void GameController::print_grid(BlockFall& game,bool deneme){
    if (deneme){
        cout << endl;
    }else{
        goToMidY(game.rows+2,4);
    }
    std::string s = "Score: " + std::to_string(game.current_score);

    printt(s,2*game.cols+4,true);

    if (game.leaderboard.head_leaderboard_entry == nullptr) {
        int high = game.current_score;

        std::string d = "Highest Score: " + std::to_string(high);
        printt(d, 2*game.cols+4, true);
    } else {
        int high = max(game.current_score, game.leaderboard.head_leaderboard_entry->score);

        std::string d = "Highest Score: " + std::to_string(high);
        printt(d, 2*game.cols+4, true);
    }
    if (game.gravity_mode_on){
        std::string d = "Gravity Mode: ON";
        printt(d, 2*game.cols+4, true);
    }else{
        std::string d = "Gravity Mode: OFF";
        printt(d, 2*game.cols+4, true);
    }
    if (game.active_rotation == nullptr || checkIfCollision(game.y, game) || game.gameOver  !=0 ) {
        print_grid_dull(game,false);
    }else {
        print_grid_dull(game,true);
    }



}


