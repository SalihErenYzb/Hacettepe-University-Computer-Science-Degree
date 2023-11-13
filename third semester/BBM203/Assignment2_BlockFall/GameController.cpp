#include "GameController.h"
#include <ctime>
bool GameController::checkIfCollision(int y,BlockFall& game){
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
    cout << "Before clearing:" << endl;
    print_grid_dull(game,false);
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
                cout << "Before clearing:" << endl;
                print_grid_dull(game,false);
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


    // TODO: Implement the gameplay here while reading the commands from the input file given as the 3rd command-line
    //       argument. The return value represents if the gameplay was successful or not: false if game over,
    //       true otherwise.
    fstream file(commands_file);
    string line;
    bool first = true;
    while (game.gameOver == 0 && getline(file, line) ) {

        line = line.substr(0,line.size()-1);

        if (line == "ROTATE_RIGHT") {
            rotate_Right(game);
        } else if (line == "ROTATE_LEFT") {
            rotate_Left(game);
        } else if (line == "MOVE_LEFT") {
            move_Left(game);
        } else if (line == "MOVE_RIGHT") {
            move_Right(game);
        } else if (line == "DROP") {
            drop(game);
        } else if (line == "POWER_UP") {
            power_up(game);
        } else if (line == "GRAVITY_SWITCH") {
            gravitySwitch(game);
        }else if(line == "PRINT_GRID"){
            if (first) {
                first = false;
            }else{
                cout << endl;
            }

            print_grid(game);
        }
        else{
            cout << "Unknown command: " << line << endl;
        }

    }
    cout << endl;

    bool ans = false;
    if (game.gameOver == 1) {
        cout << "YOU WIN!\nNo more blocks.\nFinal grid and score:\n" << endl;
        ans = true;
    }else if (game.gameOver == 0){
        game.gameOver = 3;
        cout << "GAME FINISHED!\nNo more commands.\nFinal grid and score:\n" << endl;
    }else{
        cout << "GAME OVER!\nNext block that couldn't fit:" << endl;
        vector<vector<bool>> x = game.active_rotation->shape;
        for (int i = 0; i < x.size(); i++) {
            for (int j = 0; j < x[i].size(); j++) {
                if (x[i][j] == 1) {
                    cout << occupiedCellChar;
                }else{
                    cout << unoccupiedCellChar;
                }
            }
            cout << endl;
        }
        cout << "\nFinal grid and score:\n" << endl;
    }
    game.leaderboard.insert_new_entry(new LeaderboardEntry(game.current_score, std::time(nullptr), game.player_name));
    print_grid(game);
    game.leaderboard.print_leaderboard();
    game.leaderboard.write_to_file("leaderboard2.txt");
    return ans;

}
void GameController::print_grid_dull(BlockFall& game){
    for (int i = 0; i < game.rows; i++) {
        for (int j = 0; j < game.cols; j++) {
            if (game.grid[i][j] == 1 ) {
                cout << occupiedCellChar;
            } else {
                cout << unoccupiedCellChar;
            }
        }
        cout << endl;
    }}
void GameController::print_grid_dull(BlockFall& game,bool isShape){
    if (!isShape) {
       print_grid_dull(game);
    }else{
        vector<vector<bool>> x = game.active_rotation->shape;

        for (int i = 0; i < game.rows; i++) {
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
            cout << endl;
        }
}
    cout << endl;
}
void GameController::print_grid(BlockFall& game){
    cout << "Score: " << game.current_score << endl;
    int high = max(game.current_score, game.leaderboard.head_leaderboard_entry->score);
    cout << "Highest Score: " << high << endl;
    if (game.active_rotation == nullptr || checkIfCollision(game.y, game) || game.gameOver  !=0 ) {
        print_grid_dull(game,false);
    }else {
        print_grid_dull(game,true);
    }
}


