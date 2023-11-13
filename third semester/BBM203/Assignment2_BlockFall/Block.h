#ifndef PA2_BLOCK_H
#define PA2_BLOCK_H

#include <vector>
#include <iostream>
using namespace std;

class Block {
public: 
    vector<vector<bool>> shape; // Two-dimensional vector corresponding to the block's shape
    Block * right_rotation = nullptr; // Pointer to the block's clockwise neighbor block (its right rotation)
    Block * left_rotation = nullptr; // Pointer to the block's counter-clockwise neighbor block (its left rotation)
    Block * next_block = nullptr; // Pointer to the next block to appear in the game


        bool operator==(const Block& other) const {
            if (other.shape.size() != shape.size()) {
                return false;
            }else if (other.shape[0].size() != shape[0].size()){
                return false;
            } else {
                for (int i = 0; i < shape.size(); i++) {
                    for (int j = 0; j < shape[0].size(); j++) {
                        if (other.shape[i][j] != shape[i][j]) {
                            return false;
                        }
                    }

                }
            }
            return true;
        }
        bool operator!=(const Block& other) const {
            if (other.shape == shape) {
                return false;
            }
            return true;
        }

        vector<vector<bool>> turnNightydeegre(const vector<vector<bool>> shape2){


            vector<vector<bool>> tmp;
            for (int i = (int)(shape2[0].size())-1; i !=-1 ; i--) {
                vector<bool> tmp2;
                for (int j = 0; j < shape2.size(); j++) {
                    int t,d ;
                    t = shape2.size()-1-j;
                    d = shape2[0].size()-1-i;

                        tmp2.push_back(shape2[t][d]);
                    
                }
                tmp.push_back(tmp2);
            }
            return tmp;
        }

        void fillTheCircular(){
            Block *head = this;
            Block *curr = head;
            vector<vector<bool>> tmp = turnNightydeegre(head->shape);

            for (int i = 0 ; i < 3 ; i++){


                if (!isItInDoubly(tmp)){
                    curr->right_rotation = new Block;
                    curr->right_rotation->shape = tmp;
                    curr->right_rotation->next_block = head->next_block;
                    curr->right_rotation->left_rotation = curr;
                    curr = curr->right_rotation;
                }
                tmp = turnNightydeegre(tmp);


            }

            curr->right_rotation = head;
            head->left_rotation = curr;

        }
private:
        bool isItInDoubly(const vector<vector<bool>> shapeToCheck){
            // Only works while building the circular doubly linked list
            Block *tmp = this;
            while (tmp != nullptr){
                if (tmp->shape == shapeToCheck){
                    return true;
                }
                tmp = tmp->right_rotation;
            }
            return false;
        }

};


#endif //PA2_BLOCK_H
