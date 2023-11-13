#include <iostream>
#include <ctime>
#include "BlockFall.h"
#include <fstream>
#include <sstream>
#include "GameController.h"
#include <AL/al.h>
#include <AL/alc.h>
using namespace std;


int main() {
    // Initialize OpenAL
    ALCdevice* device = alcOpenDevice(nullptr);
    if (!device) {
        // Handle initialization error
        return 1;
    }

    ALCcontext* context = alcCreateContext(device, nullptr);
    if (!context) {
        // Handle initialization error
        alcCloseDevice(device);
        return 1;
    }

alcMakeContextCurrent(context);

    // Play the audio file
    
    Mix_PlayMusic(music, -1);  // -1 plays the music indefinitely



    return 0;
}