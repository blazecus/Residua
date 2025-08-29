#include <iostream>
#include <src/game/game.h>

int main(int argc, char* argv[])
{

    Game game;

    game.init();

    game.run();

    game.end();

    return 0;
}
