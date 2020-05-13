#include <stdio.h>
#include <unistd.h>
#include <ctype.h>
#include <stdlib.h>
#include <wiringPi.h>

void printHelp()
{
    printf("Usage : <app_name> <msec for vibration> <msec for times>\n");
}


int isStringDigit(const char * str)
{
    char * add = (char * )str;

    while(*add)
    {
        if('0' <= *add && *add <= '9')
        {
            add++;
        }
        else
        {
            return 0;
        }
    }

    return 1;
}

#define MOTOR 4

int main(int argc, char ** argv)
{
    int times;
    int delay;
    int i,j;

    if(argc != 3)
    {
        printHelp();
        return 1;
    }

    if(isStringDigit(argv[1]) == 0)
    {
        printHelp();
        return 1;
    }

    if (wiringPiSetup () == -1) return 1;
  
    delay = atoi(argv[1]);
    times = atoi(argv[2]);


    for(j = 0 ; j < times ; j++)
    {
        pinMode (MOTOR, OUTPUT);
        digitalWrite (MOTOR, 0);

        // vib
        for(i = 0; i < delay ; i++)
        {
            usleep(1000);
        }

        digitalWrite (MOTOR, 1);

        usleep(150000);
    }
    

    return 0;
}