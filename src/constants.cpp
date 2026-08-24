#include "../headers/general.h"
#include <iomanip>
#include <chrono>
#include <ctime>

std::string get_time_current(){
    auto now =std::chrono::system_clock::now();
    std::time_t now_time= std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);
    std::ostringstream oss;
    oss<<std::put_time(local_time,"%Y_%m_%d_%H_%M_%S");
    return oss.str();
}
