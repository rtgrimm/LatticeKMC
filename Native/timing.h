#pragma once

#include <string>
#include <iostream>
#include <chrono>

namespace Nano {
    namespace Util {
        void log(const std::string& tag, const std::string& message) {
            std::cout << message << std::endl;
        }


        class Timing {
        public:
            Timing(double percent_size) : _percent_size(percent_size) {}

            void start_step() {

            }

            void end_step() {

            }

        private:


            double _percent_size = 1.0;
        };
    }
}