#pragma once
#include <string>
#include <iostream>

namespace Nano {
    enum class MessageType {
        Info,
        Error
    };

    std::string_view message_type_to_message(MessageType type) {
        if(type == MessageType::Error) {
            return "Error";
        }

        if(type == MessageType::Info) {
            return "Info";
        }

        return "";
    }

#ifdef NANO_DEBUG_MODE
    void log(MessageType message_type, std::string_view message) {
        std::cout << "[" << message_type_to_message(message_type) << "] " << message << std::endl;
    }
#else
    void log(MessageType message_type, std::string_view message) {}
#endif
}