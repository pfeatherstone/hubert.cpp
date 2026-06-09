#pragma once

#include <span>
#include <memory>

namespace hubert
{

//----------------------------------------------------------------------------------------------------------------

    class model
    {
    private:
        struct impl;
        std::unique_ptr<impl> state;
        
    public:
        model();
        ~model();
        model(model&& other);
        model& operator=(model&& other);

        std::span<const float> encode(std::span<const float> audio);
    };

//----------------------------------------------------------------------------------------------------------------

}