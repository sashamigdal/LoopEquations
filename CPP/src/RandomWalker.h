#pragma once

#include <random>

class RandomWalker
{
public:
    RandomWalker(std::int64_t N_pos, std::int64_t N_neg) : N_pos(N_pos), N_neg(N_neg), alpha(), gen(std::random_device{}()), unif(0, 1) {}

    int Advance()
    {
        int sigma = RandomSign();
        (sigma == 1 ? N_pos : N_neg)--;
        alpha += sigma;
        return sigma;
    }

    std::int64_t get_alpha() const { return alpha; }

private:
    int RandomSign()
    {
        return (unif(gen) * double(N_pos + N_neg) <= N_neg) ? -1 : 1;
    }

    std::int64_t N_pos;
    std::int64_t N_neg;
    std::int64_t alpha; // alpha[i]
    std::minstd_rand gen;
    std::uniform_real_distribution<double> unif;
};
