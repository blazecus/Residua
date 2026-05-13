#include "movement_state.h"

void MovementStateBuffer::push(const MovementState& s)
{
    states[head] = s;
    head = (head + 1) % STATE_HISTORY_LEN;
    if (count < STATE_HISTORY_LEN) ++count;
}

const MovementState& MovementStateBuffer::at(uint32_t n) const
{
    uint32_t clamped = (count > 0 && n < count) ? n : (count > 0 ? count - 1 : 0);
    uint32_t idx = (head + STATE_HISTORY_LEN - 1 - clamped) % STATE_HISTORY_LEN;
    return states[idx];
}
