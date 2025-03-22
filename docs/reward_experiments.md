# Reward Function Experiments

This document tracks the evolution and performance of different reward function implementations for the multi-elevator environment.

## Performance Metrics
- **Average Movements**: Total number of elevator movements across runs
- **Average Waiting Time**: Average time passengers wait for pickup (in steps)

## Version History

### Version 1
**Reward Function:**
```python
reward = -(np.sum(self.waiting_passengers) + 
          sum(len(passengers) for passengers in self.passengers_in_elevators))
reward -= 1 * np.sum(np.array(action) != 0)  # Movement penalty

# Extra penalty for unnecessary movement
if (np.sum(self.waiting_passengers) == 0 and 
    sum(len(passengers) for passengers in self.passengers_in_elevators) == 0):
    reward -= 2 * np.sum(np.array(action) != 0)
```

**Results:**
- Average Movements: 272.4
- Average Waiting Time: 18.94 steps

### Version 2
**Key Changes:**
- Reduced movement penalty from 1.0 to 0.75

**Results:**
- Average Movements: 217.33
- Average Waiting Time: 15.89 steps

### Version 3
**Key Changes:**
- Increased extra penalty for unnecessary movement from 2.0 to 3.0

**Results:**
- Average Movements: 228.2
- Average Waiting Time: 21.48 steps

### Version 4
**Key Changes:**
- Removed movement penalties
- Focused purely on passenger waiting time

**Results:**
- Average Movements: 498.1
- Average Waiting Time: 8.22 steps

### Version 5
**Key Changes:**
- Introduced complex reward structure with multiple components:
  - Base reward (-0.1 per step)
  - Pickup rewards (up to 5.0)
  - Delivery rewards (up to 25.0)
  - Idle elevator penalties
  - Waiting time penalties

**Results:**
- Average Movements: 500.0
- Average Waiting Time: 8.65 steps

### Version 6
**Key Changes:**
- Refined reward structure:
  - Increased pickup rewards (up to 6.0)
  - Increased delivery rewards (up to 32.0)
  - Adjusted movement penalties (0.3 per movement)
  - Increased waiting time penalties

**Results:**
- Average Movements: 483.1
- Average Waiting Time: 8.51 steps

### Version 7
**Key Changes:**
- Increased movement penalty from 0.3 to 0.5

**Results:**
- Average Movements: 440.3
- Average Waiting Time: 11.97 steps

### Version 8
**Key Changes:**
- Introduced context-aware movement penalties:
  - 0.2 for purposeful movement (with passengers or moving to pickup)
  - 0.4 for speculative movement
- Added exponential decay to rewards
- Introduced strategic positioning rewards

**Results:**
- Average Movements: [To be added]
- Average Waiting Time: [To be added]

## Analysis

1. **Version 1-3**: Focused on simple penalties for movement and waiting time
2. **Version 4**: Achieved best waiting time but at cost of excessive movement
3. **Version 5-7**: Introduced more sophisticated reward structure with multiple components
4. **Version 8**: Added context awareness and strategic positioning

## Best Performing Version
Based on the balance between waiting time and movement efficiency, Version 6 appears to be the most effective, achieving:
- Good waiting time (8.51 steps)
- Reasonable movement count (483.1)
- Balanced reward structure 