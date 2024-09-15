# Problem of threshold in approval voting

If we choose an architecture in which voters always provide a sum-normalized preference vector
for all voting rules, then approval voting has to have a threshold value to determine which options are approved.
This may take autonomy away from the voters, but it ensures that every voting rule is based on the same conditions  
increasing comparability. It may also help to add more rules later on.

### Idea

Setting a fixed threshold of $ \frac{1}{m} $ for approval voting where m is the number of options.

### Definitions and Setup

- **Sum-normalized vector**: A preference vector $ \mathbf{p} = (p_1, p_2, \ldots, p_m) $ where each entry $ p_i $ represents the preference score for option $ i $, with the constraint $ \sum_{i=1}^m p_i = 1 $.
- **Threshold**: A fixed threshold of $ \frac{1}{m} $ is used to determine approval. If $ p_i \geq \frac{1}{m} $, the option $ i $ is considered "approved."

### Average Number of Approved Values

To find the average number of values approved, let's consider how many entries $ p_i $ would meet the threshold $ p_i \geq \frac{1}{m} $.

1. **Expectation Calculation**:
   - The expected number of approvals can be found by looking at the expected value of each $ p_i $ being greater than or equal to $ \frac{1}{m} $.
   - For a sum-normalized vector, the average value of any $ p_i $ is $ \frac{1}{m} $. This is because the sum of all entries equals 1, and there are $ m $ entries.

2. **Probability of Approval**:
   - If the vector entries are randomly distributed, the probability of any given $ p_i $ being above the threshold is approximately 50%. This stems from the fact that the mean is $ \frac{1}{m} $, and assuming a uniform or symmetric distribution around this mean, half the entries would be above, and half below, in expectation.

3. **Expected Number of Approvals**:
   - Since each entry has a 50% chance of being above $ \frac{1}{m} $ in a uniform random distribution, the expected number of approved values is $ \frac{m}{2} $.

Therefore, **on average, $ \frac{m}{2} $ values will be approved**.

### Range of the Number of Approved Values

The number of approved values can vary depending on how the preference scores are distributed. Here's the possible range:

1. **Minimum Approved Values**:
   - If all entries are below $ \frac{1}{m} $, then none would be approved. However, given the constraint that the vector sums to 1, at least one entry must be $ \frac{1}{m} $ or higher. Hence, the minimum number of approved values is **1**.

2. **Maximum Approved Values**:
   - The maximum occurs when as many values as possible are at least $ \frac{1}{m} $. In the extreme case, you could have all $ m $ entries equal $ \frac{1}{m} $ exactly, making them all approved. Thus, the maximum number of approved values is **m**.

### Conclusion

- **Average number of approved values**: $ \frac{m}{2} $.
- **Range of approved values**: From 1 (minimum) to $ m $ (maximum).

Hence, in theory, voters can still approve between 1 and $ m $ options, 
giving them the whole range of flexibility that approval voting offers.

### Possibility for improvement

We should consider implementing rule-specific voting into the agent's decision-making process
instead of leaving all rule-specifics to the aggregation process.
This would allow for a more realistic comparison of the rules.
For some rules, it would also give opportunities to significantly speed up the simulation process.