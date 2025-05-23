# Agents

Agents represent autonomous decision-making entities participating in grid-based simulations. 
They operate under defined constraints (e.g., budgets, limited knowledge) 
and use machine-learned strategies to optimize their outcomes.

---

### **VoteAgent Class**

Defined in: `participation_agent.py`

#### **Key Attributes**
- **`unique_id`**: An identifier for the agent.
- **`personality`**: A numpy array representing the agent's preferences among colors.
- **`assets`**: Represents personal resources or motivation; consumed when participating in elections.
- **`confidence`**: Confidence in estimating the true color distribution.
- **`known_cells`**: Knowledge about a number of cells.

#### **Key Methods**
1. **`ask_for_participation(area)`**
   - Decides whether to participate in a given area's election.
   - Returns `True` or `False`.

   ```python
   # TODO
   ```

2. **`decide_altruism_factor`**
   - Uses a trained decision tree to determine the altruism factor for voting.

   ```python
   # TODO
   ```

3. **`update_known_cells(area)`**
   - Updates the known cells by sampling the provided area.

4. **`vote(area)`**
   - Calculates the agent's preference ranking vector for the given area.

