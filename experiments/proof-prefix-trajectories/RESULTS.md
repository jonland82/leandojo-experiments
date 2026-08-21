# Results: proof-prefix trajectories

The prefix study supports a local semantic-diversification mechanism, with an
important qualification for global isolation. Across 300 proofs with at least
16 traced tactics, the median cosine to the proof's initial top-100 neighbors
fell from **0.714** at one tactic to **0.587** at 16 tactics. The fitted common
attenuation was **0.830**, and median residual energy outside the initial
neighbor span rose from **0.141** to **0.260**. The residual increased in 81.3%
of proofs.

This departure was associated with sparsification. Median global effective
neighborhood size fell from **3.44** to **1.90**, while the change in subspace
residual correlated **-0.637** with the change in effective neighborhood size.
Among the initial above-threshold neighbors, the median retained count fell
from **5** to **0**. At 16 tactics the proof had nevertheless acquired a median
of **1** new above-threshold neighbor, leaving a median global count of **2**.
Real proof growth therefore usually changes community, with incomplete
replacement of the broad neighborhood visible at the first tactic.

The repetition control prevents a simple causal reading. Repeating the first
tactic 16 times moved less far from the initial neighborhood: its median
initial-neighbor cosine was **0.665** and subspace residual **0.207**. But this
unnatural repetition found no new community and had median global effective
size **0**. Thus semantic diversification specifically explains departure from
the old community; global sparsity also depends on whether new neighbors replace
the old ones and on encoder sensitivity to repeated text.

Embedding reliability was high. Exact duplicate checkpoint-1 inputs had median
cosine **1.000** (minimum 0.9996), and re-embedded full proofs had median cosine
**0.99986** to their archived vectors (minimum 0.99942).

These results motivate a conditional geometric proposition rather than a
universal complexity law: orthogonal semantic extensions attenuate all
similarities to a fixed local community, and a positive threshold makes its
weighted effective size nonincreasing. The observed global isolation follows
when replacement by new neighbors is incomplete.
