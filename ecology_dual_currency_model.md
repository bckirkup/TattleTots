# Dual-Currency Information Ecology: Internal Metabolism + External Attention Economy

## 1. The Two Currencies

An information ecosystem operates on two fundamentally different resource currencies:

### Currency 1: Information (Internal, Biogeochemical)
- **Analogous to:** Carbon, nitrogen, phosphorus, electrons in biological ecosystems
- **Source:** Raw data streams from the observed world
- **Flow:** Compressed through trophic layers; residuals passed upward
- **Conservation law:** Total information content is conserved (Shannon entropy of source)
- **Recycling:** Dead agents' models and parameters return to the gene pool
- **Limiting factor:** Signal rank (finite structured components in the source)

### Currency 2: Attention (External, Solar)
- **Analogous to:** Solar radiation — the primary energy input from outside the ecosystem
- **Source:** Human stakeholders who allocate cognitive bandwidth
- **Flow:** Top-down from humans to the agents that earn their engagement
- **Conservation law:** Zero-sum across agents: $\sum_i \alpha_i(t) \leq A_k(t)$ for user $k$
- **Not recycled:** Spent attention is gone; trust is the only persistent asset
- **Limiting factor:** Human cognitive bandwidth, willingness to engage

### Why Both Are Needed

Information metabolism determines what the ecosystem *can* do (compress, detect, predict).
Attention economics determines what the ecosystem *will* do (which signals get escalated, which agents survive).

A system with excellent information metabolism but no attention income starves.
A system with abundant attention but poor metabolism wastes human time.

The coupling is:
$$\text{Attention allocated} = f(\text{past verified value of reports})$$
$$\text{Agent survival} = g(\text{information yield}, \text{attention income})$$

## 2. Formal Model

### 2.1 Users and Attention Budgets

Let there be $K$ human users, each with:
- Attention budget $A_k(t)$ (cognitive bandwidth available at time $t$)
- Priority vector $\mathbf{p}_k \in \mathbb{R}^D$ (what topics/threats they care about)
- Trust state $\tau_{k,i}(t) \in [0,1]$ for each agent $i$ (updated by verification)

### 2.2 Agent Energy Dynamics (Dual-Currency)

Agent $i$ has two energy reserves:

**Information energy** (internal metabolism):
$$\frac{d\, e_i^{info}}{dt} = \underbrace{E_i(t)}_{\text{compression yield}} + \underbrace{\rho_i(t)}_{\text{downstream subsidy}} - \underbrace{c_i^{compute}}_{\text{processing cost}}$$

**Attention energy** (external funding):
$$\frac{d\, e_i^{attn}}{dt} = \sum_{k=1}^{K} \underbrace{\alpha_{k,i}(t)}_{\text{attention from user } k} \cdot \underbrace{v_{k,i}(t)}_{\text{verified value}} - \underbrace{c_i^{maint}}_{\text{maintenance cost}} - \underbrace{\pi_i(t)}_{\text{false alarm penalty}}$$

**Survival condition:** Agent $i$ persists if BOTH reserves are positive:
$$e_i^{info}(t) > 0 \quad \text{AND} \quad e_i^{attn}(t) > 0$$

An agent with excellent information yield but no attention income is a "tree falling in the forest" — correct but unfunded.
An agent with high attention income but no information yield is a parasite consuming human bandwidth.

### 2.3 Attention Allocation

Each user $k$ allocates attention according to a softmax over agent relevance:
$$\alpha_{k,i}(t) = A_k(t) \cdot \frac{\tau_{k,i}(t) \cdot r_{k,i}(t)}{\sum_j \tau_{k,j}(t) \cdot r_{k,j}(t)}$$

where $r_{k,i}(t) = \mathbf{p}_k \cdot \mathbf{s}_i(t)$ is the relevance of agent $i$'s current signal to user $k$'s priorities.

### 2.4 Trust Dynamics

Trust is the persistent asset. It updates on verified outcomes:
$$\tau_{k,i}(t+1) = \tau_{k,i}(t) + \beta \cdot \begin{cases}
+\Delta^{+} & \text{if agent } i \text{ escalated and was correct} \\
-\Delta^{-} & \text{if agent } i \text{ escalated and was wrong} \\
-\Delta^{miss} & \text{if agent } i \text{ failed to escalate a true event} \\
0 & \text{otherwise}
\end{cases}$$

with $\Delta^{-} \gg \Delta^{+}$ (trust is hard to build, easy to destroy).

### 2.5 The Alarm Asymmetry (Emergent, Not Imposed)

In the original document, the alarm asymmetry was a designed feature ($C_e$ as a fixed escalation cost). In the attention-economy model, it **emerges** from the trust dynamics:

- False alarm → trust drops → future attention allocation drops → agent starves
- Missed event → trust drops (but less, because the user may not know)
- Correct alarm → trust rises → more attention → more resources

The system evolves precision-oriented behavior because imprecise reporters literally lose their funding. No penalty parameter needs to be tuned.

## 3. Multi-User Niche Partitioning

With $K$ users having different priority vectors $\mathbf{p}_k$, the attention landscape becomes multi-dimensional:

**Specialist agents** align their signal vector $\mathbf{s}_i$ with a single user's priorities. High yield from one source, vulnerable to that user disengaging.

**Generalist agents** produce signals relevant to multiple users. Lower per-user yield but diversified funding.

**Broker agents** (a new trophic role) don't compress data themselves — they translate between the information ecology's internal representations and specific users' priority languages. They consume other agents' compressed outputs and repackage them for human consumption. This is the "reporter looking up at the boss" — the trophic level that interfaces between the ecology and its energy source.

### Niche Overlap and Competition

Two agents targeting the same user with similar signals compete directly for attention:
$$\text{Competition}_{ij}^{(k)} = \frac{\mathbf{s}_i \cdot \mathbf{s}_j}{|\mathbf{s}_i||\mathbf{s}_j|} \cdot \mathbb{1}[\text{both serve user } k]$$

High overlap → zero-sum attention competition → one agent displaced (competitive exclusion).
Low overlap → coexistence in different attention niches.

## 4. System Viability and Metabolic Pivoting

The total attention income for the ecosystem:
$$A_{total}^{in}(t) = \sum_{k=1}^K \sum_{i=1}^N \alpha_{k,i}(t) \cdot v_{k,i}(t)$$

The total operating cost:
$$C_{total}(t) = \sum_{i=1}^N c_i^{maint} + c_i^{compute}$$

**System viability condition:**
$$A_{total}^{in}(t) > C_{total}(t)$$

When this condition fails, the ecosystem must either:
1. **Shed agents** (downsize) until costs match income
2. **Find new users** (market expansion)
3. **Pivot to alternative revenue** — selling compressed data products, generating training datasets, producing compliance documentation
4. **Hibernate** — reduce to a minimal viable ecology that can restart when attention returns

Option 3 is fascinating because it's a metabolic mode switch: from attention-funded surveillance to product-funded data services. The trophic structure persists but the terminal energy source changes.

## 5. Connections to Human Organizations

This model describes many semi-agentic human systems:

| System | Information Currency | Attention Currency | Parasitism | Whistleblowing |
|--------|---------------------|-------------------|------------|----------------|
| Newsroom | Events → stories | Readership | Clickbait | Fact-checking |
| Intelligence agency | SIGINT → analysis | Policymaker time | Threat inflation | Red teams |
| Hospital | Vitals → diagnosis | Physician cognition | Alert fatigue generators | Clinical pharmacists |
| Research group | Data → papers | Grant funding | p-hacking | Peer review |
| Corporation | Market data → strategy | Board/investor attention | Empire building | Internal audit |
| Military C2 | Sensor data → COP | Commander's decision bandwidth | CYA reporting | Inspector general |

The same replicator dynamics (H-D-W equilibrium), the same trophic depth constraints, the same domestication and niche construction mechanics apply across all these systems. The BMA formalism may be capturing something fundamental about how information-processing hierarchies work under attention scarcity.

## 6. Implications for Simulation Design

The dual-currency model changes the simulation architecture:

1. **Two fitness dimensions:** Agents must track both information energy and attention energy.
2. **User agents:** Human users are modeled as external entities with budgets, priorities, and trust states.
3. **Escalation as trophic interface:** The boundary between the ecology and its users is a trophic level, not just a threshold.
4. **Market dynamics:** Multiple users create a market for agent attention, with prices set by relevance and trust.
5. **Metabolic pivoting:** The simulation should allow the ecosystem to switch funding sources.

The wildfire monitoring track is ideal for testing this: multiple park managers with different zones of responsibility, a limited fire-response budget (the "attention" analog), and agents competing to be the one whose alert triggers a dispatch.
