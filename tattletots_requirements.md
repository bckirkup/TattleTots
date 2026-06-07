# TattleTots: Requirements Specification
## What the system must do, not how to build it

### 1. What TattleTots Is

A simulation engine for populations of information-processing agents ("Tots") that:
- Consume data streams and compress them, producing residuals
- Feed on each other's residuals, forming trophic hierarchies
- Compete for human attention by escalating reports
- Are born, develop, metabolize, reproduce, and die
- Evolve over generations through mutation and recombination

The engine is domain-agnostic. It knows about streams, agents, and users — not about fires, networks, or ships.

### 2. What It Must Accept (Inputs)

**From the domain environment, each time step:**
- One or more raw data streams (multivariate time series, any dimensionality)
- Ground truth labels (after the fact) indicating what was actually happening in the world
- User profiles: who is paying attention, what they care about, how much bandwidth they have

**At initialization:**
- A seed population of agent genomes (or rules for random generation)
- Configuration: population limits, energy scales, mutation rates, reproduction thresholds, development duration

### 3. What It Must Produce (Outputs)

**Each time step:**
- Reports: which agents escalated what signals to which users, with what confidence
- Residual streams: the unmodeled remainder from each agent's compression, available for other agents to consume
- Population state: who is alive, what they're connected to, how much energy they have

**Over time (for analysis):**
- Trophic topology history: how the network of who-feeds-on-whom changed over time
- Energy flow accounting: information yield and attention income per agent per step
- Demographic history: births, deaths, lineages, genome evolution
- Detection performance: true positives, false positives, missed events, detection latency (computed against ground truth)
- Cost accounting: total surveillance cost, response cost (from dispatches triggered by escalations), and damage cost (from missed or late detections)

### 4. Behavioral Requirements (What Must Emerge, Not Be Hardcoded)

#### 4.1 Trophic Self-Organization
Agents must be free to attach to any available stream (raw or residual). The trophic hierarchy — who feeds on whom — must emerge from agents choosing inputs that maximize their metabolic yield, not from assigned roles.

#### 4.2 Dual-Currency Survival
Every agent must maintain two energy reserves:
- **Information energy:** earned by compressing data (extracting structure from input streams), and by receiving subsidy from downstream agents that benefit from their residuals.
- **Attention energy:** earned by having their reports read and valued by human users.

An agent dies if either reserve hits zero. This means an agent that compresses well but never reports anything useful to a human eventually starves — and an agent that gets human attention but doesn't actually compress anything also starves.

#### 4.3 Trust-Based Attention Allocation
Users allocate their finite attention budget based on past performance. An agent that cried wolf loses trust and gets less attention. An agent that caught a real event gains trust. Trust must be asymmetric: hard to build, easy to destroy.

#### 4.4 Reproduction and Evolution
Agents above an energy threshold can reproduce. Offspring inherit the parent's genome with mutations (small random changes to model type, hyperparameters, input preferences, escalation threshold, target user). Sexual recombination (crossing two genomes) must also be possible.

#### 4.5 Whistleblowing
It must be possible for an agent to consume another agent's OUTPUT (not its residual) as its input stream, and detect inconsistencies between what that agent claims and what the ground truth or other agents show. This is a whistleblower: an agent whose food source is other agents' dishonesty.

#### 4.6 Domestication (Niche Construction)
It must be possible for a downstream agent to send a signal upstream along its trophic link — not just energy subsidy, but a hint about what kind of residual would be most useful. This allows higher-level agents to shape the compression behavior of their food sources.

### 5. What It Must NOT Do

- **Not hardcode trophic levels.** No agent is assigned to "Level 1" or "Level 3." Levels are measured, not assigned.
- **Not hardcode model types.** The genome specifies a model class, but the population should be able to contain a mix of different model types (statistical, ML, rule-based, whatever) competing in the same ecology.
- **Not assume a specific domain.** Streams are abstract multivariate time series. The engine doesn't know what the numbers mean.
- **Not optimize globally.** There is no central objective function. Each agent acts locally to maximize its own survival. Global performance is an emergent property.

### 6. Mathematical Properties That Must Be Preserved

These come from the theoretical work and simulations completed in Phase 0. The implementation must not violate them:

1. **Residual entropy decreases through the chain.** An agent's residual must have equal or lower structured variance than its input. (An agent cannot create information.)

2. **Chain depth is bounded by signal rank.** With K independent structured components and agents extracting k components each, the maximum useful chain depth is approximately ceil(K/k). Deeper chains should produce agents that starve.

3. **Branching topologies are more stable than linear chains.** Linear chains exhaust the signal; branching trees distribute it. The engine must allow branching (multiple agents consuming the same stream).

4. **The H-D-W equilibrium.** Under the right detection parameters, crude parasites should be driven extinct while sophisticated deceivers and whistleblowers coexist with honest agents. If the engine's incentive structure produces all-parasite or all-honest outcomes regardless of parameters, something is wrong.

5. **Domestication improves total yield only when signals overlap.** If two signal components are perfectly orthogonal, downward feedback from a higher agent should have no effect. If they share feature space, feedback should improve total system information extraction.

6. **Attention is zero-sum across agents for each user.** More attention to agent A means less to agent B. This is a hard constraint, not a soft penalty.

### 7. Interface to Domain Repositories

The domain repository (e.g., FireEcology, CruiseEcology) is responsible for:

1. **Generating data streams** each time step from its simulated or real environment
2. **Providing ground truth** (what actually happened) for verification of agent reports
3. **Defining user profiles** (attention budgets, priority vectors)
4. **Scoring relevance** (how well a given report matches a given user's priorities — this is domain-specific)
5. **Computing domain costs** (surveillance, response, damage — these depend on the domain's physics)

TattleTots provides:
1. **Escalation events** (what agents reported to which users)
2. **Population telemetry** (who's alive, what topology, energy states)
3. **Hooks** for the domain to inject new streams or remove failed sensors

### 8. Testing and Validation

The engine should ship with at least one built-in synthetic test scenario (no domain repo needed):
- A multivariate Gaussian stream with K=10 structured components, Gaussian noise, and a distribution shift at the midpoint.
- 2 synthetic users with different priority vectors.
- Expected behavior: trophic hierarchy forms, agents specialize, shift triggers partial extinction and re-colonization, detection of the shift is escalated.

This is the "smoke test" that confirms the engine works before plugging in a real domain.

### 9. What We Will Measure (Success Criteria for the Engine Itself)

Before any domain application, the engine passes if:
1. Trophic hierarchies of depth > 2 emerge spontaneously from a random seed population
2. Agent population reaches a stable equilibrium (births ≈ deaths) within a reasonable number of steps
3. Removing basal streams causes upstream extinction cascades
4. False-alarm agents lose trust and die; accurate agents gain trust and reproduce
5. At least two distinct "species" (genome clusters) coexist at equilibrium
